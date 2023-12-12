import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_embedding import load_word_embeddings
from .common import LabelSmoothingLoss,complex_similarity
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SEBlock(nn.Module):
    def __init__(self,dset, args, reduction_ratio=12):
        super(SEBlock, self).__init__()
        self.excitation = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(args.emb_dim // reduction_ratio, args.emb_dim),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(dset.feat_dim, dset.feat_dim * 2),
            nn.BatchNorm1d(dset.feat_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(dset.feat_dim * 2, args.emb_dim),
        )
    def forward(self, Xc):
        Xc = self.fc(Xc)
        out = self.excitation(Xc)
        return out

class AGV(nn.Module):
    def __init__(self, cin, dset, args):
        super(AGV,self).__init__()
        self.activation_head = nn.Conv2d(cin, args.emb_dim, kernel_size=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(args.emb_dim)
        self.args = args
        self.CH = SEBlock(dset,args)
        self.fc = nn.Sequential(
            nn.Linear(dset.feat_dim, dset.feat_dim * 2),
            nn.BatchNorm1d(dset.feat_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(dset.feat_dim * 2, self.args.emb_dim),
        )
    def forward(self,x,X_c):
        x = x.permute(0,2,1).view(-1,768,14,14)
        N, C, H, W = x.size()
        a = self.activation_head(x) #64 300 14 14
        cam = torch.sigmoid(self.bn_head(a))
        ccam_ = cam.reshape(N,self.args.emb_dim,H*W) #64 300 14*14
        x = x.reshape(N, C, H*W).permute(0, 2, 1).contiguous()
        fg_feats = torch.matmul(ccam_, x)/(H*W*self.args.emb_dim)
        fg_feats = fg_feats.sum(1)
        fg_feats = fg_feats.reshape(x.size(0), -1)
        fg_feats = self.fc(fg_feats)
        return F.normalize(self.CH(X_c)*fg_feats+fg_feats,dim=-1)

class OGA(nn.Module):
    def __init__(self, cin, args):
        super(OGA, self).__init__()
        self.filter = nn.Conv2d(cin, args.emb_dim, kernel_size=1, padding=0, bias=False)
        self.comp_decoder_embed = nn.Sequential(
            nn.Linear(args.emb_dim*2, args.emb_dim*4),
            nn.BatchNorm1d(args.emb_dim*4),
            nn.ReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(args.emb_dim*4,args.emb_dim),
        )
    def forward(self,x,p_o,omega,x_agv):
        bs,hw,ch = x.size()
        x = x.permute(0, 2, 1).view(-1, 768, 14, 14)
        barz = self.filter(x).view(bs,-1,hw).permute(0,2,1).unsqueeze(0) #128,12
        p_o = p_o.unsqueeze(1).unsqueeze(1)
        B = torch.sqrt(torch.mean((barz-p_o)**2,dim=2)) #12,128,300
        #print(B.shape,omega.shape)
        B = (omega.T.unsqueeze(2)*B).sum(0)
        x_a = F.normalize(self.comp_decoder_embed(torch.cat((x_agv,B),dim=-1)),dim=-1)
        return x_a

class IMAX(nn.Module):
    def __init__(self, dset, args):
        super(IMAX, self).__init__()
        self.args = args
        self.dset = dset
        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(device)
            objs = torch.LongTensor(objs).to(device)
            pairs = torch.LongTensor(pairs).to(device)
            return attrs, objs, pairs
        # Validation
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)

        # for indivual projections
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.scale_pairs = self.args.cosine_scale_pairs
        self.scale_c = self.args.cosine_scale_components
        if dset.open_world:
            self.train_forward = self.train_forward_closed
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask).to(device) * 1.
            self.activated = False
            # Init feasibility-related variables
            self.attrs = dset.attrs
            self.objs = dset.objs
            self.possible_pairs = dset.pairs
            self.validation_pairs = dset.val_pairs
            self.feasibility_margin = (1 - self.seen_mask).float()
            self.epoch_max_margin = self.args.epoch_max_margin
            self.cosine_margin_factor = -args.margin
            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.known_pairs:
                self.obj_by_attrs_train[a].append(o)
            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.known_pairs:
                self.attrs_by_obj_train[o].append(a)
        else:
            self.train_forward = self.train_forward_closed

        # Precompute training compositions
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]

        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(device), \
                                          torch.arange(len(self.dset.objs)).long().to(device)
        self.pairs = dset.pairs

        self.composition = args.composition
        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current])
            self.train_idx = torch.LongTensor(train_idx).to(device)

        self._build_smoothlabel()
        self._setup_word_composer(dset)
        self._setup_image_embedding(dset)
        self.label_smooth_loss = LabelSmoothingLoss(smoothing=0.2)
        self.t = args.cosine_scale_pairs

    def _build_smoothlabel(self):
        self.label_smooth = torch.zeros(self.num_pairs, self.num_pairs)
        for i in range(self.num_pairs):
            for j in range(self.num_pairs):
                if self.pairs[j][1] == self.pairs[i][1]:
                    self.label_smooth[i, j] = self.label_smooth[i, j] + 2
                if self.pairs[j][0] == self.pairs[i][0]:
                    self.label_smooth[i, j] = self.label_smooth[i, j] + 1
        self.label_smooth = self.label_smooth[:, self.train_idx]
        self.label_smooth = self.label_smooth[self.train_idx, :]

        K_1 = (self.label_smooth == 1).sum(dim=1)
        K_2 = (self.label_smooth == 2).sum(dim=1)
        K = K_1+2*K_2
        self.epi = self.args.smooth_factor
        template = torch.ones_like(self.label_smooth)/K
        template = template*self.epi
        self.label_smooth = self.label_smooth*template
        for i in range(self.label_smooth.shape[0]):
            self.label_smooth[i,i] = 1-(self.epi)
        self.label_smooth = self.label_smooth.cuda()

    def _setup_word_composer(self, dset):
        if self.args.emb_init == 'word2vec':
            self.word_vector_dim = 300
        elif self.args.emb_init == 'glove':
            self.word_vector_dim = 300
        elif self.args.emb_init == 'fasttext':
            self.word_vector_dim = 300
        else:
            self.word_vector_dim = 600

        self.attr_embedder = nn.Embedding(len(dset.attrs), self.word_vector_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), self.word_vector_dim)
        pretrained_weight = load_word_embeddings(self.args.emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)
        pretrained_weight = load_word_embeddings(self.args.emb_init, dset.objs)
        self.obj_embedder.weight.data.copy_(pretrained_weight)

        if self.args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        self.compose = nn.Sequential(
            nn.Linear(self.word_vector_dim * 2, self.word_vector_dim * 3),
            nn.BatchNorm1d(self.word_vector_dim * 3),
            nn.ReLU(0.1),
            nn.Linear(self.word_vector_dim * 3, self.word_vector_dim * 2),
            nn.BatchNorm1d(self.word_vector_dim * 2),
            nn.ReLU(0.1),
            nn.Linear(self.word_vector_dim * 2, self.args.emb_dim)
        )
        self.obj_projection = nn.Linear(self.word_vector_dim, self.args.emb_dim)
        self.attr_projection = nn.Linear(self.word_vector_dim, self.args.emb_dim)

        self.vp_a = nn.Linear(self.args.emb_dim, self.args.emb_dim)
        self.vp_o = nn.Linear(self.args.emb_dim, self.args.emb_dim)

    def _setup_image_embedding(self, dset):
        dset.feat_dim = 768
        self.image_embedder = nn.Sequential(
            nn.Linear(self.args.emb_dim*2, self.args.emb_dim*4),
            nn.BatchNorm1d(self.args.emb_dim*4),
            nn.ReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(self.args.emb_dim*4, self.args.emb_dim),
        )
        self.AGV_a = AGV(dset.feat_dim,dset,self.args)
        self.AGV_o = AGV(dset.feat_dim,dset,self.args)
        self.OGA = OGA(dset.feat_dim,self.args)

        self.cp_a = nn.Sequential(
            nn.Linear(self.args.emb_dim, self.args.emb_dim * 2),
            nn.BatchNorm1d(self.args.emb_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(self.args.emb_dim * 2, self.args.emb_dim),
        )
        self.cp_o = nn.Sequential(
            nn.Linear(self.args.emb_dim, self.args.emb_dim * 2),
            nn.BatchNorm1d(self.args.emb_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(self.args.emb_dim * 2, self.args.emb_dim),
        )
    def compose_word_embeddings(self, mode='train'):
        output_emb = dict()
        if mode == 'train':
            output_emb['attr'] = self.attr_embedder(self.uniq_attrs) # [n_pairs, word_dim].
            output_emb['obj'] = self.obj_embedder(self.uniq_objs) # # [n_pairs, word_dim].

            output_emb['attr'] = F.normalize(self.attr_projection(output_emb['attr']),dim=-1) # [n_pairs, word_dim].
            output_emb['obj'] = F.normalize(self.obj_projection(output_emb['obj']),dim=-1) # # [n_pairs, word_dim].

            attrs, objs = self.attr_embedder(self.train_attrs), self.obj_embedder(self.train_objs)
            output_emb['pair'] = torch.cat([attrs, objs], 1)
            output_emb['pair'] = self.compose(output_emb['pair'])

        elif mode == 'train_comp':
            output_emb['attr'] = self.attr_embedder(self.train_attrs) # [n_pairs, word_dim].
            output_emb['obj'] = self.obj_embedder(self.train_objs)
            print
            output_emb['attr'] = F.normalize(self.attr_projection(output_emb['attr']),dim=-1) # [n_pairs, word_dim].
            output_emb['obj'] = F.normalize(self.obj_projection(output_emb['obj']),dim=-1) # # [n_pairs, word_dim].

        elif mode == 'all':
            output_emb['attr'] = self.attr_embedder(self.val_attrs) # [n_pairs, word_dim].
            output_emb['obj'] = self.obj_embedder(self.val_objs)

            output_emb['attr'] = F.normalize(self.attr_projection(output_emb['attr']),dim=-1) # [n_pairs, word_dim].
            output_emb['obj'] = F.normalize(self.obj_projection(output_emb['obj']),dim=-1) # # [n_pairs, word_dim].

            attrs, objs = self.attr_embedder(self.val_attrs), self.obj_embedder(self.val_objs)
            output_emb['pair'] = torch.cat([attrs, objs], 1)
            output_emb['pair'] = self.compose(output_emb['pair'])

        return output_emb

    def cross_entropy(self, logits, label):
        logits = F.log_softmax(logits, dim=-1)
        loss = -(logits * label).sum(-1).mean()
        return loss

    def C_y_logits(self,img,attr,objs):
        text = self.C_y(attr,objs).permute(1,0)
        logits = torch.mm(img,text)
        return logits
    def val_forward(self, x):
        img = x[0]
        '''Real Space'''
        if self.args.image_extractor!='dino':
            X_c = F.avg_pool2d(img.detach(), kernel_size=7).view(-1, self.dset.feat_dim)
            X_p = img.detach()
        else:
            X_c = img[:,0,:].detach()
            X_p = img[:,1:,:].detach()

        word_embedding = self.compose_word_embeddings(mode='all')

        x_o = self.AGV_o(X_p,X_c)
        omega = F.softmax(x_o.detach()@word_embedding['obj'].T.detach(),dim=-1)

        x_agv = self.AGV_a(X_p,X_c)
        x_a = self.OGA(X_p,word_embedding['obj'],omega.detach(),x_agv)
        logit_attr = self.t*x_a@word_embedding['attr'].T
        logit_obj = self.t*x_o@word_embedding['obj'].T

        X = F.normalize(torch.cat((x_a, x_o), dim=-1), dim=-1)
        img_feats = self.image_embedder(X)
        img_feats = F.normalize(img_feats,dim=-1)
        logits_ds = img_feats@word_embedding['pair'].T

        X_c = torch.zeros((img.shape[0], 2, self.args.emb_dim)).cuda()
        X_attr_P = F.normalize(self.cp_a(x_a),dim=-1)
        X_obj_P = F.normalize(self.cp_o(x_o),dim=-1)
        X_c[:, 0], X_c[:, 1] = X_attr_P.clone(), X_obj_P.clone()

        text_complex = torch.zeros((word_embedding['attr'].shape[0], 2, self.args.emb_dim)).cuda()
        attr_ = self.vp_a(word_embedding['attr'])
        objs_ = self.vp_o(word_embedding['obj'])
        text_complex[:, 0], text_complex[:, 1] = objs_, attr_
        logits = self.t*complex_similarity(X_c, text_complex,'minos')

        score = logit_attr+logit_obj+logits+logits_ds

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]

        return None, scores

    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        targets = self.label_smooth[pairs]
        '''Real Space'''
        if self.args.image_extractor!='dino':
            X_c = F.avg_pool2d(img.detach(), kernel_size=7).view(-1, self.dset.feat_dim)
            X_p = img.detach()
        else:
            X_c = img[:,0,:].detach()
            X_p = img[:,1:,:].detach()

        word_embedding = self.compose_word_embeddings(mode='train')
        word_embedding_p = self.compose_word_embeddings(mode='train_comp')

        x_o = self.AGV_o(X_p,X_c)
        omega = F.softmax(x_o.detach()@word_embedding['obj'].T.detach(),dim=-1)

        x_agv = self.AGV_a(X_p,X_c)
        x_a = self.OGA(X_p,word_embedding['obj'],omega.detach(),x_agv)
        logit_attr = self.t*x_a@word_embedding['attr'].T
        logit_obj = self.t*x_o@word_embedding['obj'].T

        X = F.normalize(torch.cat((x_a,x_o),dim=-1),dim=-1)

        img_feats = self.image_embedder(X)
        img_feats = F.normalize(img_feats,dim=-1)
        logits_ds = img_feats@word_embedding['pair'].T

        loss_comp = self.cross_entropy(self.t * (logits_ds), targets)
        loss_attr = F.cross_entropy(self.t * logit_attr,attrs)
        loss_objs = F.cross_entropy(self.t * logit_obj, objs)
        loss_real = loss_attr+loss_objs+loss_comp

        '''Complex Space'''
        X_c = torch.zeros((img.shape[0], 2, self.args.emb_dim)).cuda()
        X_attr_P = F.normalize(self.cp_a(x_a),dim=-1)
        X_obj_P = F.normalize(self.cp_o(x_o),dim=-1)
        X_c[:, 0], X_c[:, 1] = X_attr_P.clone(), X_obj_P.clone()

        text_complex = torch.zeros((word_embedding_p['attr'].shape[0], 2, self.args.emb_dim)).cuda()
        attr_ = F.normalize(self.vp_a(word_embedding_p['attr']),dim=-1)
        objs_ = F.normalize(self.vp_o(word_embedding_p['obj']),dim=-1)
        text_complex[:, 0], text_complex[:, 1] = objs_, attr_
        logits = self.t*complex_similarity(X_c, text_complex,'minos')
        loss_cls = self.cross_entropy(logits, targets)

        loss = loss_real+loss_cls
        return loss, None

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred



