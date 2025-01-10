
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class Disentangler(nn.Module):
    def __init__(self, emd_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emd_dim, emd_dim)
        self.bn1_fc = nn.BatchNorm1d(emd_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q


class SEBlock(nn.Module):
    def __init__(self, emd_dim, reduction_ratio=12):
        super(SEBlock, self).__init__()
        self.excitation = nn.Sequential(
            nn.Linear(emd_dim, emd_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(emd_dim // reduction_ratio, emd_dim),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(emd_dim, emd_dim * 2),
            nn.BatchNorm1d(emd_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(emd_dim * 2, emd_dim),
        )
    def forward(self, Xc):
        Xc = self.fc(Xc)
        out = self.excitation(Xc)
        return out

class AGV(nn.Module):
    def __init__(self, emd_dim):
        super(AGV,self).__init__()
        self.activation_head = nn.Conv2d(emd_dim, emd_dim, kernel_size=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(emd_dim)
        self.CH = SEBlock(emd_dim)
        self.fc = nn.Sequential(
            nn.Linear(emd_dim, emd_dim * 2),
            nn.BatchNorm1d(emd_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(emd_dim * 2, emd_dim),
        )
        self.emd_dim = emd_dim
    def forward(self, X_c, x):
        x = x.permute(0,2,1).view(-1, 768, 16, 16)
        N, C, H, W = x.size()
        a = self.activation_head(x) #64 300 14 14
        cam = torch.sigmoid(self.bn_head(a))
        ccam_ = cam.reshape(N, self.emd_dim, H * W) #64 300 14*14
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()
        fg_feats = torch.matmul(ccam_, x) / (H * W * self.emd_dim)
        fg_feats = fg_feats.sum(1)
        fg_feats = fg_feats.reshape(x.size(0), -1)
        fg_feats = self.fc(fg_feats)
        return F.normalize(self.CH(X_c) * fg_feats + fg_feats, dim=-1)

class OGA(nn.Module):
    def __init__(self, emd_dim):
        super(OGA, self).__init__()
        self.filter = nn.Conv2d(emd_dim, emd_dim, kernel_size=1, padding=0, bias=False)
        self.comp_decoder_embed = nn.Sequential(
            nn.Linear(emd_dim * 2, emd_dim * 4),
            nn.BatchNorm1d(emd_dim * 4),
            nn.ReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(emd_dim * 4, emd_dim),
        )
    def forward(self, x, p_o, omega, x_agv):
        bs, hw, ch = x.size()
        x = x.permute(0, 2, 1).view(-1, 768, 16, 16)
        barz = self.filter(x).view(bs, -1, hw).permute(0, 2, 1).unsqueeze(0) #128,12
        p_o = p_o.unsqueeze(1).unsqueeze(1)
        B = torch.sqrt(torch.mean((barz - p_o)**2,dim=2)) #12,128,300
        B = (omega.T.unsqueeze(2) * B).sum(0)
        x_a = F.normalize(self.comp_decoder_embed(torch.cat((x_agv, B),dim=-1)),dim=-1)
        return x_a

class obj_distangler(nn.Module):
    def __init__(self, emd_dim):
        super(obj_distangler, self).__init__()
        self.AGV = AGV(emd_dim)
    def forward(self, xc, xp):
        return self.AGV(xc, xp)

class att_distangler(nn.Module):
    def __init__(self, emd_dim):
        super(att_distangler, self).__init__()
        self.AGV = AGV(emd_dim)
        self.OGA = OGA(emd_dim)
    def forward(self, xc, xp, p_o, omega):
        x_agv = self.AGV(xc, xp)
        x_a = self.OGA(xp, p_o, omega.detach(), x_agv)
        return x_a
    
class IMAX(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.additional_visual_params = self.add_visual_tunable_params()

        output_dim = self.clip.visual.output_dim

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        self.patch_norm = nn.LayerNorm(output_dim)

        self.idx_mapping = {'attr': 1, 'obj': 2, 'pair': 0}
        self._set_embeddings()
    def _set_embeddings(self):
        output_dim = self.clip.visual.output_dim
        self.obj_distangler = obj_distangler(output_dim)
        self.att_distangler = att_distangler(output_dim)
        self.composer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 4),
            nn.BatchNorm1d(output_dim * 4),
            nn.ReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim * 4, output_dim),
        )

        self.cp_a = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.cp_o = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.vp_a = nn.Linear(output_dim, output_dim)
        self.vp_o = nn.Linear(output_dim, output_dim)
    def tocomplex_space(self, img_obj, img_attr, obj, attr):
        img_r = self.cp_o(img_obj)
        img_i = self.cp_a(img_attr)
        text_r = self.vp_o(obj)
        text_i = self.vp_a(attr)
        return (img_r / img_r.norm(dim=-1, keepdim=True), img_i / img_i.norm(dim=-1, keepdim=True),
                text_r / text_r.norm(dim=-1, keepdim=True), text_i / text_i.norm(dim=-1, keepdim=True),)

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width, 
                                    bottleneck=self.config.adapter_dim, 
                                    dropout=self.config.adapter_dropout
                                ) for _ in range(adapter_num)])
        return params


    def encode_image(self, x: torch.Tensor):
        if self.config.adapter:
            return self.encode_image_with_adapter(x)
        else:
            return self.encode_image_wo_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature[:, 1:, :]

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip.dtype)
        
        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)
        token_tensor[0][
            :, 1 : len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
            :self.offset
        ].type(self.clip.dtype)
        token_tensor[1][
            :, 1 : len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.offset:
        ].type(self.clip.dtype)
        token_tensor[2][
            :, 1 : len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor
    

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        pair_logits, attr_logits, obj_logits, comp_logits, complex = predict
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_pair = loss_fn(pair_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_complex = loss_fn(complex, batch_target)
        loss = loss_pair * self.config.pair_loss_weight +\
               loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight + \
               loss_comp * self.config.comp_loss_weight +\
               loss_complex * self.config.complex_loss_weight
        return loss


    def logit_infer(self, predict, pairs):
        pair_logits, attr_logits, obj_logits, comp_logits, complex = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(pair_logits.shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            pair_logits[:, i_comp] = \
                (pair_logits[:, i_comp] * self.config.pair_inference_weight
                + comp_logits[:, i_comp] * self.config.comp_inference_weight
                 + complex[:, i_comp] * self.config.comp_inference_weight
                    + weighted_attr_pred * weighted_obj_pred)
        return pair_logits

    def forward(self, batch, idx):
        text_features = {}
        batch_features = {}

        batch_img = batch[0].cuda()
        l, _ = idx.shape
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype)) #(32 768), (32, 257, 768)

        batch_features['pair'] = batch_img / batch_img.norm(dim = -1, keepdim=True)

        token_tensors = self.construct_token_tensors(idx)

        logits = []
        for stage in ['pair', 'attr', 'obj']:
            i_element = self.idx_mapping[stage]
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )
            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features[stage] = idx_text_features

        logits.append(self.clip.logit_scale.exp() * batch_features['pair'] @ text_features['pair'].t())

        batch_features['obj'] = self.obj_distangler(batch_img, batch_patch)
        omega = F.softmax(batch_features['obj'] @ text_features['obj'].t() , dim = -1)

        batch_features['attr'] = self.att_distangler(batch_img, batch_patch, text_features['obj'], omega)

        comp_features = torch.cat((batch_features['attr'], batch_features['obj']), dim = -1)
        comp_features = comp_features / comp_features.norm(dim = -1, keepdim=True)
        batch_features['composed'] = self.composer(comp_features)
        batch_features['composed'] = batch_features['composed'] / batch_features['composed'].norm(dim = -1, keepdim=True)

        logits.append(self.clip.logit_scale.exp() * batch_features['attr'] @ text_features['attr'].t())
        logits.append(self.clip.logit_scale.exp() * batch_features['obj'] @ text_features['obj'].t())

        if self.config.comp_loss_weight or self.config.comp_inference_weight:
            logits.append(self.clip.logit_scale.exp() * batch_features['composed'] @ text_features['pair'].t())
        else:
            logits.append(self.clip.logit_scale.exp() * batch_features['pair'] @ text_features['pair'].t())

        attr_idx, obj_idx = idx[:, 0], idx[:, 1]
        text_features['attr_pair'], text_features['obj_pair'] = text_features['attr'][attr_idx], text_features['obj'][obj_idx]

        img_r, img_i, text_r, text_i = self.tocomplex_space(batch_features['obj'], batch_features['attr'],
                                                            text_features['obj_pair'], text_features['attr_pair'])

        real = torch.mm(img_r, text_r.T) + torch.mm(img_i, text_i.T)
        imagine = torch.mm(img_i, text_r.T) - torch.mm(img_r, text_i.T)

        logits.append(self.clip.logit_scale.exp() * (real - imagine))
        return logits