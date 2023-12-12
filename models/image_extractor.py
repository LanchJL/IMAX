import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock
from .vision_transformer import vit_base
import timm

class ViT(nn.Module):
    def __init__(self, model_name="vit_large_patch16_224_in21k", pretrained=True):
        super(ViT, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        # Others variants of ViT can be used as well
        '''
        1 --- 'vit_small_patch16_224'
        2 --- 'vit_base_patch16_224'
        3 --- 'vit_large_patch16_224',
        4 --- 'vit_large_patch32_224'
        5 --- 'vit_deit_base_patch16_224'
        6 --- 'deit_base_distilled_patch16_224',
        '''
        print(self.vit)
        # Change the head depending of the dataset used
        self.vit.head = nn.Identity()
        self.vit.dist_token = None
    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        if self.vit.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        x = x[:, 1:]
        x = x.view(-1,768,14,14)
        x = F.avg_pool2d(x,kernel_size = 14).view(-1,768)
        return x
class ResNet18_conv(ResNet):
    def __init__(self):
        super(ResNet18_conv, self).__init__(BasicBlock, [2, 2, 2, 2])
        
    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def get_image_extractor(arch = 'resnet18', pretrained = True, feature_dim = None, checkpoint = ''):
    '''
    Inputs
        arch: Base architecture
        pretrained: Bool, Imagenet weights
        feature_dim: Int, output feature dimension
        checkpoint: String, not implemented
    Returns
        Pytorch model
    '''

    if arch == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(512, feature_dim)
        #print(model)

    if arch == 'resnet18_conv':
        model = ResNet18_conv()
        model.load_state_dict(models.resnet18(pretrained=True).state_dict())

    elif arch == 'resnet50':
        model = models.resnet50(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim) 

    elif arch == 'resnet50_cutmix':
        model = models.resnet50(pretrained = pretrained)
        checkpoint = torch.load('/home/ubuntu/workspace/pretrained/resnet50_cutmix.tar')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim)

    elif arch == 'dino':
        model = vit_base()
        state_dict = torch.load('./pretrain/dino_vitbase16_pretrain.pth')
        model.load_state_dict(state_dict, strict=True)

    elif arch == 'resnet152':
        model = models.resnet152(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim) 

    elif arch == 'vgg16':
        model = models.vgg16(pretrained = pretrained)
        modules = list(model.classifier.children())[:-3]
        model.classifier=torch.nn.Sequential(*modules)
        if feature_dim is not None:
            model.classifier[3]=torch.nn.Linear(4096,feature_dim)

    elif arch == 'vit':
        model = ViT(model_name='vit_base_patch16_224',pretrained=True)
    return model

