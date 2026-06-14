import torch
import torch.nn as nn
import torch.nn.functional as F
from MobileSAM.mobile_sam.modeling.common import LayerNorm2d


# 新的导入方式
from timm.layers.cbam import CbamModule as CBAM


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
    
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        if m.bias is not None:
            truncated_normal_(m.bias, mean=0, std=0.001)
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if type(m) == nn.BatchNorm2d:
        nn.init.uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

        
class MSConv2d(nn.Module):
    def __init__(self, ch, groups=4):
        super(MSConv2d, self).__init__()
        assert ch % groups == 0
        group_ch = ch // groups
        self.convs = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1)
        ])
        for i in range(1, groups):
            self.convs.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, groups=group_ch)
            )
        self.activate = nn.GELU()
        self.norm = nn.BatchNorm2d(ch)
        self.groups = groups

    def forward(self, x):
        features = x.chunk(self.groups, dim=1)
        outs = []
        for i in range(len(features)):
            outs.append(self.convs[i](features[i]))
        net = torch.cat(outs, dim=1)
        net = self.norm(net)
        net = self.activate(net)
        return net


class PromptGen(nn.Module):
    def __init__(self, blk, reduction=4, cls_token=False, reshape=False, seq_size=None) -> None:
        super(PromptGen, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        prompt_dim = dim // reduction
        self.prompt_learn = nn.Sequential(
            # nn.Linear(dim, 32),
            # nn.GELU(),
            # nn.Linear(32, dim),
            # nn.GELU()
            nn.Conv2d(dim, prompt_dim, 1, 1),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, groups=prompt_dim, bias=False),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, dim, 1, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )
        self.cls_token = cls_token
        self.reshape = reshape
        self.seq_size = seq_size
        self.prompt_learn.apply(init_weights)
    
    def forward(self, x):
        if self.cls_token:
            tokens = x[:,1:]
            bs, seq_len, dim = tokens.size()
            if self.reshape:
                tokens = tokens.reshape(-1, self.seq_size, self.seq_size, dim).permute(0, 3, 1, 2)
            prompt = self.prompt_learn(tokens)
            promped = tokens + prompt
            promped = promped.reshape(bs, dim, seq_len).transpose(1, 2)
            promped = torch.cat([x[:, 0].unsqueeze(1), promped], dim=1)
        else:
            prompt = self.prompt_learn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # prompt = self.prompt_learn(x)
            promped = x + prompt
        net = self.block(promped)
        return net
    
class LearnablePromptSAM(nn.Module):
    def __init__(self, sam=None, num_classes=None, reduction=4, upsample_times=2, groups=4):
        super().__init__()
        self.sam = sam
        self.num_classes = num_classes
        self.cbam=CBAM(64).cuda()
        # freeze image encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
 
        blocks = []
        for block in self.sam.image_encoder.blocks: 
            blocks.append(
                PromptGen(block, reduction=reduction)
            )

        self.sam.image_encoder.blocks = nn.Sequential(
            *blocks
        )

        ########################################
        out_dim = self.sam.image_encoder.neck[0].out_channels
        self.img_size = self.sam.image_encoder.img_size

        del self.sam.prompt_encoder
        del self.sam.mask_decoder            

        self.up_conv = nn.ModuleDict()
        self.up_times = upsample_times
        dim = out_dim
        for i in range(upsample_times):
            self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                    nn.ConvTranspose2d(dim, dim//2, 2, 2),
                    LayerNorm2d(dim // 2),
                    nn.GELU()
                )
            dim = dim // 2
        self.ms_conv = MSConv2d(dim, groups=groups)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )


    def upscale(self, x, times=2):
        for i in range(times):
            x = self.up_conv["up_{}".format(i+1)](x)
        return x
    
    def forward(self, x, boxes=None):
        out = self.sam.image_encoder(x)

        out = self.upscale(out, self.up_times)
        out = self.ms_conv(out)
        # print(out.shape)
        out = self.cbam(out)
        # seg_out.shape torch.Size([2, 8, 256, 256])
        seg_out = self.decoder(out)
        # seg_out.shape torch.Size([2, 8, 1024, 1024])
        seg_out = F.interpolate(seg_out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return seg_out
    

class EncoderOnlyMobileSAM(nn.Module):
    def __init__(self, sam=None, num_classes=None, reduction=4, upsample_times=2, groups=4):
        super().__init__()
        self.sam = sam
        self.num_classes = num_classes
        # Tag=True
        # # freeze image encoder
        # for param in self.sam.image_encoder.parameters():
        #     if(Tag==True):
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        #     Tag= not Tag;
        out_dim = self.sam.image_encoder.neck[0].out_channels
        self.img_size = self.sam.image_encoder.img_size
        del self.sam.prompt_encoder
        del self.sam.mask_decoder
        self.up_conv = nn.ModuleDict()
        self.up_times = upsample_times
        dim = out_dim
        for i in range(upsample_times):
            self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                    nn.ConvTranspose2d(dim, dim//2, 2, 2),
                    LayerNorm2d(dim // 2),
                    nn.GELU()
                )
            dim = dim // 2
        self.ms_conv = MSConv2d(dim, groups=groups)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )


    def upscale(self, x, times=2):
        for i in range(times):
            x = self.up_conv["up_{}".format(i+1)](x)
        return x
    
    def forward(self, x, boxes=None):
        out = self.sam.image_encoder(x)
        out = self.upscale(out, self.up_times)
        out = self.ms_conv(out)
        
        # seg_out.shape torch.Size([2, 8, 256, 256])
        seg_out = self.decoder(out)
        # seg_out.shape torch.Size([2, 8, 1024, 1024])
        seg_out = F.interpolate(seg_out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return seg_out
