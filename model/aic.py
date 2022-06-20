import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .captioning_model import CaptioningModel


class AIC(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.decoder=decoder
    
    def forward(self,detection,captions):
        # print('detection',detection.shape)
        # print('captions',captions.shape)
            
        x=self.encoder(detection)
        # print(x.shape)
        x=self.decoder(x,captions)
        # print('output',x.shape)
        return x
