# -*- coding: utf-8 -*-
import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any

from utils.inc_net import BaseNet, get_backbone
    

class AC_Linear(nn.Module):
    def __init__(self, in_features, Hidden, out_dim):
        super(AC_Linear, self).__init__()
        self.in_features = in_features
        self.hidden = Hidden
        self.out_features = out_dim
        bias_fe = False

        self.fc = nn.Sequential(
            nn.Linear(self.in_features, self.hidden, bias=bias_fe),
            nn.ReLU(),
            nn.Linear(self.hidden, self.out_features, bias=False)
        )

    def forward(self, input):
        hidden_feature = self.fc[:2](input)
        out = self.fc(input)
        return {'buffer_feature': hidden_feature, 'logits': out}


class VILA(BaseNet):
    def __init__(self, args, pretrained=None):
        super().__init__(args, pretrained)
        self.args=args
        self._device = None

        self.backbone.mlp0 = nn.Linear(self.feature_dim, self.args['init_cls'])
        self.ac_model = None

        self.clip, _, self.tokenizer = get_backbone(
            {"model_name": "vila", "backbone_type": "laion400m_e32_clip"}, 
            pretrained
        )

    def forward(self, x: torch.Tensor, clip_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features =  self.backbone(x)
        features = features / features.norm(dim=-1, keepdim=True)

        clip_features = self.clip_encode_image(clip_x)
        
        features = torch.cat([features, clip_features], dim=1)
        outputs = {"features": features, 'features': features, 'clip_features': clip_features}
        
        output = self.ac_model(features)
        output.update(outputs)
        return output

    def clip_encode_image(self, x: torch.Tensor) -> torch.Tensor:
        image_features =  self.clip.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def clip_encode_text(self, x: torch.Tensor) -> torch.Tensor:
        text_features = self.clip.encode_text(x)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def update_fc(self, hidden: int, nb_classes: int) -> None:
        ac_model = AC_Linear(self.feature_dim + self.clip.out_dim, hidden, nb_classes)
        
        if self.ac_model is not None:
            nb_output = self.ac_model.out_features
            hidden_weight = copy.deepcopy(self.ac_model.fc[0].weight.data)
            ac_model.fc[0].weight = nn.Parameter(hidden_weight.float())

            weight = copy.deepcopy(self.ac_model.fc[-1].weight.data)
            weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, hidden).to(self._device)])

            ac_model.fc[-1].weight = nn.Parameter(weight.float())

        del self.ac_model
        self.ac_model = ac_model
