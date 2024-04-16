"""
We have reused part of CoSLAM's code in this file and include our code for NARUTO.
For CoSLAM License, refer to https://github.com/HengyiWang/Co-SLAM/blob/main/LICENSE.
"""


''' Differences v.s. original CoSLAM decoder
- include uncertainty prediction (UncertaintyNet)
'''

import tinycudann as tcnn
import torch
import torch.nn as nn

from third_parties.coslam.model.decoder import SDFNet, ColorNet


class SDFNetNaruto(SDFNet):
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.model = self.get_model(config=config)
    
    def forward(self, x, return_geo=True):
        if self.config['decoder']['uncert_grid']:
            x = x.float()
            uncert, x = x[:, 0], x[:, 1:]
            out = self.model(x)
            out = torch.cat([out, uncert[..., None]], dim=1)
        else:
            out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, config):
        tcnn_network = config['decoder']['tcnn_network']
        uncert = 1 if config['decoder']['pred_uncert'] else 0
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim + uncert,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                #dtype=torch.float
            )
        else:
            sdf_net = []
            for l in range(self.num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim 
                
                if l == self.num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim + uncert # 1 sigma + 15 SH features for color + optional uncertainty
                else:
                    out_dim = self.hidden_dim 
                
                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))
        

class ColorSDFNet_v2_Naruto(nn.Module):
    '''
    No color grid
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet_v2_Naruto, self).__init__()
        self.config = config
        self.pred_uncert = config['decoder']['pred_uncert'] or config['decoder']['uncert_grid']
        self.color_net = ColorNet(config, 
                input_ch=input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNetNaruto(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])
            
    def forward(self, embed, embed_pos):

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if self.pred_uncert:
            geo_feat, uncert = geo_feat[..., :-1], geo_feat[..., -1:]
            sdf = torch.cat([sdf, uncert], -1)  # put uncertainty to last channel of the sdf.
        
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)