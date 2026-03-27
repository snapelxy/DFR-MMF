# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import cv2
import copy
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import torchvision
from vedacore.misc import build_from_cfg, registry
from scipy.ndimage import gaussian_filter


@registry.register_module('common_model')
class RD4AD(torch.nn.Module):
    def __init__(self, **kargs):
        """ Initializes the model.
        """
        super().__init__()
        add_cfg = {}
        if "add_cfg" in kargs.keys():
            add_cfg = copy.deepcopy(kargs["add_cfg"])
            del kargs["add_cfg"]
        self.use_dist = add_cfg["use_dist"] if "use_dist" in add_cfg.keys() else False
        self.ModuleList = self.build_net(kargs)
        # self.backbone = build_backbone(backbone)
        self.train_inited = False
        self.use_vqe = True
        
    
    def cp_vqe(self):
        for k in self.ModuleList:
            if k == "vqvae_net":
                obj = getattr(self, k)
                obj.cp_embed=True

    def build_net(self, cfg):
        ModuleList = []
        for k, v in cfg.items():
            setattr(self, k, build_from_cfg(v, registry, 'common_model'))
            ModuleList.append(k)
        return ModuleList

    def forward(self, input_dict, **kargs):
        if self.training:
            if not self.train_inited:
                self.encoder.eval()
                self.train_inited = True
            outputs = self.forward_img(input_dict)
        else:
            outputs = self.forward_img(input_dict)

        return outputs

    def forward_img(self, input_dict, **kargs):
        with torch.no_grad():
            feats = self.encoder(input_dict["input_x"])
        feats_en = copy.deepcopy(feats)
        for k in self.ModuleList:
            if k == "vqvae_net":
                if self.use_vqe or (not self.training):
                    obj = getattr(self, k)
                    feats, vqe_diff, embed_ind,dist_mask = obj(feats)
            elif k != "encoder":
                obj = getattr(self, k)
                # n0 = feats[0].std(dim=1).detach().cpu().numpy()
                # np.save("./f_transist/n0",n0)
                feats = obj(feats)
                # n0_f = feats[0].std(dim=1).detach().cpu().numpy()
                # np.save("./f_transist/n0_f",n0_f)

        return {
            "feats_en": feats_en,
            "feats_de": feats,
        }

    def get_anomal_map(self, input_dict, out_size=256, amap_mode='mul', sigma=4, **kargs):
        if amap_mode == 'mul':
            anomaly_map = np.ones([out_size, out_size])
        else:
            anomaly_map = np.zeros([out_size, out_size])
        fs_list = input_dict["feats_en"]#[-2:-1]
        ft_list = input_dict["feats_de"]#[-2:-1]
        a_map_list = []
        for i in range(len(ft_list)):
            # i=0
            fs = fs_list[i]
            ft = ft_list[i]
            # fs_norm = F.normalize(fs, p=2)
            # ft_norm = F.normalize(ft, p=2)
            if self.use_dist:
                a_map = ft.unsqueeze(1)
            else:
                a_map = 1 - F.cosine_similarity(fs, ft)
                a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size,
                                  mode='bilinear', align_corners=True)
            a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
            a_map_list.append(a_map)
            if amap_mode == 'mul':
                anomaly_map *= a_map
            else:
                anomaly_map += a_map
        anomaly_map = gaussian_filter(anomaly_map, sigma=5)
        return anomaly_map, a_map_list

    def min_max_norm(self, image):
        a_min, a_max = image.min(), image.max()
        return (image-a_min)/(a_max - a_min)

    def cvt2heatmap(self, gray):
        heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
        return heatmap

    def show_cam_on_image(self, img, anomaly_map):
        # if anomaly_map.shape != img.shape:
        #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
        img = cv2.resize(img,(256,256))
        cam = np.float32(anomaly_map)/255 + np.float32(img)/255
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)


if __name__ == '__main__':
    
    x1 = torch.randn((1,3,256,256)) #.cuda()
    
    model = MobNetv3(pretrained=True)
    out = model(x1)
    
    print("finished")