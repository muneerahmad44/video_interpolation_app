import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.optim import AdamW 
from repo.model.IFNet import IFNet
from repo.model.IFNet_m import IFNet_m
from repo.model.loss import EPE, SOBEL  # Removed LapLoss import
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F



# Add the path where ECCV2022RIFE folder exists
#sys.path.append('//Computer Vision/Object detection/tomato_detection_project_fasterrcnn/web_app_for_tomato_detection/frame_interpolation_app/backend/ECCV2022-RIFE')  # replace with your actual path


# Add the parent directory of the current file to sys.path



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        if arbitrary:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3)
        self.epe = EPE()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
      def convert(param):
        return {
            k.replace("module.", "") if k.startswith("module.") else k: v
            for k, v in param.items()
        }
      if rank <= 0:
        checkpoint = torch.load(f'{path}/flownet.pkl', map_location=torch.device('cpu'))
        checkpoint = convert(checkpoint)
        self.flownet.load_state_dict(checkpoint)

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if not TTA:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])

        # Replaced LapLoss with L1 loss
        loss_l1 = F.l1_loss(merged[2], gt)
        loss_tea = F.l1_loss(merged_teacher, gt)

        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]

        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
        }
