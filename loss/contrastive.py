"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets,epoch):
        device = torch.device('cuda')

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(3, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(len(self.cls_num_list)).to(device)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:3 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 3).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:3 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logit = torch.exp(logits)

        # class-averaging
        exp_logits = exp_logit * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            3 * batch_size, 3 * batch_size + len(self.cls_num_list)) - mask

        '''betas = 0.9999
        effective_num = 1.0 - torch.pow(betas, per_ins_weight)
        per_cls_weights = (1.0 - betas) / effective_num.clone().detach().requires_grad_(True)
        per_cls_weights = per_cls_weights / torch.sum(per_cls_weights) * len(per_ins_weight)
        #per_cls_weights = torch.FloatTensor(per_cls_weights)

        per_cls_weights = torch.mul(per_cls_weights, per_ins_weight)'''

        if epoch <= 30:
            betas = 0.9999
            effective_num = 1.0 - torch.pow(torch.tensor(betas), per_ins_weight)
            per_cls_weights = (1.0 - betas) / effective_num.clone().detach().requires_grad_(True)
            per_cls_weights = per_cls_weights / torch.sum(per_cls_weights) * len(batch_cls_count)
            # per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        else:
            betas = 0.9999
            effective_num = 1.0 - torch.pow(torch.tensor(betas), per_ins_weight)
            per_cls_weights = (1.0 - betas) / effective_num.clone().detach().requires_grad_(True)
            per_cls_weights = per_cls_weights / torch.sum(per_cls_weights) * len(batch_cls_count)

            per_cls_weights = torch.mul(per_ins_weight, per_cls_weights)

        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        exp_logits_sum = per_cls_weights.sum(dim=1, keepdim=True) * exp_logits_sum

        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(3, batch_size).mean()
        return loss


