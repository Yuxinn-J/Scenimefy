from packaging import version
import torch
from torch import nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class SRC_Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, only_weight=False, epoch=None):
        '''
        :param feat_q: target
        :param feat_k: source
        :return: SRC loss, weights for hDCE
        '''

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = 1   # self.opt.batch_size
        feat_k = Normalize()(feat_k)
        feat_q = Normalize()(feat_q)

        ## SRC
        feat_q_v = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_v = feat_k.view(batch_dim_for_bmm, -1, dim)

        spatial_q = torch.bmm(feat_q_v, feat_q_v.transpose(2, 1))
        spatial_k = torch.bmm(feat_k_v, feat_k_v.transpose(2, 1))

        weight_seed = spatial_k.clone().detach()
        diagonal = torch.eye(self.opt.num_patches, device=feat_k_v.device, dtype=self.mask_dtype)[None, :, :]

        HDCE_gamma = self.opt.HDCE_gamma
        if self.opt.use_curriculum:
            HDCE_gamma = HDCE_gamma + (self.opt.HDCE_gamma_min - HDCE_gamma) * (epoch) / (self.opt.n_epochs + self.opt.n_epochs_decay)
            if (self.opt.step_gamma)&(epoch>self.opt.step_gamma_epoch):
                HDCE_gamma = 1


        ## weights by semantic relation
        weight_seed.masked_fill_(diagonal, -10.0)
        weight_out = nn.Softmax(dim=2)(weight_seed.clone() / HDCE_gamma).detach()
        wmax_out, _ = torch.max(weight_out, dim=2, keepdim=True)
        weight_out /= wmax_out

        if only_weight:
            return 0, weight_out

        spatial_q = nn.Softmax(dim=1)(spatial_q)
        spatial_k = nn.Softmax(dim=1)(spatial_k).detach()

        loss_src = self.get_jsd(spatial_q, spatial_k)

        return loss_src, weight_out

    def get_jsd(self, p1, p2):
        '''
        :param p1: n X C
        :param p2: n X C
        :return: n X 1
        '''
        m = 0.5 * (p1 + p2)
        out = 0.5 * (nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p1))
                     + nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p2)))
        return out