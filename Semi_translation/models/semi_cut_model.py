import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from torch.distributions.beta import Beta
from torch.nn import functional as F
from models.hDCE import PatchHDCELoss
from models.SRC import SRC_Loss
import torch.nn as nn
import math

from models.vgg import VGG19

import matplotlib.pyplot as plt

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


def get_spa_lambda(alpha=1.0, size=None, device=None):
    '''Return lambda'''
    if alpha > 0.:
        lam = torch.from_numpy(np.random.beta(alpha, alpha, size=size)).float().to(device)
    #         lam = Beta()
    else:
        lam = 1.
    return lam


class SEMICUTModel(BaseModel):
    """ This class implements Semi-supervised I2I model, described in the paper
    Scenimefy: Learning to Craft Anime Scene via Semi-Supervised Image-to-Image Translation
    ICCV, 2023

    The code borrows heavily from the PyTorch implementation of CUT
    https://github.com/taesungp/contrastive-unpaired-translation
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_GAN_p', type=float, default=1.0, help='weight for supervised GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_HDCE', type=float, default=0.1, help='weight for HDCE loss: HDCE(G(X), X)')
        parser.add_argument('--lambda_SRC', type=float, default=0.05, help='weight for SRC loss: SRC(G(X), X)')
        parser.add_argument('--lambda_NCE_s', type=float, default=0.1, help='weight for StylePatchNCE loss: NCE(G(X^p), Y^p)')
        parser.add_argument('--lambda_VGG', type=float, default=0.1, help='weight for VGG content loss: VGG(G(X), Y)')
        parser.add_argument('--isDecay', type=bool, default=True, help='gradually decrease the weight for the supervised training branch')
        parser.add_argument('--dce_idt', action='store_true')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--use_curriculum', action='store_true')
        parser.add_argument('--HDCE_gamma', type=float, default=50)
        parser.add_argument('--HDCE_gamma_min', type=float, default=10)
        parser.add_argument('--step_gamma', action='store_true')
        parser.add_argument('--step_gamma_epoch', type=int, default=200)
        parser.add_argument('--no_Hneg', action='store_true')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.train_epoch = None
        self.N_EPOCHS = opt.n_epochs + opt.n_epochs_decay

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'G_GAN_p', 'D_p_real', 'D_p_fake']

        if opt.lambda_HDCE > 0.0:
            self.loss_names.append('HDCE')
            if opt.dce_idt and self.isTrain:
                self.loss_names += ['HDCE_Y']

        if opt.lambda_SRC > 0.0:
            self.loss_names.append('SRC')

        if opt.lambda_VGG > 0.0:
            self.loss_names.append('VGG')

        if opt.lambda_NCE_s > 0.0:
            self.loss_names.append('NCE_s')

        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_A_p', 'fake_B_p', 'real_B_p']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.alpha = opt.alpha
        if opt.dce_idt and self.isTrain:
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'D_p', 'F_s']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_s = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define the paired discriminator
            netD_p_input_nc = opt.input_nc + opt.output_nc
            self.netD_p = networks.define_D(netD_p_input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # extra content loss
            if opt.lambda_VGG > 0.0:
                vgg_model = 'vgg19'
                self.VGG = VGG19(init_weights=vgg_model, feature_mode=True)
                self.VGG.to(self.device)
                self.VGG.eval()
                self.criterionVGG = nn.MSELoss().to(self.device)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionHDCE = []

            for i, nce_layer in enumerate(self.nce_layers):
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
                self.criterionHDCE.append(PatchHDCELoss(opt=opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_p = torch.optim.Adam(self.netD_p.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_p)

            self.criterionR = []
            for nce_layer in self.nce_layers:
                self.criterionR.append(SRC_Loss(opt).to(self.device))

    def data_dependent_initialize(self, data, paired_input):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data, paired_input)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        bs_p_per_gpu = self.real_A_p.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A_p = self.real_A_p[:bs_p_per_gpu]
        self.real_B_p = self.real_B_p[:bs_p_per_gpu]

        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_D_p_loss().backward()  # calculate gradients for D_p
            
            self.compute_G_loss().backward()  # calculate gradients for G
            
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)
            self.optimizer_F_s = torch.optim.Adam(self.netF_s.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F_s)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update D_p
        self.set_requires_grad(self.netD_p, True)
        self.optimizer_D_p.zero_grad()
        self.loss_D_p = self.compute_D_p_loss()
        self.loss_D_p.backward()
        self.optimizer_D_p.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_p, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
            self.optimizer_F_s.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            self.optimizer_F_s.step()

    def set_input(self, input, paired_input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # load paired data
        self.real_A_p = paired_input['A' if AtoB else 'B'].to(self.device)
        self.real_B_p = paired_input['B' if AtoB else 'A'].to(self.device)
        self.image_paths_p = paired_input['path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.dce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        # forward unpaired data
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.dce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        # forward paired data
        self.fake_p = self.netG(self.real_A_p)
        self.fake_B_p = self.fake_p[:self.real_A_p.size(0)]

    def set_epoch(self, epoch):
        self.train_epoch = epoch

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_D_p_loss(self):
        """Calculate GAN loss for the paired discriminator"""
        # use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A_p, self.fake_B_p), 1)
        pred_fake = self.netD_p(fake_AB.detach())
        self.loss_D_p_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A_p, self.real_B_p), 1)
        pred_real = self.netD_p(real_AB)
        self.loss_D_p_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D_p = (self.loss_D_p_fake + self.loss_D_p_real) * 0.5
        return self.loss_D_p

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        fake_AB = torch.cat((self.real_A_p, self.fake_B_p), 1)

        # cosine decays: current weight for the supervised branch
        lambda_pair = 1.0
        if self.opt.isDecay:
            lambda_pair = math.cos(math.pi/2*self.N_EPOCHS * (self.train_epoch - 1))
        
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            # paired discriminator
            pred_fake_p = self.netD_p(fake_AB)
            self.loss_G_GAN_p = self.criterionGAN(pred_fake_p, True) * self.opt.lambda_GAN_p
        else:
            self.loss_G_GAN = 0.0
            self.loss_G_GAN_p = 0.0

        ## get feat
        fake_B_feat = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
        real_A_feat = self.netG(self.real_A, self.nce_layers, encode_only=True)

        fake_B_pool, sample_ids = self.netF(fake_B_feat, self.opt.num_patches, None)
        real_A_pool, _ = self.netF(real_A_feat, self.opt.num_patches, sample_ids)

        if self.opt.dce_idt:
            idt_B_feat = self.netG(self.idt_B, self.nce_layers, encode_only=True)
            if self.opt.flip_equivariance and self.flipped_for_equivariance:
                idt_B_feat = [torch.flip(fq, [3]) for fq in idt_B_feat]
            real_B_feat = self.netG(self.real_B, self.nce_layers, encode_only=True)

            idt_B_pool, _ = self.netF(idt_B_feat, self.opt.num_patches, sample_ids)
            real_B_pool, _ = self.netF(real_B_feat, self.opt.num_patches, sample_ids)

        # StylePatchNCE loss
        self.loss_NCE_s = self.calculate_NCE_loss(self.real_B_p, self.fake_B_p) * self.opt.lambda_NCE_s

        ## Relation Loss
        self.loss_SRC, weight = self.calculate_R_loss(real_A_pool, fake_B_pool, epoch=self.train_epoch)

        # HDCE
        if self.opt.lambda_HDCE > 0.0:
            self.loss_HDCE = self.calculate_HDCE_loss(real_A_pool, fake_B_pool, weight)
        else:
            self.loss_HDCE, self.loss_HDCE_bd = 0.0, 0.0

        self.loss_HDCE_Y = 0
        if self.opt.dce_idt and self.opt.lambda_HDCE > 0.0:
            _, weight_idt = self.calculate_R_loss(real_B_pool, idt_B_pool, only_weight=True, epoch=self.train_epoch)
            self.loss_HDCE_Y = self.calculate_HDCE_loss(real_B_pool, idt_B_pool, weight_idt)
            loss_HDCE_both = (self.loss_HDCE + self.loss_HDCE_Y) * 0.5
        else:
            loss_HDCE_both = self.loss_HDCE

        # Extract content features from content image
        # the pre-trained VGG uses the input range of [0, 1]
        if self.opt.lambda_VGG > 0.0:
            fake_feature = self.VGG((self.fake_B + 1 / 2.))
            nature_feature = self.VGG((self.real_A + 1) / 2.)
            self.loss_VGG = self.criterionVGG(fake_feature, nature_feature) * self.opt.lambda_VGG
            self.loss_G = self.loss_VGG
        else:
            self.loss_G = 0.0
            
        # add G_GAN_p loss
        self.loss_G += self.loss_G_GAN + loss_HDCE_both + self.loss_SRC + (self.loss_G_GAN_p +self.loss_NCE_s ) * lambda_pair
        return self.loss_G

    def calculate_HDCE_loss(self, src, tgt, weight=None):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_HDCE_loss = 0.0
        for f_q, f_k, crit, nce_layer, w in zip(feat_q_pool, feat_k_pool, self.criterionHDCE, self.nce_layers, weight):
            if self.opt.no_Hneg:
                w = None
            loss = crit(f_q, f_k, w) * self.opt.lambda_HDCE
            total_HDCE_loss += loss.mean()

        return total_HDCE_loss / n_layers

    def calculate_R_loss(self, src, tgt, only_weight=False, epoch=None):
        n_layers = len(self.nce_layers)

        feat_q_pool = tgt
        feat_k_pool = src

        total_SRC_loss = 0.0
        weights = []
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionR, self.nce_layers):
            loss_SRC, weight = crit(f_q, f_k, only_weight, epoch)
            total_SRC_loss += loss_SRC * self.opt.lambda_SRC
            weights.append(weight)
        return total_SRC_loss / n_layers, weights

    # --------------------------------------------------------------------------------------------------------
    def calculate_Patchloss(self, src, tgt, num_patch=4):
        feat_org = self.netG(src, mode='encoder')
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_org = torch.flip(feat_org, [3])

        N, C, H, W = feat_org.size()

        ps = H // num_patch
        lam = get_spa_lambda(self.alpha, size=(1, 1, num_patch ** 2), device=feat_org.device)
        feat_org_unfold = F.unfold(feat_org, kernel_size=(ps, ps), padding=0, stride=ps)

        rndperm = torch.randperm(feat_org_unfold.size(2))
        feat_prm = feat_org_unfold[:, :, rndperm]
        feat_mix = lam * feat_org_unfold + (1 - lam) * feat_prm
        feat_mix = F.fold(feat_mix, output_size=(H, W), kernel_size=(ps, ps), padding=0, stride=ps)

        out_mix = self.netG(feat_mix, mode='decoder')
        feat_mix_rec = self.netG(out_mix, mode='encoder')

        fake_feat = self.netG(tgt, mode='encoder')

        fake_feat_unfold = F.unfold(fake_feat, kernel_size=(ps, ps), padding=0, stride=ps)
        fake_feat_prm = fake_feat_unfold[:, :, rndperm]
        fake_feat_mix = lam * fake_feat_unfold + (1 - lam) * fake_feat_prm
        fake_feat_mix = F.fold(fake_feat_mix, output_size=(H, W), kernel_size=(ps, ps), padding=0, stride=ps)

        PM_loss = torch.mean(torch.abs(fake_feat_mix - feat_mix_rec))

        return 10 * PM_loss


    # StylePatchNCE loss
    def calculate_NCE_loss(self, src, tgt):
        """
        The code borrows heavily from the PyTorch implementation of CUT
        https://github.com/taesungp/contrastive-unpaired-translation
        """
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF_s(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF_s(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE_s
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

# --------------------------------------------------------------------------------------------------------