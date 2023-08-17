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


def show_np_r(array, min, max, num):
    plt.figure(num)
    plt.imshow(array, norm=None, cmap='gray', vmin= min, vmax=max)
    plt.axis('off')
    plt.show()

def show_hot_r(array, num):
    plt.figure(num)
    plt.imshow(array, norm=None, cmap='hot')
    plt.axis('off')
    plt.show()

def show_torch_rgb(array, min, max, num):
    plt.figure(num)
    plt.imshow(array.detach().cpu()[0].permute(1,2,0).numpy()*255,  norm=None, cmap='gray', vmin= min, vmax=max)
    plt.axis('off')
    plt.show()


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def get_lambda(alpha=1.0,size=None,device=None):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
#         lam = Beta()
    else:
        lam = 1.
    return lam
def get_spa_lambda(alpha=1.0,size=None,device=None):
    '''Return lambda'''
    if alpha > 0.:
        lam = torch.from_numpy(np.random.beta(alpha, alpha,size=size)).float().to(device)
#         lam = Beta()
    else:
        lam = 1.
    return lam
class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_HDCE', type=float, default=1.0, help='weight for HDCE loss: HDCE(G(X), X)')
        parser.add_argument('--lambda_SRC', type=float, default=1.0, help='weight for SRC loss: SRC(G(X), X)')
        parser.add_argument('--dce_idt', action='store_true')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--use_curriculum', action='store_true')
        parser.add_argument('--HDCE_gamma', type=float, default=1)
        parser.add_argument('--HDCE_gamma_min', type=float, default=1)
        parser.add_argument('--step_gamma', action='store_true')
        parser.add_argument('--step_gamma_epoch', type=int, default=200)
        parser.add_argument('--no_Hneg', action='store_true')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.train_epoch = None

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']

        if opt.lambda_HDCE > 0.0:
            self.loss_names.append('HDCE')
            if opt.dce_idt and self.isTrain:
                self.loss_names += ['HDCE_Y']

        if opt.lambda_SRC > 0.0:
            self.loss_names.append('SRC')


        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.alpha = opt.alpha
        if opt.dce_idt and self.isTrain:
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)


        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

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
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionR = []
            for nce_layer in self.nce_layers:
                self.criterionR.append(SRC_Loss(opt).to(self.device))


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            # if self.opt.lambda_NCE > 0.0:
            #     self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            #     self.optimizers.append(self.optimizer_F)
            #
            # elif self.opt.lambda_HDCE > 0.0:
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            # if self.opt.lambda_NCE > 0.0:
            #     self.optimizer_F.zero_grad()
            # elif self.opt.lambda_HDCE > 0.0:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            # if self.opt.lambda_NCE > 0.0:
            #     self.optimizer_F.step()
            # elif self.opt.lambda_HDCE > 0.0:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.dce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.dce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]


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

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

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


        ## Relation Loss
        self.loss_SRC, weight = self.calculate_R_loss(real_A_pool, fake_B_pool, epoch=self.train_epoch)


        ## HDCE
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

        self.loss_G = self.loss_G_GAN + loss_HDCE_both + self.loss_SRC
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
        weights=[]
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionR, self.nce_layers):
            loss_SRC, weight = crit(f_q, f_k, only_weight, epoch)
            total_SRC_loss += loss_SRC * self.opt.lambda_SRC
            weights.append(weight)
        return total_SRC_loss / n_layers, weights

    
#--------------------------------------------------------------------------------------------------------    
    def calculate_Patchloss(self, src, tgt, num_patch=4):

        feat_org = self.netG(src, mode='encoder')
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_org = torch.flip(feat_org, [3])
        
        N,C,H,W = feat_org.size()
        
        ps = H//num_patch
        lam = get_spa_lambda(self.alpha,size=(1,1,num_patch**2),device = feat_org.device)
        feat_org_unfold  = F.unfold(feat_org,kernel_size=(ps,ps),padding=0,stride=ps)

        rndperm = torch.randperm(feat_org_unfold.size(2))
        feat_prm = feat_org_unfold[:,:,rndperm]
        feat_mix = lam*feat_org_unfold + (1-lam)*feat_prm
        feat_mix = F.fold(feat_mix,output_size=(H,W),kernel_size=(ps,ps),padding=0,stride=ps)
        
        out_mix = self.netG(feat_mix,mode='decoder')
        feat_mix_rec = self.netG(out_mix,mode='encoder')
        
        fake_feat = self.netG(tgt,mode='encoder')

        fake_feat_unfold  = F.unfold(fake_feat,kernel_size=(ps,ps),padding=0,stride=ps)
        fake_feat_prm = fake_feat_unfold[:,:,rndperm]
        fake_feat_mix = lam*fake_feat_unfold + (1-lam)*fake_feat_prm
        fake_feat_mix = F.fold(fake_feat_mix,output_size=(H,W),kernel_size=(ps,ps),padding=0,stride=ps)
        
        
        PM_loss = torch.mean(torch.abs(fake_feat_mix - feat_mix_rec))

        return 10*PM_loss

#--------------------------------------------------------------------------------------------------------   