import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import numpy as np
import jittor as jt
from jittor import init
from jittor import nn
import models.losses as losses
import models_en.networks as networks
import torch


class DP_GAN_model(nn.Module):
    def __init__(self, opt):
        super(DP_GAN_model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        self.netG = generators.DP_GAN_Generator(opt)
        self.netE = networks.define_E(opt)
        if opt.phase == "train" or opt.phase == "val":
            self.netD = discriminators.DP_GAN_Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with jt.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()
        #--- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
        self.GAN_loss = losses.GANLoss()
        self.MSELoss = nn.MSELoss(reduction='mean')
        self.KLDLoss = networks.KLDLoss()

    def align_loss(self, feats, feats_ref):
        loss_align = 0
        for f, fr in zip(feats, feats_ref):
            loss_align += self.MSELoss(f, fr)
        return loss_align
    # 补充函数
    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.mul(std) + mu

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        # if self.opt.use_vae and self.opt.isTrain:
        z, mu, logvar = self.encode_z(real_image)
        if compute_kld_loss:
            KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        return fake_image, KLD_loss

    def generate_fake_EMA(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        # if self.opt.use_vae and self.opt.isTrain:
        z, mu, logvar = self.encode_z(real_image)
        if compute_kld_loss:
            KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netEMA(input_semantics, z=z)

        return fake_image, KLD_loss

    def execute(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        if mode == "losses_G":
            loss_G = 0
            # fake = self.netG(label)
            fake, loss_KLD = self.generate_fake(label, image, True)
            loss_G += loss_KLD
            output_D, scores, feats = self.netD(fake)
            _, _, feats_ref = self.netD(image)
            loss_G_adv = losses_computer.loss(output_D, label, for_real=True)
            loss_G += loss_G_adv
            loss_ms = self.GAN_loss(scores, True, for_discriminator=False)
            # loss_G += loss_ms.item()
            loss_G += loss_ms
            loss_align = self.align_loss(feats, feats_ref)
            loss_G += loss_align
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == "losses_D":
            loss_D = 0
            with jt.no_grad():
                # fake = self.netG(label)
                fake, _ = self.generate_fake(label, image, True)
            output_D_fake, scores_fake, _ = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            loss_ms_fake = self.GAN_loss(scores_fake, False, for_discriminator=True)
            # loss_D += loss_D_fake + loss_ms_fake.item()
            loss_D += loss_D_fake + loss_ms_fake
            output_D_real, scores_real, _ = self.netD(image)
            loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            loss_ms_real = self.GAN_loss(scores_real, True, for_discriminator=True)
            # loss_D += loss_D_real + loss_ms_real.item()
            loss_D += loss_D_real + loss_ms_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed, _, _ = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
                                                                                output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "generate":
            with jt.no_grad():
                if self.opt.no_EMA:
                    # fake = self.netG(label)
                    fake, _ = self.generate_fake(label, image, True)
                else:
                    fake, _= self.generate_fake_EMA(label, image, True)
            return fake

        if mode == "eval":
            with jt.no_grad():
                pred, _, _ = self.netD(image)
            return pred
        
        if mode == "encode":
            with jt.no_grad():
                z, _1, _2 = self.encode_z(image)
            return z

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netE.load_state_dict(jt.load(path + "E.pkl"))
                self.netG.load_state_dict(jt.load(path + "G.pkl"))
            else:
                self.netE.load_state_dict(jt.load(path + "E.pkl"))
                self.netEMA.load_state_dict(jt.load(path + "EMA.pkl"))
        elif self.opt.phase == "eval":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netD.load_state_dict(jt.load(path + "D.pkl"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(jt.load(path + "G.pkl"))
            self.netD.load_state_dict(jt.load(path + "D.pkl"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(jt.load(path + "EMA.pkl"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    # param_count += sum([p.data.nelement() for p in module.parameters()])
                    param_count += sum([np.size(p.data) for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.gauss_(m.weight, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_gauss_(m.weight, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        # jt.flags.use_cuda = 1
        # jt.cudnn.set_max_workspace_ratio(0.0)
        pass
    else:
        # model.module = model
        pass
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    # if opt.gpu_ids != "-1":
    #     data['label'] = data['label']
    #     data['image'] = data['image']
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        # input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
        input_label = jt.zeros((bs, nc, h, w))
    else:
        # input_label = torch.FloatTensor(bs, nc, h, w).zero_()
        input_label = jt.zeros((bs, nc, h, w))
    input_semantics = input_label.scatter_(1, label_map, jt.float32(1.0))
    return data['image'], input_semantics


def generate_labelmix(label, fake_image, real_image):
    target_map, _ = jt.argmax(label, dim = 1, keepdims = True)
    all_classes = jt.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = jt.randint(0,2,(1,))
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image
    return mixed_image, target_map
