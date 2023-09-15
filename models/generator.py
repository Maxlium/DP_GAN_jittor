import jittor as jt
from jittor import nn

import models.norms as norms


class DP_GAN_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))

        
        self.seg_pyrmid = nn.ModuleList([])
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)
            self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()))
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)
            self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(self.opt.semantic_nc, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()))

        self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU()))
        for i in range(len(self.channels)-2):
            self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU()))


    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def execute(self, input, z):
        seg = input
        z_tmp = z
        # if self.opt.gpu_ids != "-1":
        #     seg.cuda()
        #     z_tmp.cuda()
        # if not self.opt.no_3dnoise:
        #     dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
        #     z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
        z_tmp = z_tmp.view(z_tmp.size(0), self.opt.z_dim, 1, 1)
        z_tmp = z_tmp.expand(z_tmp.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
        seg = jt.concat((z_tmp, seg), dim = 1)
        x = nn.interpolate(seg, size=(self.init_W, self.init_H))

        pyrmid = []
        s = seg
        for op in self.seg_pyrmid:
            s = op(s)
            pyrmid.append(s)

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            seg_f = jt.concat(
                [nn.interpolate(pyrmid[-1], size=(x.shape[2], x.shape[3]), mode='bilinear'), 
                 pyrmid[self.opt.num_res_blocks - i]], 1
            )
            # print('seg_f.shape', seg_f.shape)
            x = self.body[i](x, seg_f)
            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
        x = self.conv_img(nn.leaky_relu(x, 2e-1))
        x = jt.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, )

    def execute(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out
