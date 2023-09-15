import jittor as jt
from jittor import nn
from models.spectral_norm import spectral_norm


class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        #self.mlp_shared = nn.Sequential(
        #    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
        #    nn.ReLU()
        #)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def execute(self, x, segmap):
        normalized = self.first_norm(x)
        #segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        #actv = self.mlp_shared(segmap)
        actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        # return torch.nn.Identity()
        return nn.Identity()
    else:
        return spectral_norm


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'syncbatch':
        # return SynchronizedBatchNorm2d(norm_nc, affine=False)
        # TODO 多卡用jittor mpi实现, 否则会出现异步的问题
        return nn.BatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)
