import warnings

import jittor as jt

import config
import dataloaders.dataloaders as dataloaders
import models.losses as losses
import models.models as models
import utils.utils as utils
from utils.fid_scores import fid_jittor

warnings.filterwarnings("ignore")
#--- read options ---#
opt = config.read_arguments(train=True)

jt.flags.use_cuda = 1
# check
# jt.cudnn.set_max_workspace_ratio(0.0)

#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloaders.dataset_seg(opt) # spilt the dataset as train_datatset and val_datatset 
# get dataloader
dataset = dataloaders.get_dataloaders(opt, load_type='train')
dataloader = dataset.set_attrs(
    batch_size = opt.batch_size, 
    shuffle = True, 
    drop_last=True, 
    num_workers=16
)
dataset_val = dataloaders.get_dataloaders(opt, load_type='val')
dataloader_val = dataset_val.set_attrs(
    batch_size = opt.batch_size, 
    shuffle = False, 
    drop_last=False, 
    num_workers=16
)
print("size train: %d, size val: %d" % (len(dataset), len(dataset_val)))

im_saver = utils.image_saver(opt)
fid_computer = fid_jittor(opt, dataloader_val)

#--- create models ---#
model = models.DP_GAN_model(opt)
model = models.put_on_multi_gpus(model, opt)

#--- create optimizers ---#
optimizerG = jt.optim.Adam(model.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = jt.optim.Adam(model.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader) + i
        image, label = models.preprocess_input(opt, data_i)

        #--- generator update ---#
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        # loss_G.sync()
        optimizerG.step(loss_G)

        #--- discriminator update ---#
        loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        # loss_D.sync()
        optimizerD.step(loss_D)

        #--- generate z ---#
        z_dim = model(image, label, "encode", losses_computer)

        #--- stats update ---#
        if not opt.no_EMA:
            utils.update_EMA(model, cur_iter, dataloader, opt, z_dim)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter, z_dim)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_ckpt == 0:
            utils.save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter, z_dim)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
        visualizer_losses(cur_iter, losses_G_list+losses_D_list)

        # check recycle mem
        # jt.sync_all()
        # jt.gc()

#--- after training ---#
utils.update_EMA(model, cur_iter, dataloader, opt, z_dim, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter, z_dim)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")

