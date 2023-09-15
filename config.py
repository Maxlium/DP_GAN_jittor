import argparse
import pickle
import os
import utils.utils as utils


def read_arguments(train=True):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser, train)
    parser.add_argument('--phase', type=str, default='train')
    opt = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    if train:
        set_dataset_default_lm(opt, parser)
        if opt.continue_train:
            update_options_from_file(opt, parser)
    opt = parser.parse_args()
    opt.phase = 'train' if train else 'test'
    if train:
        opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)
    utils.fix_seed(opt.seed)
    print_options(opt, parser)
    if train:
        save_options(opt, parser)
    return opt


def add_all_arguments(parser, train):
    #--- general options --- ywb
    parser.add_argument('--name', type=str, default='jittor_train4', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--input_path', type=str, default='./B/val_B_labels_resized', help='test labels path')
    parser.add_argument('--json_path', type=str, default='./B/label_to_img.json', help='json path, in the same dir as input_path')  
    parser.add_argument('--img_path', type=str, default='./train_resized/imgs', help='imgs path')
    parser.add_argument('--output_path', type=str, default='./results', help='results path')
    parser.add_argument('--dataset_path', type=str, default='./datasets', help='new dataset used for train and val')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--no_spectral_norm', action='store_true', help='this option deactivates spectral norm in all layers')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--dataset_mode', type=str, default='ade20k', help='this option indicates which dataset should be loaded')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

    # for generator
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=64, help='# of gen filters in first conv layer in generator')
    parser.add_argument('--param_free_norm', type=str, default='syncbatch', help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--no_EMA', action='store_true',help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
    parser.add_argument('--no_3dnoise', action='store_true', default=False, help='if specified, do *not* concatenate noise to label maps')
    # parser.add_argument('--z_dim', type=int, default=64, help="dimension of the latent z vector")
    # for encoder
    parser.add_argument('--init_type', type=str, default='xavier',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02,
                        help='variance of the initialization distribution')
    parser.add_argument('--z_dim', type=int, default=256,
                        help="dimension of the latent z vector")
    parser.add_argument('--norm_E', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64,
                        help='# of gen filters in first conv layer')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--lambda_kld', type=float, default=0.05)
    if train:
        parser.add_argument('--freq_print', type=int, default=1000, help='frequency of showing training results')
        parser.add_argument('--freq_save_ckpt', type=int, default=20000, help='frequency of saving the checkpoints')
        parser.add_argument('--freq_save_latest', type=int, default=10000, help='frequency of saving the latest model')
        parser.add_argument('--freq_smooth_loss', type=int, default=250, help='smoothing window for loss visualization')
        parser.add_argument('--freq_save_loss', type=int, default=2500, help='frequency of loss plot updates')
        parser.add_argument('--freq_fid', type=int, default=5000, help='frequency of saving the fid score (in training iterations)')
        parser.add_argument('--continue_train', action='store_true', help='resume previously interrupted training')
        parser.add_argument('--which_iter', type=str, default='latest', help='which epoch to load when continue_train')
        parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs to train')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='G learning rate, default=0.0001')
        parser.add_argument('--lr_d', type=float, default=0.0004, help='D learning rate, default=0.0004')

        parser.add_argument('--channels_D', type=int, default=64, help='# of discrim filters in first conv layer in discriminator')
        parser.add_argument('--add_vgg_loss', action='store_true', help='if specified, add VGG feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for VGG loss')
        parser.add_argument('--no_balancing_inloss', action='store_true', default=False, help='if specified, do *not* use class balancing in the loss function')
        parser.add_argument('--no_labelmix', action='store_true', default=False, help='if specified, do *not* use LabelMix')
        parser.add_argument('--lambda_labelmix', type=float, default=10.0, help='weight for LabelMix regularization')
    else:
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves testing results here.')
        parser.add_argument('--ckpt_iter', type=str, default='best', help='which epoch to load to evaluate a model')
        
    return parser


def set_dataset_default_lm(opt, parser):
    if opt.dataset_mode == "ade20k":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
        #parser.set_defaults(lr_g=0.0001)
        #parser.set_defaults(lr_d=0.0001)
        parser.set_defaults(lr_g=0.0004)
        parser.set_defaults(lr_d=0.0004)
        #parser.set_defaults(lr_g=0.00005)
        #parser.set_defaults(lr_d=0.00005)
    if opt.dataset_mode == "cityscapes":
        #parser.set_defaults(lr_g=0.0004)
        parser.set_defaults(lr_g=0.0002)
        parser.set_defaults(lr_d=0.0002)
        parser.set_defaults(lambda_labelmix=5.0)
        parser.set_defaults(freq_fid=2500)
        parser.set_defaults(EMA_decay=0.999)
    if opt.dataset_mode == "coco":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
        parser.set_defaults(num_epochs=100)


def save_options(opt, parser):
    path_name = os.path.join(opt.checkpoints_dir,opt.name)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)


def update_options_from_file(opt, parser):
    new_opt = load_options(opt)
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_options(opt):
    file_name = os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    return new_opt


def load_iter(opt):
    if opt.which_iter == "latest":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    elif opt.which_iter == "best":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "best_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    else:
        return int(opt.which_iter)


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
