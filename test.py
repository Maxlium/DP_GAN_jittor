import time
import warnings

import jittor as jt

import config
import dataloaders.dataloaders as dataloaders
import models.models as models
import utils.utils as utils

jt.flags.use_cuda = 1

warnings.filterwarnings("ignore")
#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
dataset_test = dataloaders.get_dataloaders(opt, load_type='test')
dataloader_test = dataset_test.set_attrs(
    batch_size = opt.batch_size, 
    shuffle = False, 
    drop_last=False, 
    num_workers=16
)
print("size test: %d " % (len(dataset_test)))
#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.DP_GAN_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

total_time = 0
#--- iterate over validation set ---#
for i, data_i in enumerate(dataloader_test):
    image, label = models.preprocess_input(opt, data_i)
    end = time.time()
    generated = model(image, label, "generate", None)

    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()

    t = time.time() - end
    total_time += t
    image_saver(label, generated, data_i["name"])

print("Avg time: ", total_time/len(dataloader_test))
