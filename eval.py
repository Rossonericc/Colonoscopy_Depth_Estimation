from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import math
import time
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

from data.oc_dataset import OC_Dataset
from oflow.optical_flow import warp_loss
from util.util import test
import torchvision.transforms as transforms

# opt = EvalOptions.parse()
opt = TrainOptions().parse()


def tensor2array(tensor):
    array = tensor.detach().cpu()
    # array = -(array.numpy())
    array = array.numpy()
    array = 0.5 + array * 0.5
    array = array.squeeze()

    return array


def eval(model):
    # load data
    data_path = './Colon_data/EndoSlam'
    dirs = ["eval_A"]
    trans = transforms.Compose([transforms.Resize((512, 512), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])
    mean_relative_l1_error = 0.0
    mean_l1_error = 0.0
    rmse = 0.0
    num = 1.0
    for d in dirs:
        path = os.path.join(data_path, d)
        imgs_path = os.listdir(path)
        for img in imgs_path:
            img_path = os.path.join(path, img)
            frame = Image.open(img_path).convert('RGB')

            frame = trans(frame)
            frame = frame.unsqueeze(dim=0)
            frame = frame.to(torch.device('cuda'))

            generated = model(label=frame, inst=None, image=None, feat=None, infer=None, is_OC=True)

            # resize to gt size 256*256
            predicted = tensor2array(generated[0])
            predicted = Image.fromarray(predicted)
            predicted = predicted.resize((256, 256), Image.BICUBIC)
            predicted = np.array(predicted)

            name = img.split("FrameBuffer")
            depth_name = name[0] + "Depth" + name[1]
            depth_path = os.path.join("./Colon_data/OC_depth/eval_B", depth_name)
            # ground_truth = Image.open(depth_path).resize((512, 512),Image.BICUBIC)
            ground_truth = Image.open(depth_path)
            gt = np.array(ground_truth).astype(np.float32) / 255.0

            # mean relative l1-error
            l1_error = abs(predicted - gt)
            rel_error = l1_error[gt != 0] / gt[gt != 0]
            # print(np.mean(rel_error))
            mean_relative_l1_error = mean_relative_l1_error + (np.mean(rel_error) - mean_relative_l1_error) / num

            # mean l1-error
            mean_l1_error = mean_l1_error + (np.mean(l1_error) - mean_l1_error) / num

            # mean RMSE
            rmse = rmse + (math.sqrt(np.mean(l1_error * l1_error)) - rmse) / num

            num = num + 1

    return mean_relative_l1_error, mean_l1_error, rmse


def main():
    model = create_model(opt)
    print(1)
    if opt.fp16:
        from apex import amp

        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D],
                                                           opt_level='O1')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    print(2)
    mean_relative_l1_error, mean_l1_error, r_mean_square_error = test(model)
    print("mean_relative_l1_error:", mean_relative_l1_error)
    print("mean_l1_error:", mean_l1_error)
    print("r_mean_square_error:", r_mean_square_error)


if __name__ == 'main':
    main()
