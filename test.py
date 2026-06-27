# encoding=UTF-8
import torch
import torch.nn.functional as F
import numpy as np
import os
import torchvision.utils as vutils
from torch.autograd import Variable
from evaluate.data import test_dataset
import time
import torchvision.transforms as transforms
from model.CTIFNet import CTIFNet
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='source dataset path')
parser.add_argument('--gt', type=str, help='ground truth dataset path')
parser.add_argument('--target', type=str, help='target dataset path')

args = parser.parse_args()
args.testsize = 224
args.trainset = 'DUTS-TR'
args.pre_model_path = './models/CTIFNet.pth'
args.mae_model_path = './models/checkpoint/mae_pretrain_vit_large.pth'

image_path = args.source if args.source.endswith('/') else args.source + '/'
gt_path = args.gt if args.gt.endswith('/') else args.gt + '/'
target_path = args.target if args.target.endswith('/') else args.target + '/'
eval_path = target_path + 'evaluation.txt'

model = CTIFNet(args)
model.cuda()

model.load_state_dict(torch.load(args.pre_model_path))
print("====== load model from {} success! ======".format(args.pre_model_path))

test_loader = test_dataset(image_path, gt_path, args.testsize)
model.eval()

os.makedirs(target_path, exist_ok=True)

torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    for i in range(test_loader.size):
        image_tran, gt, orignin_image, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = Variable(image_tran).cuda()
        result_res, result_trans = model(image)

        result_res = F.interpolate(result_res, size=gt.shape[:2], mode='bilinear', align_corners=False)
        result_trans = F.interpolate(result_trans, size=gt.shape[:2], mode='bilinear', align_corners=False)
        temp_name, _ = os.path.splitext(name)

        vutils.save_image(result_res, target_path + temp_name + ".png")

torch.cuda.synchronize()
end_time = time.time()

total_time = end_time - start_time
images_processed = test_loader.size
average_time_per_image = total_time / images_processed if images_processed > 0 else 0

with open(eval_path, 'w') as f:
    f.write(f"Total time: {total_time:.4f} seconds\n")
    f.write(f"Images processed: {images_processed}\n")
    f.write(f"Average time per image: {average_time_per_image:.4f} seconds\n")



