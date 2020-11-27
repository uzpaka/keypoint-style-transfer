import argparse
import numpy as np
import os
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm
import warnings

from lib.exceptions import NoGradientError
from lib.my_loss_job_dnim import loss_function
from lib.my_model import D2Net
from lib.ns_transformer_net import TransformerNet
from lib.ns_vgg import Vgg16
from lib.model_test import D2Net as D2Net_test

from lib.utils import preprocess_image
import cv2
import lib.ns_utils
import math

#Todo:  loading of image1.jpg, image2.jpg  (we transform image1, i.e. the style of image2 will be transferred to image1, so image1 should be night and image2 should be day).
#       saving image1.jpg_image2.jpg_original_matches_ninliers_.png
#       saving image1.jpg_image2.jpg_new_matches_ninliers_.png
#       saving numpy data (kp1 for image1 and kp2 for image2), note that both original and translator matches are only mutual nn matches! so they need to be filtered out with ransac to be useful
#       remove padding

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def tocuda(data):
    for key in data:
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()
    return data

# Seed
torch.manual_seed(1135) #was 1, switched to 1135
if use_cuda:
    torch.cuda.manual_seed(1135)
np.random.seed(1135)

# Argument parsing
parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('--preprocessing', type=str, default='caffe', help='image preprocessing (caffe or torch)')
parser.add_argument('--model_file', type=str, default='models/d2_tf.pth', help='path to d2-net model')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--output_path', type=str, default="output_results/", help='path to the output files')
parser.add_argument('--num_steps_per_image', type=int, default=800, help='total number of training steps per image')
parser.add_argument('--content_steps', type=int, default=400, help='the first number of steps where the model is trained only for the content loss')
parser.add_argument('--image1_path', type=str, default="input_images/dnim1.jpg", help='path to image with adverse conditions to be transformed (night image)')
parser.add_argument('--image2_path', type=str, default="input_images/dnim2.jpg", help='path to image with non-adverse conditions (day image)')

args = parser.parse_args()

learning_rate = args.lr
output_path = args.output_path
num_steps_per_image = args.num_steps_per_image
content_steps = args.content_steps
image1_path = args.image1_path
image2_path = args.image2_path

print(args)

if not os.path.isdir(output_path):
    os.makedirs(output_path)

model = D2Net(
    model_file=args.model_file,
    use_cuda=use_cuda
)

d2net_test_model = D2Net_test(
    model_file=args.model_file,
    use_relu=True,
    use_cuda=use_cuda
)

ns_transformer = TransformerNet().to(device)
ns_vgg = Vgg16(requires_grad=False).to(device)

# Optimizer
params = list(filter(lambda p: p.requires_grad, ns_transformer.parameters()))
optimizer = optim.Adam(params, lr=learning_rate)

def main():           
    process_image(image1_path, image2_path, model, ns_transformer, ns_vgg, d2net_test_model, loss_function, optimizer, device)


def process_image(img1_path, img2_path, model, ns_transformer, ns_vgg, d2net_test_model, loss_function, optimizer, device, train=True):
    torch.set_grad_enabled(train)
    batch = {}
    batch['preprocessing'] = args.preprocessing
    preprocessing = batch['preprocessing']

    #img1_path = images_path + img1_name
    #img2_path = images_path + img2_name
    
    img1_original = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2_original = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    #img1 = cv2.resize(img1_original, (800,800))
    #img2 = cv2.resize(img2_original, (800,800))
    
    factor1 = (800.0*800.0) / (img1_original.shape[0]*img1_original.shape[1])
    factor2 = (800.0*800.0) / (img2_original.shape[0]*img2_original.shape[1])
    
    factor1 = max(factor1, 800.0/max(img1_original.shape[0], img1_original.shape[1]))
    factor2 = max(factor2, 800.0/max(img2_original.shape[0], img2_original.shape[1]))

    #img1_original = np.copy(img1)
    #img2_original = np.copy(img2)

    if factor1 < 1.0:
        #need to resize image1
        new_h = int(img1_original.shape[0]*factor1)
        new_w = int(img1_original.shape[1]*factor1)
        new_h = int(math.ceil((new_h+0.0)/4)*4) #This will make sure that the size is a multiple of 4
        new_w = int(math.ceil((new_w+0.0)/4)*4)
        img1 = cv2.resize(img1_original, (new_w, new_h))
        print("Image 1 had to be resized from " + str(img1_original.shape) + " to " + str(img1.shape))
    else:
        #You don't need to resize, but still you need to pad the image to next multiple of 4 (because the transformer pools the size by a factor of 4 before upsampling, so this will preserve the original shape)
        h, w = img1_original.shape[0:2]
        new_h = int(math.ceil((h+0.0)/4)*4)
        new_w = int(math.ceil((w+0.0)/4)*4)
        
        img1 = np.zeros((new_h, new_w, 3), dtype = img1_original.dtype)        
        img1[0:h, 0:w, :] = img1_original
        print("Image 1 had to be zero-padded from " + str(img1_original.shape) + " to " + str(img1.shape))
        
    if factor2 < 1.0:
        #need to resize image2
        new_h = int(img2_original.shape[0]*factor2)
        new_w = int(img2_original.shape[1]*factor2)
        new_h = int(math.ceil((new_h+0.0)/4)*4) #This will make sure that the size is a multiple of 4
        new_w = int(math.ceil((new_w+0.0)/4)*4)
        img2 = cv2.resize(img2_original, (new_w, new_h))
        print("Image 2 had to be resized from " + str(img2_original.shape) + " to " + str(img2.shape))
    else:
        h, w = img2_original.shape[0:2]
        new_h = int(math.ceil((h+0.0)/4)*4)
        new_w = int(math.ceil((w+0.0)/4)*4)
        
        img2 = np.zeros((new_h, new_w, 3), dtype = img2_original.dtype)        
        img2[0:h, 0:w, :] = img2_original
        print("Image 2 had to be zero-padded from " + str(img2_original.shape) + " to " + str(img2.shape))
    
    batch['image1_rgbraw'] = [img1]
    batch['image2_rgbraw'] = [img2]

    batch['image1_rgbraw_original'] = [img1_original]
    batch['image2_rgbraw_original'] = [img2_original]

    batch['image1'] = torch.stack( [ torch.from_numpy(preprocess_image(np.copy(img1), preprocessing=preprocessing).astype(np.float32)) ] )
    batch['image2'] = torch.stack( [ torch.from_numpy(preprocess_image(np.copy(img2), preprocessing=preprocessing).astype(np.float32)) ] )

    batch['image_path1'] = img1_path
    batch['image_path2'] = img2_path

    batch["image_name1"] = img1_path #img1_name
    batch["image_name2"] = img2_path #img2_name

    batch = tocuda(batch)

    #We will transfer the style of the second image to the first image. So translate/transform the first image
    style = np.transpose(batch['image2_rgbraw'][0], [2, 0, 1]) # H,W,C t o C,H,W
    style = torch.from_numpy(style).unsqueeze(0).cuda().float()
    features_style = ns_vgg(lib.ns_utils.normalize_batch(style))
    gram_style = [lib.ns_utils.gram_matrix(y) for y in features_style]
    transformer_input = torch.from_numpy(np.transpose(batch['image1_rgbraw'][0], [2, 0, 1])).cuda().float()
    transformer_input = transformer_input.unsqueeze(0)
    transformer_input_vgg = lib.ns_utils.normalize_batch(transformer_input)
    features_tin = ns_vgg(transformer_input_vgg)

    progress_bar = tqdm(range(num_steps_per_image))
    my_global_step = 0
    for batch_idx in progress_bar:
    #for batch_idx in range(num_steps_per_image):
        if train:
            optimizer.zero_grad()
        try:
            loss = loss_function(model, ns_transformer, ns_vgg, d2net_test_model, features_tin, gram_style, transformer_input, batch, device, my_global_step=my_global_step, num_steps_per_image=num_steps_per_image, content_steps=content_steps, output_path=output_path)
            my_global_step += 1
        except NoGradientError:
            continue

        #current_loss = loss.data.cpu().numpy()[0] #Make sure to write [0] when using with the D2-Net loss
        #epoch_losses.append(current_loss)
        #progress_bar.set_postfix(loss=('%.4f' % np.mean(epoch_losses)))
        
        if train:
            loss.backward()
            optimizer.step()


main()

