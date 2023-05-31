import pickle as p
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

import cv2

def img_save(x,n_pic,method,ifchange=0):
    for i in range(n_pic):
        imgs = x[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB",(i0,i1,i2))
        name = "img" + str(i) + '_'+ method +".png"
        img.save("../pics/"+name,"png")


n_pic = 5 # num of pictures to be augmented

# load single batch of cifar 
with open("../data/cifar-100-python/test", 'rb')as f:
    datadict = p.load(f, encoding='bytes')
    X = datadict[b'data']
    Y = datadict[b'fine_labels']
    X = X.reshape(10000, 3, 32, 32)
    Y = np.array(Y)

ranlist = [84,9,10,2000,8435]
x = X[ranlist]
y = Y[ranlist]


# cutout
mask = np.ones((32, 32), np.float32)
y0 = 8#np.random.randint(32)
x0 = 8#np.random.randint(32)
y1 = np.clip(y0 - 16 // 2, 0, 32)
y2 = np.clip(y0 + 16 // 2, 0, 32)
x1 = np.clip(x0 - 16 // 2, 0, 32)
x2 = np.clip(x0 + 16 // 2, 0, 32)
mask[y1: y2, x1: x2] = 0.
print(mask)
#mask = torch.from_numpy(mask)
#mask = mask.expand_as(torch.zeros(32,32))
#mask = np.expand_dims(mask,axis=-1)
x_cutout = x #* mask#.numpy()

for i in range(n_pic):
    imgs = x#(x_cutout[i]*255).astype(np.uint8)
    img0 = imgs[0] * mask#.numpy()
    img1 = imgs[1] * mask#.numpy()
    img2 = imgs[2] * mask#.numpy()
    print(img0.shape)
    img0 = np.uint8(img0).reshape(32,32,3)
    img1 = np.uint8(img1).reshape(32,32,3)
    img2 = np.uint8(img2).reshape(32,32,3)
    i0 = Image.fromarray(img0)
    i1 = Image.fromarray(img1)
    i2 = Image.fromarray(img2)
    img = Image.merge("RGB",(i0,i1,i2))
    name = "img" + str(i) + '_cutout' +".png"
    img.save("../pics/"+name,"png")




# baseline
img_save(x,n_pic,'baseline')

# baseline+
for i in range(n_pic):
    imgs = x[i]
    img0 = imgs[0]
    img1 = imgs[1]
    img2 = imgs[2]
    i0 = Image.fromarray(img0)
    i1 = Image.fromarray(img1)
    i2 = Image.fromarray(img2)
    img = Image.merge("RGB",(i0,i1,i2))
    #transforms.ToTensor()(img)
    img = transforms.RandomCrop(size=32, padding=4)(img)
    img = transforms.RandomVerticalFlip()(img)
    img = transforms.RandomHorizontalFlip()(img)
    name = "img" + str(i) + '_baseline+' +".png"
    img.save("../pics/"+name,"png")


# mixup
alpha = 0.5
#lam = np.random.beta(alpha, alpha)
lam = 0.7
index = torch.randperm(n_pic)
x_mixup = np.int32(lam*x + (1-lam)*x[index, :])
x_mixup = x_mixup.astype(np.uint8)
for i in range(n_pic):
    imgs = x_mixup[i]
    img0 = imgs[0]
    img1 = imgs[1]
    img2 = imgs[2]
    i0 = Image.fromarray(img0)
    i1 = Image.fromarray(img1)
    i2 = Image.fromarray(img2)
    img = Image.merge("RGB",(i0,i1,i2))
    name = "img" + str(i) + '_mixup' +".png"
    img.save("../pics/"+name,"png")

# cutmix
W = 32
H = 32
alpha = 0.5    
lam = 0.75#np.random.beta(alpha, alpha)
rand_index = torch.randperm(n_pic)#.cuda()
cut_rat = np.sqrt(1. -lam)
cut_w = np.int32(W * cut_rat)
cut_h = np.int32(H * cut_rat)
cx = np.random.randint(W) # uniform
cy = np.random.randint(H)
bbx1 = np.clip(cx - cut_w // 2, 0, W)
bby1 = np.clip(cy - cut_h // 2, 0, H)
bbx2 = np.clip(cx + cut_w // 2, 0, W)
bby2 = np.clip(cy + cut_h // 2, 0, H)
x_cutmix = x
x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
img_save(x_cutmix,n_pic,'cutmix')

