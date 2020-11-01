# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 23:37:10 2020
3분 딥러닝 파이토치맛 Ch 3 
경사하강법으로 이미지 복원하기 실습 
@author: gh
"""


import PIL.Image as pilimg
import torch
import matplotlib.pyplot as plt
import numpy as np

image = pilimg.open('super.png')
image = image.resize((100,100))     #픽셀 수 줄이기 
pix = np.array(image)               
new_image = torch.from_numpy(pix)   #numpy to torch

plt.imshow(new_image.view(100,100))     #원본
new_image = new_image.view(10000)

def weird_function(x, n_iter = 5):
    h = x
    filt = torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0*0])
        h_l = torch.cat((zero_tensor, h[:-1]), 0)
        h_r = torch.cat((h[1:], zero_tensor), 0)
        h = filt[0]*h + filt[2] * h_l + filt[1] * h_r
        if i%2 == 0:
            h = torch.cat( (h[h.shape[0]//2:], h[:h.shape[0]//2]), 0)
    return h

broken_image = weird_function(new_image.float())    #오염 시키기
plt.imshow(broken_image.view(100,100))

def distance_loss(hypothesis, broken_image):
    return torch.dist(hypothesis, broken_image)

random_tensor = torch.randn(10000, dtype = torch.float)
lr = 20     #learning rate = 20

for i in range(0,20000):
    random_tensor.requires_grad_(True)
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    loss.backward()
    with torch.no_grad():
        random_tensor = random_tensor - lr*random_tensor.grad
    if i % 1000 == 0:
        print('Loss at {} = {}'.format(i, loss.item()))


plt.imshow(random_tensor.view(100,100).data)