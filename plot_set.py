# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:36:16 2017

@author: chandler
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:34:05 2017

@author: chandler
"""
import caffe;
import cv2;
import numpy as np;
import math;
import argparse
import os.path;
argparse.ArgumentParser
parser = argparse.ArgumentParser(description="plot output landmark on source image");
parser.add_argument("imgPath",help="the path to source image");
parser.add_argument("modelPath",help="the path to caffe model(.caffemodel)");
args = parser.parse_args();

#prefix = os.path.splitext(os.path.basename(args.imgPath))[0]+"_iter"+(os.path.splitext(os.path.basename(args.modelPath))[0]).split("_")[2];
#prefix = os.path.splitext(os.path.basename(args.imgPath))[0]+"_iter"+(os.path.splitext(os.path.basename(args.modelPath))[0]).split("_")[2];
caffe.set_mode_gpu();

#model_path = "/home/tracking/work/git/caffe/models/hyperface/landmark_deploy.prototxt";
model_path = "/home/tracking/work/src/hyperface/kaggle/kaggle_deploy.prototxt";
#snapshot_path = "/home/tracking/work/git/caffe/models/alexnet_chandler/bvlc_alexnet.caffemodel";
#snapshot_path = "/home/tracking/work/git/caffe/models/hyperface/snapshot/landmark_iter_1000.caffemodel";


net = caffe.Net(str(model_path),str(args.modelPath),caffe.TEST);

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
#transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR

net.blobs['data'].reshape(1,        # batch 大小
                          1,         # 3-channel (BGR) images
                          96,96)  # 图像大小为:227x227
   
if not os.path.exists('set'):
    os.makedirs('set');
#image_name = "image71331.jpg";
                       
#image = caffe.io.load_image(args.imgPath);
#image = cv2.imread(args.imgPath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
test_set = np.load('test.npy');

test_set.reshape(test_set.shape[0],1,96,96)

for i in range(test_set.shape[0]):
#    print(test_set[i].shape)
    transformed_image = transformer.preprocess('data', test_set[i]);
    
    net.blobs['data'].data[...] = test_set[i].reshape(1,1,96,96)
    output = net.forward()
    
    #print(output.shape);
    output = output['fc3']
    
#    print(output);
    
    output = output*48+48;
    
#    print(output)
    
#    print(output)
    
    img = test_set[i].reshape(96,96,1)
    
    
    for j in range(0,output.shape[1],2):
        cv2.circle(img,
                   (int(math.floor(output[0][j])),
                    int(math.floor(output[0][j+1]))),3,(0,255,0),-1);                 
    cv2.imwrite("/home/tracking/work/src/hyperface/kaggle/plot/set/test_set_"+str(i)+".jpg",img);
    