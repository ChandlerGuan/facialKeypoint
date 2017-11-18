# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 22:37:28 2017

@author: chandler
"""
#import sys
#sys.path.append('/home/chandlev/caffe/python/')
import caffe;
import numpy.random
import math;

#class SimpleLayer(caffe.Layer):
#    """A layer that just multiplies by ten"""
#
#    def setup(self, bottom, top):
#        pass
#
#    def reshape(self, bottom, top):
#        top[0].reshape(*bottom[0].data.shape)
#
#    def forward(self, bottom, top):
#        top[0].data[...] = 10 * bottom[0].data
#
#    def backward(self, top, propagate_down, bottom):
#        bottom[0].diff[...] = 10 * top[0].diff
        
class AugmentLayer(caffe.Layer):
    """A layer that do data augmentation for hdf5 dataset"""

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need both data and label input.")        
        
        params = eval(self.param_str)
        
        if 'mirror_rate' in params:
            self.mirror_rate = params['mirror_rate'];
        else:
            self.mirror_rate = 0.5;
            
        self.flip_indices = [
            (0, 2), (1, 3),
            (4, 8), (5, 9), (6, 10), (7, 11),
            (12, 16), (13, 17), (14, 18), (15, 19),
            (22, 24), (23, 25),
            ]


    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        batch_size = bottom[0].data.shape[0];
        
#        choose image to mirror based on mirror rate
        indices = numpy.random.choice(batch_size,int(math.floor(batch_size*self.mirror_rate)),replace=False);
        
#        mirror the selected images
        top[0].data[...] = bottom[0].data;
        top[0].data[indices] = top[0].data[indices,:,:,::-1];
        
#        reverse all the x coordinated
        top[1].data[...] = bottom[1].data;
        top[1].data[indices,::2] = top[1].data[indices,::2] * -1;
#        swap left and right organic
        for a,b in self.flip_indices:
            top[1].data[indices,a],top[1].data[indices,b] = (
                top[1].data[indices,b],top[1].data[indices,a]            
            )
        
    def backward(self, top, propagate_down, bottom):
        pass