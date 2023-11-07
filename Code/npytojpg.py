#!/usr/bin/env python
# coding: utf-8

from Path import *

import numpy as np
import pandas as pd

import os
from skimage.transform import resize
import skimage.io
import cv2
#from tensorflow.keras.utils import array_to_img
from PIL import Image as im

from keras.models import *
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow.keras.backend as K

im_height = 128
im_width = 128
im_depth = 128

def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.4)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 4)
    conv6 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv3D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv3D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv3D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv3D(5, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def dice_coef(y_true, y_pred, smooth=1):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
def dice_coef_loss(y_true, y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (1-(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return dice
from tensorflow.keras.losses import binary_crossentropy
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) +dice_coef_loss(y_true, y_pred)*2


def get_depth(path):
    img = np.load(path)
    img = img.transpose()
    depth=img.shape[0]
    return depth



def get_image(img,export_path):
    
    #image=np.zeros((img.shape[0], img.shape[1], img.shape[2],1), dtype=np.int16)
    #image=resize(img, (img.shape[0], img.shape[1], img.shape[2], 1), mode = 'constant', preserve_range=True)
    #array_to_img(image[70])
    for i in range(img.shape[0]):
        img1=resize(img[i], (img.shape[1], img.shape[2]), mode = 'constant', preserve_range=True)
        data = im.fromarray(img1) 
        # saving the final output  
        # as a jpg file 
        data = data.convert('RGB')
        data.save(export_path+str(i)+'.jpg')

def get_image2(img,img2,export_path):
    
    #image=np.zeros((img.shape[0], img.shape[1], img.shape[2],1), dtype=np.int16)
    #image=resize(img, (img.shape[0], img.shape[1], img.shape[2], 1), mode = 'constant', preserve_range=True)
    #array_to_img(image[70])
    img=img+img2*10000
    for i in range(img.shape[0]):
        img1=resize(img[i], (img.shape[1], img.shape[2]), mode = 'constant', preserve_range=True)
        data = im.fromarray(img1) 
        # saving the final output  
        # as a jpg file 
        data = data.convert('RGB')
        data.save(export_path+str(i)+'.jpg')


def predict(img):
    
    print('1')
    final=np.zeros(shape=(img.shape[0],img.shape[1],img.shape[2]),dtype = int)
    # Cropping the image into the necessary area
    image=img[50:178,150:450,150:400]
    print('2')
    # Intensity based segmentation in the range [-300, 300]
    image=(image+200)
    mask=np.logical_and(image>0, image<1800) 
    image=image*mask
    # Normalizing
    image=(image)/1800
    print('3')
    # Extracting the required area
    #image=(image*mask2)
    x_img = resize(image, (im_height, im_width, im_depth, 1), mode = 'constant', preserve_range=True)
    x_img=np.array([x_img])
    print('model 1')
    #best_model=tf.keras.models.load_model('/kaggle/input/model-ckpt/model.ckpt')
    model=unet(input_size=(im_height, im_width, im_depth, 1))
    print('model 2')
    checkpoint = tensorflow.train.Checkpoint(model)
    print('model 3')
    checkpoint.restore(path+'Model\\model.ckpt')
    
    print('restored model')
    
    pred= model.predict(x_img)*255
    print(np.min(x_img),np.max(x_img))
    pred=pred
    print('predicted')
    x_img = resize(pred[0], (image.shape[0],image.shape[1],image.shape[2]), mode = 'constant', preserve_range=False)
    print(x_img.shape,type(x_img),image.shape[0],image.shape[1],image.shape[2])
    final[50:178,150:450,150:400] =[[[k[0] for k in j] for j in i] for i in x_img]
    print(np.min(final),np.max(final))
    print(final.shape)
    final=final
    #final=pred
    return final


#get_depth("/kaggle/input/pancreasimage/0001.npy")



#get_image("C:\\Users\\bjaya\\Downloads\\application\\0001np\\0001.npy",10)