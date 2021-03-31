import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from timeit import default_timer as timer
import cv2 
import itertools
import math
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

class NW_SEM_image:
    '''Defining metadata (instance attributes)'''
    def __init__(self, path, size, tilt, magn, pitch, metric_size):
        self.path = path 
        self.size = size
        self.tilt = tilt 
        self.magn = magn
        self.pitch = pitch
        self.metric_size = metric_size
        self.__imgWidth = self.size[1]
        self.__imgHeight = self.size[0]
        return  
    
    '''Print instance's info'''
    def __str__(self):
        img_info = f" \
            Path: {self.path} \n \
            Size: {self.size} \n \
            Tilt: {self.tilt} degrees \n \
            Magnification: {self.magn}k \n \
            Pitch: {self.pitch} \n \
            Metric: {self.metric_size} \n\
            Width: {self.__imgWidth} \n \
            Height: {self.__imgHeight} \n\
            "
        return img_info

    '''Check image parameters'''
    def params_ctrl(self):
        case_1 = [1280,960,25]
        case_2 = [1280,960,6]
        params_list = [self.__imgWidth, self.__imgHeight, self.magn]
        if params_list == case_1:
            key = 1
        elif params_list == case_2:
            key = 2
        else:
            raise ValueError('Not supported set of image parameters')
        return key

    '''Switcher'''
    def one(self):
        return int(0.588*self.__imgWidth)
    def two(self):
        return int(0.745*self.__imgWidth)
    def img_sizes(self, key):
        switcher = {
            0: lambda: 'zero',
            1: self.one,
            2: self.two
        }
        func = switcher.get(key, lambda: 'Not supported set of image parameters')
        return func()

    
    def img_processing(self):
        '''
        Method to crop, find length scale, etc. of the image.
        If non standard images are used, i.e. not taken with the PDI SEM
        the cropping proportions will have to be modified and the loacation
        of the reference metric (to draw a rectangle around it). 
        NOTE: The leftmost horizontal vertex is defined in the Switcher function.
        '''
        img = cv2.imread(self.path)
        img_h, img_w = img.shape[0:2]
        if [img_h, img_w] != self.size:
            raise ValueError('Please enter an accepted image size in pixels')
        key = self.params_ctrl()

        # Find length scale
        x1_ls = self.img_sizes(key)
        x2_ls = int(0.98*img_w)
        y1_ls = int(0.94*img_h)
        y2_ls = int(0.96*img_h)
        ls_width = x2_ls-x1_ls
        nm_per_px = self.metric_size/ls_width
        rect_thickness = 5
        cv2.rectangle(img, (x1_ls,y1_ls), (x2_ls,y2_ls), (255, 0, 0), rect_thickness)

        # Crop image's height by 10%
        crop_h = img_h - int(img_h/10)
        img_cr = img[0:crop_h,:]

        return img, img_cr
       
        
# num_clusters = 6
# top_clusters = [1,2]

    '''Method to find adequate threshold for image filtering'''
    def img_threshold(self):
        
        return

    '''Method to filter the image'''
    def img_filtering(self):
        
        return

    '''Method to print images'''
    def printer(self, num_plots):
        img, img_cr = self.img_processing()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,num_plots,1)
        ax1.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        ax2 = fig.add_subplot(1,num_plots,2)
        ax2.imshow(img_cr)
        plt.title('Cropped Image')
        plt.axis('off')
        plt.show()
        return


img_path = 'Images\InGaAs\M3_2390_50nm0.4um-15Tilt-10K.jpg'
# size = [1280,960]
size = [960,1280]
tilt = 15
magn = 25
pitch = 400
metric_size = 5000

# Instance
test_img = NW_SEM_image(path=img_path, size=size, tilt=tilt, magn=magn, pitch=pitch, metric_size=metric_size)
print(test_img)

# Methods
# test_img.img_processing()
test_img.printer(num_plots=2)