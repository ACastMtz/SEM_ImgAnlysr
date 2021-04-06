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
    
    def __init__(self, path, size, tilt, magn, pitch, metric_size, verbosity):
        '''Defining metadata (instance attributes)'''
        self.path = path 
        self.size = size
        self.tilt = tilt 
        self.magn = magn
        self.pitch = pitch
        self.metric_size = metric_size
        self.verbosity = verbosity
        self.__imgWidth = size[0]
        self.__imgHeight = size[1]
        self.__nmToPx = 0
        
        return  
    
    
    def __str__(self):
        '''Print instance's info'''
        img_info = f" \
            Path: {self.path} \n \
            Size: {self.size} \n \
            Tilt: {self.tilt} degrees \n \
            Magnification: {self.magn}k \n \
            Pitch: {self.pitch} \n \
            Metric: {self.metric_size} \n \
            Width: {self.__imgWidth} \n \
            Height: {self.__imgHeight} \n \
            Verbosity: {self.verbosity} \n \
            "
        return img_info

    
    def img_loader(self):
        '''Load image'''
        return cv2.imread(self.path)

    
    def params_ctrl(self):
        '''Check image parameters'''
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

    
    def one(self):
        return int(0.588*self.__imgWidth)
    def two(self):
        return int(0.745*self.__imgWidth)
    def img_sizes(self, key):
        '''Switcher'''
        switcher = {
            0: lambda: 'zero',
            1: self.one,
            2: self.two
        }
        func = switcher.get(key, lambda: 'Not supported set of image parameters')
        return func()

    # ----------------------------------------------------------------------------------- #

    def img_processing(self):
        '''
        Method to crop, find length scale, etc. of the image.
        If non standard images are used, i.e. not taken with the PDI SEM
        the cropping proportions will have to be modified and the loacation
        of the reference metric (to draw a rectangle around it). 
        NOTE: The leftmost horizontal vertex is defined in the Switcher function.
        '''
        img = self.img_loader()
        img_h, img_w = img.shape[0:2]
        if [img_w, img_h] != self.size:
            raise ValueError('Please enter an accepted image size in pixels')
        key = self.params_ctrl()

        # Find length scale
        x1_ls = self.img_sizes(key)
        x2_ls = int(0.98*img_w)
        y1_ls = int(0.94*img_h)
        y2_ls = int(0.96*img_h)
        ls_width = x2_ls-x1_ls
        self.nmToPx = self.metric_size/ls_width
        rect_thickness = 5
        cv2.rectangle(img, (x1_ls,y1_ls), (x2_ls,y2_ls), (255, 0, 0), rect_thickness)

        # Crop image
        crop_h = img_h - int(img_h/10)
        img_cr = img[0:crop_h,:]

        return img, img_cr

    
    def img_filtering(self, n_clusters):
        '''Method to filter the image'''
        img, img_cr = self.img_processing()

        # KMeans thresholding (Pixel clustering)
        grayscale = cv2.cvtColor(img_cr , cv2.COLOR_BGR2GRAY)
        grayscale_n = grayscale.reshape(grayscale.shape[0]*grayscale.shape[1],1)/255
        clusterer = KMeans(n_clusters=n_clusters, random_state=0).fit(grayscale_n)
        threshold = np.sort(clusterer.cluster_centers_, axis=0)
        if self.verbosity == True:
            print('Threshold Center Values: \n {}'.format(threshold))
        
        # Binarize image
        thr = int(threshold[n_clusters-1][0]*255)
        ret, bw = cv2.threshold(grayscale, thr, 255, cv2.THRESH_BINARY)

        return grayscale_n, bw

    
    def connected_components(self, n_clusters):
        ''' Find connected components '''
        _, img_cr = self.img_processing()
        grayscale_n, bw, thr = self.img_filtering(n_clusters)

        connectivity = 8
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
        sizes = stats[1:, -1] 
        nb_components = nb_components - 1
        img = np.zeros((img_cr.shape), np.uint8)
        diag = []
        base = []
        height = []
        for i in range(0, nb_components+1):
            color = np.random.randint(255,size=3)
            x1 = stats[i][0]
            y1 = stats[i][1]
            x2 = stats[i][0]+stats[i][2]
            y2 = stats[i][1]+stats[i][3]
            diag.append(math.sqrt((x2-x1)**2+(y2-y1)**2))
            base.append(abs(x2-x1))
            height.append(abs(y2-y1))
            cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0), 2)
            img[output == i + 1] = color
        diag = np.asarray(diag)*self.nmToPx
        base = np.asarray(base)*self.nmToPx
        height = np.asarray(height)*self.nmToPx/math.sin(tilt*math.pi/180)
        if self.verbosity == True:
            print('Threshold: {}'.format(thr))
            print('Number of labels: {}'.format(nb_components))
            print('Diagonal array length: {}'.format(diag[1:].shape[0]))
            print('Base array length: {}'.format(base[1:].shape[0]))
            print('Height array length: {}'.format(height[1:].shape[0]))
        return img, diag, base, height


    def printer(self, num_plots):
        '''Method to print images'''
        img, img_cr = self.img_processing()
        grayscale_n , bw,_ = self.img_filtering(n_clusters=3)
        img_conComp,_,_,_ = self.connected_components(n_clusters=3)

        fig = plt.figure()
        ax1 = fig.add_subplot(2,num_plots/2,1)
        ax1.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        ax2 = fig.add_subplot(2,num_plots/2,2)
        ax2.imshow(img_cr)
        plt.title('Cropped Image')
        plt.axis('off')
        ax3 = fig.add_subplot(2,num_plots/2,3)
        counts , vals = np.histogram(grayscale_n, bins=500)
        plt.plot(np.linspace(0,1, num=500), counts, label='original')
        plt.title('Grayscale image histogram')
        plt.xlabel('Pixel intensity')
        plt.ylabel('Count')
        ax4 = fig.add_subplot(2,num_plots/2,4)
        ax4.imshow(img_conComp)
        plt.title('Detected Objects')
        plt.axis('off')
        plt.show()
        return


img_path = 'Images\InGaAs\M3_2390_50nm0.4um-15Tilt-10K.jpg'
size = [1280,960]
# size = [960,1280]
tilt = 15
magn = 25
pitch = 400
metric_size = 5000
num_plots = 4
verbosity = True

# Instance
test_img = NW_SEM_image(path=img_path, size=size, tilt=tilt, magn=magn, pitch=pitch, metric_size=metric_size, verbosity=verbosity)
print(test_img)

# Methods
test_img.printer(num_plots=num_plots)