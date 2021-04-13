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
from sklearn import preprocessing

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
        self.__imgCrSize = []
        
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

    def img_processing(self,img):
        '''
        Method to crop, find length scale, etc. of the image.
        If non standard images are used, i.e. not taken with the PDI SEM
        the cropping proportions will have to be modified and the loacation
        of the reference metric (to draw a rectangle around it). 
        NOTE: The leftmost horizontal vertex is defined in the Switcher function.
        '''
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
        self.__imgCrSize = img_cr.shape

        return img_cr

    
    def img_filtering(self, img, n_clusters):
        '''Method to filter the image'''

        # KMeans thresholding (Pixel clustering)
        grayscale = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        grayscale_n = grayscale.reshape(grayscale.shape[0]*grayscale.shape[1],1)/255
        clusterer = KMeans(n_clusters=n_clusters, random_state=0).fit(grayscale_n)
        threshold = np.sort(clusterer.cluster_centers_, axis=0)
        if self.verbosity:
            print('Threshold Center Values: \n {}'.format(threshold))
        
        # Binarize image
        thr = int(threshold[n_clusters-1][0]*255)
        ret, bw = cv2.threshold(grayscale, thr, 255, cv2.THRESH_BINARY)

        return grayscale, grayscale_n, bw, thr

    
    def connected_components(self, bw, n_clusters):
        ''' Find connected components '''
        connectivity = 8
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
        sizes = stats[1:, -1] 
        nb_components = nb_components - 1
        img = np.zeros((self.__imgCrSize), np.uint8)
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
        if self.verbosity:
            print('Threshold: {}'.format(thr))
            print('Number of labels: {}'.format(nb_components))
            print('Diagonal array length: {}'.format(diag[1:].shape[0]))
            print('Base array length: {}'.format(base[1:].shape[0]))
            print('Height array length: {}'.format(height[1:].shape[0]))
        base = base[1:].reshape(base[1:].shape[0],1)
        height = height[1:].reshape(height[1:].shape[0],1)

        return img, diag, base, height

    def results_clustering(self, size_centers, rect_labels):
        points_per_cluster = []
        val_cluster = []
        for i in range(size_centers.shape[0]):
            p_per_cl = rect_labels[rect_labels == i]
            points_per_cluster.append(len(p_per_cl))
            x = size_centers[i][0]
            y = size_centers[i][1]
            val_cluster.append(math.sqrt(x**2 + y**2))      # Distance of cluster center from origin
        if self.verbosity:
            print('Size Center Values: \n {}'.format(size_centers))
        df = pd.DataFrame(
            list(zip(val_cluster, points_per_cluster)),
            columns=['Cluster Value','Num. Elements']).sort_values(by=['Cluster Value'], ascending=False)
        
        return df

    def NW_stats(self, grayscale, df, top_clusters=[1,2]):
        # NW Stats
        pitch = self.pitch/self.nmToPx    #  nm per pixel
        col_vert = int(grayscale.shape[0]/pitch)
        row_vert = int(grayscale.shape[1]/pitch)
        num_holes = col_vert*row_vert
        NW_num = df['Num. Elements'].iloc[top_clusters].sum()
        NW_ind = top_clusters
        widths = base[1:]
        heights = height[1:]
        print(NW_ind)
        NW_height = np.mean(heights[rect_labels == NW_ind[0]])
        NW_height_un = np.std(heights[rect_labels == NW_ind[0]])
        NW_width = np.mean(widths[rect_labels == NW_ind[0]])
        NW_width_un = np.std(widths[rect_labels == NW_ind[0]])
        print(df.iloc[top_clusters])
        if self.verbosity:
            print('Ratio nm/pixel: {:.2f}'.format(self.nmToPx))
            print('Pitch [pixels]: {:.2f}'.format(pitch))
            print('Row vert.: {}'.format(row_vert))
            print('Column vert.: {}'.format(col_vert))
            print('Number of holes: {}'.format(num_holes))
            print('Number of NWs: {}'.format(NW_num))
            print('Vertical yield: {:.2%}'.format(NW_num/num_holes))
        results_ind = ['Vertical Yield', 'Avg. Height[nm]', 'Avg. Diameter [nm]']
        results_cols = ['Value', 'Uncertainty']
        results_vals = [
            [NW_num/num_holes, 'NaN'],
            [NW_height, NW_height_un],
            [NW_width, NW_width_un]
            ]
        results_df = pd.DataFrame(
            data=results_vals,
            index=results_ind,
            columns=results_cols
        )

        return results_df

    def printer(self, num_plots):
        '''Method to print images'''
        img = self.img_loader()
        img_cr = self.img_processing(img)
        _, grayscale_n , bw,_ = self.img_filtering(img=img_cr, n_clusters=3)
        img_conComp,_,_,_ = self.connected_components(bw=bw, n_clusters=3)

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

def det_obj_clustering(base, height, n_clusters):
    '''KMeans clustering for detecting rectangles'''
    # diag_2d = diag[1:].reshape(diag[1:].shape[0],1)
    X = np.concatenate((height, base), axis=1)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    kmeans_model =KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_kmeans = kmeans_model.predict(X)
    size_centers = kmeans_model.cluster_centers_
    rect_labels = kmeans_model.labels_

    return size_centers, rect_labels

def autolabels(freq,bins,patches):
    '''
    Annotate graph points with the sample's name
    '''
    bin_centers = np.diff(bins)*0.5 + bins[:-1]
    n = 0
    for _,j,_ in zip(freq,bin_centers,patches):
        h = int(freq[n])
        plt.annotate(
            '{}'.format(h),
            xy = (j, h),
            xytext = (0,0),
            textcoords = 'offset points',
            ha = 'center', va = 'bottom',
            color='r'
        )
        n += 1

#===============================================================================================#

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
img = test_img.img_loader()
img_cr = test_img.img_processing(img)
grayscale, grayscale_n , bw, thr = test_img.img_filtering(img=img_cr, n_clusters=3)
img_conComp, diag, base, height = test_img.connected_components(bw=bw, n_clusters=3)
size_centers, rect_labels = det_obj_clustering(base, height, n_clusters=3)
df = test_img.results_clustering(size_centers, rect_labels)
print(df)

# From analysis define number of top_clusters
top_clusters = [1,2]

# Plots
test_img.printer(num_plots=num_plots)