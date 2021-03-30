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
    def __init__(self, name, size, tilt, magn):
        self.name = name 
        self.size = size
        self.tilt = tilt 
        self.magn = magn
        return  
    
    '''Method to find adequate threshold for image filtering'''
    def img_threshold(self):
        
        return

    '''Method to filter the image'''
    def img_filtering(self):
        
        return

    '''Method to crop, find length scale, etc. of the image'''
    def img_processing(self):
        
        return

    '''Method to print images'''
    def printer(self):
        
        return