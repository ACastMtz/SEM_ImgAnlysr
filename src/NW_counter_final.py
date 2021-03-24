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
    def __init__(self, name, size):
        self.name = name 
        self.size = size
        return  
    
    '''Method to print images'''
    def printer(self):
        
        return