import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance
import numpy as np
import random
import matplotlib.colors as cl
import os
import time
import gc

kp=[]
THRESHOLD=0
TABLE=[]
TABLE.append([])

class kp1:
    def __init__(self,x,y):
        self.pt=[]
        self.pt.append(x)
        self.pt.append(y)
class cluster_obj:
    
   def  __init__(self, cluster,contentx,contenty,centroid,size):
        self.cluster=cluster
        self.contentx=[]
        self.contenty=[]
        if type(contentx) is int:    
            self.contentx.append(contentx)
        else:
            self.contentx=contentx
        if type(contenty) is int:
            self.contenty.append(contenty)
        else:
            self.contenty=contenty
        self.size=size
        self.centroid=centroid
def initialize():
    global TABLE
    global THRESHOLD
    global kp
    kp=[]
    THRESHOLD=0
    TABLE=[]
    TABLE.append([])
     
def heirarchyCluster(kp,local_THRESHOLD):
    global TABLE
    global THRESHOLD
    initialize()
    THRESHOLD=local_THRESHOLD
    emptyTable(len(kp))
    createClusters(kp)
    buildTable(kp)
    while(True):
        minimum_distance=minimumDistance()
        if(minimum_distance['minimum_distance']>THRESHOLD or len(TABLE)==2):
            vector= TABLE[0]
            freeMemory(vector)
            return vector
        ith=minimum_distance['i']
        jth=minimum_distance['j']
        cluster=mergeCluster(TABLE[ith][0],TABLE[0][jth])        
        updateTable(cluster,minimum_distance)
def createClusters(kp):
    global TABLE
    len_of_kp=len(kp)
    for i in range(0,len_of_kp):
        kpx=kp[i].pt[0]
        kpy=kp[i].pt[1]
        key_points=[kpx,kpy]
        x1=kp[i].pt[0]
        y1=kp[i].pt[1]
        cluster=cluster_obj(None,x1,y1,key_points,1)
        TABLE[i+1].append(cluster)
        TABLE[0].append(cluster)  
def buildTable(kp):
    global TABLE
    len_of_kp=len(kp)
    for i in range(1,len_of_kp+1):
        for j in range(1,len_of_kp+1):
            if(i==j):
                dist=sys.maxsize
            else:
                dist=distance.euclidean(kp[i-1].pt,kp[j-1].pt)            
            TABLE[i].append(dist)
    print("buildTable %s"%(len(TABLE)))
def updateTable(cluster,minimumDistance):
    global TABLE
    removeClusterFromTable(minimumDistance['i'],minimumDistance['j'])
    length_of_table=len(TABLE)
    array=[]
    
    TABLE[0].append(cluster)
    array.append(cluster)
    for i in range(1,length_of_table):
        dist=distance.euclidean(TABLE[i][0].centroid,cluster.centroid)
        TABLE[i].append(dist)
        array.append(dist)
    array.append(sys.maxsize)
    TABLE.append(array)    
def minimumDistance():
    global TABLE
    x=-1
    y=-1
    length=len(TABLE)
    minimum_distance=sys.maxsize
    for i in range(1,length):
        for j in range(1,i):
            if TABLE[i][j]<=minimum_distance:
                minimum_distance=TABLE[i][j]
                x=i
                y=j
    return {
        'minimum_distance':minimum_distance,
        'i':x,
        'j':y
        }
def mergeCluster(cluster1,cluster2):
    global TABLE
    cluster_centroid=[]
    
    contentx=[]
    contenty=[]
    
    cluster1x_array=np.asarray(cluster1.contentx)
    cluster1y_array=np.asarray(cluster1.contenty)
    cluster1_size=cluster1x_array.size

    cluster2x_array=np.asarray(cluster2.contentx)
    cluster2y_array=np.asarray(cluster2.contenty)
    cluster2_size=cluster2x_array.size

    for i in range(0,cluster1_size):
        contentx.append(cluster1x_array[i])
        contenty.append(cluster1y_array[i])
    for j in range(0,cluster2_size):
        contentx.append(cluster2x_array[j])
        contenty.append(cluster2y_array[j])
    
   

    size_of_content_cluster1=cluster1.size
    size_of_content_cluster2=cluster2.size
    size_of_content_cluster=size_of_content_cluster1+size_of_content_cluster2
    c1_x=cluster1.centroid[0]*size_of_content_cluster1
    c2_x=cluster2.centroid[0]*size_of_content_cluster2
    c1_y=cluster1.centroid[1]*size_of_content_cluster1
    c2_y=cluster2.centroid[1]*size_of_content_cluster2

    cluster_centroidx=(c1_x+c2_x)/size_of_content_cluster
    cluster_centroidy=(c1_y+c2_y)/size_of_content_cluster
    
    cluster_centroid.append(cluster_centroidx)
    cluster_centroid.append(cluster_centroidy)
    new_cluster=cluster_obj(None,contentx,contenty,cluster_centroid,size_of_content_cluster)
    
    cluster1=None
    cluster2=None
    del cluster2
    del cluster1
    gc.collect()
    return new_cluster
def emptyTable(m):
    global TABLE
    
    for i in range (0,m):
        TABLE.append([])
    TABLE[0].append(None)
    print("empty table %s"%len(TABLE))   
def removeClusterFromTable(i,j):
    global TABLE
    if(i<j):
        temp=i
        i=j
        j=temp
    del TABLE[i]
    del TABLE[j]
    for row in TABLE:
        del row[i]
        del row[j]
    return TABLE
def printFile(F,name):
    file=open(name+".txt",'w+')
    file.write(str(F))
def freeMemory(vector):
    global TABLE
    TABLE=None
    del TABLE
    for i in range(1,len(vector)):
        vector[i].cluster=None
    gc.collect()