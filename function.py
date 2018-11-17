
import cv2
import clustering1 as cl
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plotImage(title, original_img, LL, LH, HL, HH):
    plt.axis("off")
    plt.imshow(original_img)
    level = 0
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    fig = plt.figure()
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(2, 2, i + 1)
        plt.axis("off")
        ax.imshow(a)
        ax.set_title(titles[i], fontsize=12)

    fig.suptitle("{}, level {}".format(title, level), fontsize=12)
    level += 1

    plt.show()


def sort1(vector_object):
    vector_object=sorted(vector_object,key=lambda x: x.getDescription())
    return vector_object
def plotImage(title, original_img, LL, LH, HL, HH):
    plt.axis("off")
    plt.imshow(original_img)
    level = 0
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    fig = plt.figure()
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(2, 2, i + 1)
        plt.axis("off")
        ax.imshow(a)
        ax.set_title(titles[i], fontsize=12)

    fig.suptitle("{}, level {}".format(title, level), fontsize=12)
    level += 1

    plt.show()

def clipp(image):
    height,width=image.shape
    for i in range(0,height):
        for j in range(0,width):
                if(image[i][j] < 0.0):
                    image[i][j]=0.0
                elif(image[i][j]>1.0):
                    image[i][j]=1.0
    return image


def printFile(image,name):
    file=open(name,'w+')
    file.write(str(image))


def convertArray(image,convert):
    height,width=image.shape   
    new_image= np.zeros((height,width), convert)
    for i in range(0,height):
        for j in range(0,width):
          new_image[i][j]=int(round(((image[i][j] - 0.0) / (1.0 - 0.0 ) * (255 - 0) + 0)))
    return new_image          
