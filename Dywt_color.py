from PIL import Image
import skimage as iaf

import matplotlib.pyplot as plt
import numpy as np

import cv2
import pywt
import pywt.data
import argparse


NO_OF_CHANNELS = 3

class ImageChannels():
    def __init__(self):
        self.img_channel_B = []
        self.img_channel_G = []
        self.img_channel_R = []

    def extractRGBValues(self, *args):
        input_image = args[0]

        self.img_channel_B = input_image[:, :, 2]
        self.img_channel_G = input_image[:, :, 1]
        self.img_channel_R = input_image[:, :, 0]
    

    def getB(self):
        return self.img_channel_B

    def getG(self):
        return self.img_channel_G

    def getR(self):
        return self.img_channel_R


def perform_dwt2(input_image):
    LL, (LH, HL, HH) = pywt.dwt2(input_image, 'db1')

    return {
        'LL': LL,
        'LH': LH,
        'HL': HL,
        'HH': HH
        }


def perform_dywt2(input_image):
    a = pywt.swt2(input_image, 'haar', level=1)
    LL, (LH, HL, HH) = a[0]

    return {
        'LL': LL,
        'LH': LH,
        'HL': HL,
        'HH': HH
        }

   

def reCreateImage(img_part, img_channel_B, img_channel_G, img_channel_R):
    width, height = img_channel_B.shape
    new_image = np.zeros((width, height, NO_OF_CHANNELS))
    new_image[:, :, 2] = img_channel_B
    new_image[:, :, 1] = img_channel_G
    new_image[:, :, 0] = img_channel_R

    if img_part == "LL":
        return (new_image / 255)
    else:
        return new_image
    

def convertarray(image,convert):
    height,width,contents=image.shape   
    new_image= np.zeros((height,width,contents), convert)
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,contents):
                new_image[i][j][k]=int(round(((image[i][j][k] - 0.0) / (1.0 - 0.0 ) * (255 - 0) + 0)))
    return new_image          


def clipp(image):
    height,width,contents=image.shape
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,contents):
                if(image[i][j][k] < 0.0):
                    image[i][j][k]=0.0
                elif(image[i][j][k]>1.0):
                    image[i][j][k]=1.0
    return image


def dywtColor(input_query):
    global NO_OF_CHANNELS
    input_image_RGB = input_query
    
    imageChannels = ImageChannels()
    imageChannels.extractRGBValues(input_image_RGB)

    wt_output_R = perform_dywt2(imageChannels.getR())
    wt_output_G = perform_dywt2(imageChannels.getG())
    wt_output_B = perform_dywt2(imageChannels.getB())

    redLL = wt_output_R['LL']
    redLH = wt_output_R['LH']
    redHL = wt_output_R['HL']
    redHH = wt_output_R['HH']

    greenLL = wt_output_G['LL']
    greenLH = wt_output_G['LH']
    greenHL = wt_output_G['HL']
    greenHH = wt_output_G['HH']
    
    blueLL = wt_output_B['LL']
    blueLH = wt_output_B['LH']
    blueHL = wt_output_B['HL']
    blueHH = wt_output_B['HH']

    new_img_LL = reCreateImage("LL", blueLL, greenLL, redLL)
    new_img_LH = reCreateImage("LH", blueLH, greenLH, redLH)
    new_img_HL = reCreateImage("HL", blueHL, greenHL, redHL)
    new_img_HH = reCreateImage("HH", blueHH, greenHH, redHH)

    
    #plot_image("DyWT2", input_image_RGB, new_img_LL, new_img_LH, new_img_HL, new_img_HH)
    return  convertarray(clipp(new_img_LL),np.uint8)
