#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 09:29:06 2018

@author: BrandoMora
"""

#TO-DO
'''
• center of mass processing for image
• Make it speak to you




'''


import cv2
import time
import keras
import numpy as np


# set up cameras
cap = cv2.VideoCapture(0)
cap_width = int(cap.get(3)) # int
cap_height = int(cap.get(4))  

# buffer for cropping
bfffc = int((cap_width - cap_height) // 2)



start_time = time.time()

process_again_flag = True


answer_key = (0,1,2,3,4,5,6,7,8,9)

#importing model
model = keras.models.load_model("mnistNet.hdf5")


def process_image(og_frame):
    #converting to greyscale
    frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2GRAY)
    
    
    
    # cropping and squishing down to 28 x 28             currently took out centering      gray_frame[0:720, bfffc:bfffc + 720] 
    frame = cv2.resize(frame[cap_height // 2:cap_height, 0:cap_height // 2], (28,28), interpolation = cv2.INTER_AREA)
    
    
    # thresholding currently adaptive
    #_, frame = cv2.threshold(squished_frame, 122, 255, cv2.THRESH_TOZERO_INV)
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3 , 11)
    
    
    frame1 = cv2.GaussianBlur(frame,(3,3),0)
    _, frame2 = cv2.threshold(frame1, 1, 255, cv2.THRESH_BINARY)
    frame3 = cv2.GaussianBlur(frame2,(3,3),0)
    
    
    
    
    
    return frame3






while True:
    _, og_frame = cap.read()
    
    frame = process_image(og_frame)
    
    cv2.rectangle(og_frame, (cap_height // 2, cap_height), (0, cap_height // 2), (255,0,0), 5) 
    
    
    cv2.imshow("Original", cv2.flip( cv2.resize(og_frame, (0,0), fx=0.4, fy = 0.4), 1))
    cv2.imshow("input",  frame)
    
    
    
    
    
    
    
    #for processnig every so many seconds
    #if int(time.time() - start_time) > 10:
        #process_again_flag = True

    if cv2.waitKey(1) & 0xFF == ord('t'):
        
        frame_list = np.ndarray.reshape(frame, (1,784))
        
        predictions = model.predict(frame_list)
        
        
        try:
            #print(predictions)
            response = answer_key[list(predictions[0]).index(1)]
            print("You are showing a... ", response)
            #start_time = time.time()
            #process_again_flag = False
            
        except:
            print("Nothing there")
        
        
        
        
        
    
    #break out of loop with q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    
    




