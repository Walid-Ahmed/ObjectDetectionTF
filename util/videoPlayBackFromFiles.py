


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import  pyplot  as plt
from PIL import Image
from IPython import get_ipython
import cv2







scaleFactor=0.6
width=130#1920
height=400#1080
folder="../Results/MLFrames"
scaleFactor=1
width=1280
height=720
width=int(width*scaleFactor)
height=int(height*scaleFactor)
filenamesList=os.listdir(folder)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

fileName="demoObjecttDetection.mp4"

#width=166
#height=315
fps=100
video_creator = cv2.VideoWriter(fileName,fourcc, fps, (width,height))
imgFiles=os.listdir(folder)
imgFiles.sort()
imgFiles=sorted(os.listdir(folder))



def  showVideo2():
    		frameNum=1
    		for filename in imgFiles:
    			print(filename)
    			img = cv2.imread(os.path.join(folder,filename))
    			#img = cv2.resize(img, (width, height)) 
    			if (img is None):
    				frameNum=frameNum+1
    				continue
    			img=cv2.resize(img,(width,height))
    			video_creator.write(img)
    			cv2.imshow('Demo',img)
				#if (frameNum==1):
					#raw_input("press any key to continue")
    			cv2.waitKey(1)
    			frameNum=frameNum+1
    			#out.write(img)
    			if cv2.waitKey(1) & 0xFF == ord('q'):
    				break
    	 		





#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter("Detect_HardHat.m4v",fourcc, 5, (1152,648))
showVideo2()
video_creator.release()







