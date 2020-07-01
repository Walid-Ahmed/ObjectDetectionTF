#python detect0bjectsInVideo.py


import os

import cv2
import shutil
import time
import pandas as pd
from detectObjectsInImage  import load_graph
from detectObjectsInImage  import get_objlabels

from detectObjectsInImage  import  detectOnSingleImage
import tensorflow as tf
import argparse
# run your code


from keras.models import load_model

DETECTIONS_DICT = dict()
showFrame=True
objNumber=1
frameNum=1


#videoFile="videos/crashlandingofdroneFlying.mp4"






if __name__ == '__main__':

    print("[INFO] Tensorflow  version used is {}".format(tf.__version__))

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelFile", default="models/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.pb", required=False,help="path to model file")
    ap.add_argument("--threshold", type=float, default=0.9,required=False, help="path to model file")
    ap.add_argument("--videoFilePath", type=str, default="videos/ZakiBashakha.mp4",required=False, help="path to model file")
    ap.add_argument("--labelsFile", default="cocoLabels.csv",type=str,  help="path to labels file")
    ap.add_argument("--everyNFrames", default=5,type=int,  help="path to labels file")



    if not os.path.exists('Results'):
        os.makedirs('Results')

    if not os.path.exists(os.path.join('Results',"MLFrames")):
        os.makedirs(os.path.join('Results',"MLFrames"))    


     #read the arguments
    args = vars(ap.parse_args())
    modelFile=args["modelFile"]
    threshold=args["threshold"]
    videoFilePath=args["videoFilePath"]
    labelsFile=args['labelsFile']

    everyNFrames=args["everyNFrames"]

    model=load_graph(modelFile)
    labels = get_objlabels(labelsFile)
    objects_num = dict()


    cap = cv2.VideoCapture(videoFilePath)
    numberOfProcessedFrames=0

    numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( "The video includes {0} frames".format(numberOfFrames))
    numberOfProcessedFrames=0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    print( "The video includes {0} frames".format(numberOfFrames))

    fileNameToSaveVideo='demo_ObjectDetection_'+os.path.basename(videoFilePath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    video_creator = cv2.VideoWriter(fileNameToSaveVideo,fourcc, 3, (960,720))


    f = open("det.txt", "w")


    while(cap.isOpened()):
        ret, frame = cap.read()
        if (frame is None):
            break
        if ((frameNum%everyNFrames)==0):
            #test(frame,labels,f)
            path=os.path.join("results","Frames","frame_"+(str(frameNum).zfill(6))+".jpg")
            image=detectOnSingleImage(frame,model,labels,threshold, objects_num,path=path)
            video_creator.write(cv2.resize(image,(960,720)))
            numberOfProcessedFrames=numberOfProcessedFrames+1
            print("Detecting objects  in frame number {0}".format(frameNum))
        frameNum=frameNum+1
    print( "The video includes {0} frames".format(numberOfFrames))
    print( "The number of processed   frames is {0}".format(numberOfProcessedFrames))
    print(DETECTIONS_DICT)
    video_creator.release()
    f.close()



