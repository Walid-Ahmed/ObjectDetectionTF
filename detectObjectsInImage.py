
#python  detectObjectsInImage..py  --modelFile  models/ssd_mobilenet_v1_coco_2018_01_28.pb

#python  detectObjectsInImage.py  --modelFile  models/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.pb  --threshold  0.5  --imagesFolderPath  images   --labelsFile  cocoLabels.csv
#python  detectObjectsInImage.py  --modelFile  models/faster_rcnn_nas_coco_2018_01_28.pb



import os
import cv2
import numpy as np
import tensorflow as tf
import argparse

import pandas as pd
from  util import paths




objNumber=1



def load_graph(modelFile):
    model = tf.Graph()
    with model.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(modelFile, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("[INFO' Model loaded sucessfully from {}".format(modelFile))        
    return model        

def get_objlabels(labelsFile):

    # ref https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
    csvFileName=labelsFile

    labels_df = pd.read_csv(csvFileName ,delimiter = ',' , header=None)
    labels=(labels_df[0]).tolist()
    return labels


def detect(model,image, labels,threshold_conf=0.5):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filteredBoxes = []
        height, width, channels = image.shape
        image_np_expanded = np.expand_dims(image, axis=0)

        image_tensor = model.get_tensor_by_name('image_tensor:0')
        boxes = model.get_tensor_by_name('detection_boxes:0')

        scores = model.get_tensor_by_name('detection_scores:0')
        classes = model.get_tensor_by_name('detection_classes:0')
        num_detections = model.get_tensor_by_name('num_detections:0')
        session = tf.compat.v1.Session(graph=model)
        (boxes, scores, classes, num_detections) = session.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes, scores, classes, num_detections = map(np.squeeze, [boxes, scores, classes, num_detections])
        print ("[INFO] Found {0} detections".format(int(num_detections)))
        num_detections=int(num_detections.item(0))

        for i in range(num_detections):
            classIndex = int(classes[i])
            if scores[i] < threshold_conf: 
                continue
            label = labels[classIndex]
            ymin, xmin, ymax, xmax = boxes[i]

            (left, right, top, bottom) = (xmin * width, xmax * width,ymin * height, ymax * height)
            box_info = int(left), int(top), int(right), int(bottom), label, scores[i]  # x1, y1, x2, y2
            filteredBoxes.append(box_info)

        return filteredBoxes


def detectOnFolderOfImages(model,threshold,labels,imagesFolderPath,objects_num):
   

    files=paths.list_images(imagesFolderPath)
    imagePaths = sorted(list(paths.list_images(imagesFolderPath)))

    for imagePath in imagePaths:
        print("[INFO]  Detecting objects in  image   {}".format(imagePath))
        image=loadImage(imagePath)
        detectOnSingleImage(image,model,labels,threshold,objects_num,imagePath)


def loadImage(path):
    print ("Testing on {0}".format(path))
    image = cv2.imread(path) 
    return image


def detectOnSingleImage(image,model,labels,threshold, objects_num,path=None):
 
    imageCopy=image.copy()
    file_name = os.path.basename(path)
    boxes=detect(model,image,labels, threshold)


    if boxes is None:
        boxes = []
    result_boxes = []
    for box in boxes:
        (objX1, objY1, objX2, objY2, label, conf) = box
        result = (objX1, objY1, objX2, objY2, label, conf)
        result_boxes.append(result)
        pathToSaveDetectedObjects=os.path.join("results",label)
        if not os.path.exists(pathToSaveDetectedObjects):
            os.makedirs(pathToSaveDetectedObjects)
        detectedObject=imageCopy[int(objY1):int(objY2), int(objX1):int(objX2)]
        fileNameToSaveObject=os.path.join(pathToSaveDetectedObjects,label+"_"+str(objects_num.get(label,0))+".png")
        cv2.imwrite(fileNameToSaveObject, detectedObject)
        if  abs( (int(objX1)-(int(objX2))) ) >400:
            continue
        cv2.rectangle(image, (int(objX1), int(objY1)), (int(objX2), int(objY2)), (0, 255, 0), 2)
        cv2.putText(image, str(label) + " - " + str(conf), (int(objX1) + 5, int(objY1) + 20),cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
        objects_num[label]=objects_num.get(label,0)+1
    
    fileNametoSaveOutput=os.path.join("Results","MLFrames",file_name) 
    cv2.imwrite(fileNametoSaveOutput, image)
    cv2.imshow('ML_output', image)
    cv2.waitKey(300)
    return image  #image with bounding boxes



if __name__ == '__main__':
    print("[INFO] Tensorflow  version used is {}".format(tf.__version__))

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelFile", default="models/ssd_mobilenet_v1_coco_2018_01_28.pb", required=False,help="path to model file")
    ap.add_argument("--threshold", type=float, default=0.5,required=False, help="path to model file")
    ap.add_argument("--imagesFolderPath", type=str, default="images",required=False, help="path to model file")
    ap.add_argument("--labelsFile", default="cocoLabels.csv",type=str,  help="path to labels file")


    if not os.path.exists('Results'):
        os.makedirs('Results')

    if not os.path.exists(os.path.join('Results',"MLFrames")):
        os.makedirs(os.path.join('Results',"MLFrames"))    


        #read the arguments
    args = vars(ap.parse_args())
    modelFile=args["modelFile"]
    threshold=args["threshold"]
    imagesFolderPath=args["imagesFolderPath"]
    labelsFile=args['labelsFile']

    model=load_graph(modelFile)
    labels = get_objlabels(labelsFile)
    objects_num = dict()





    detectOnFolderOfImages(model,threshold,labels,imagesFolderPath,objects_num)
    print(objects_num)

