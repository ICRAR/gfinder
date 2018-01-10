#! /usr/bin/env python3

#Compile graph with
#'mvNCCompile graphs/test-graph/test-graph.meta -in=images -on=fc_weights_1 -o=./graphs/test-graph/test-graph-compiled'

#Imports
from mvnc import mvncapi as mvnc    #For Movidius NCS API
import sys
import numpy
import cv2

#Where to find NCS compiled graph
graph_path = "./graphs"
graph_name = "test-graph"

#Set logging level
mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

#Find NCS'
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print("Error: no devices found, exiting")
    quit()
else:
    print("Operating on " + str(len(devices)) + " devices")

#TODO: more than one device
#Open the first device
device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
print("Loading graph from compiled graph file")
with open(graph_path + "/" + graph_name + "/" + graph_name + "-compiled", mode='rb') as f:
    graph_file = f.read()

#Load categories
categories = ["Y", "N"]

#Put graph onto NCS
print("Allocating graph onto NCS")
graph = device.AllocateGraph(graph_file)

print("Loading image onto NCS")
'''
#TODO get image
#Get image and download onto NCS
img =
print("Downloading input image to NCS ... ", end="")
graph.LoadTensor(img.astype(numpy.float32), "input_image")
print("X")
'''

#Get output of graph
print("Evaluating output of neural network ... ", end="")
output, userobj = graph.GetResult()
print("X")

#TODO
#Announce output of image into neural net


#Deallocate graph
graph.DeallocateGraph()

#TODO: deviceS
#Close opened device
device.CloseDevice()
