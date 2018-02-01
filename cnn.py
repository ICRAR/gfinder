#! /usr/bin/env python3

#Visualise with: 'tensorboard --logdir='logs''

#Imports
import sys                          #For embedded python workaround
import os                           #For file reading and warning suppression

#Embedded python work around
if not hasattr(sys, 'argv'):
    sys.argv = ['']

import shutil                               #For directory deletion
import numpy as np                          #For transforming blocks
import matplotlib.pyplot as plt             #For visualisation
from texttable import Texttable             #For outputting
import math                                 #For logs
import time                                 #For debugging with catchup
from datetime import datetime, timedelta    #For timing inference runs
import threading                            #For parallel inferencing
from multiprocessing import Queue, Value    #For sharing data between threads
import ctypes as c                          #For sharing numpy arrays
import subprocess                           #For compiling graphs
import socket                               #For IPC
import select                               #For waiting for socket to fill
import errno                                #Handling socket errors
import struct                               #For converting kdu_uint32s to uint32s
import tensorflow as tf                     #For deep learning
from mvnc import mvncapi as mvnc            #For Movidius NCS API

#Set suppressed logging level for Movidius NCS API
mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 1)

#Set non-verbose error checking for tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  #1 filters all
                                        #2 filters warnings
                                        #3 filters none

#Global variables
#Relative paths
GRAPHS_FOLDER = "./graphs"
LOGS_FOLDER = "./logs"

#Variables for compiling evaluation graphs for NCS
#See 'new_graph' function for explanation of names here options
INPUT_NAME = "images"
OUTPUT_CPU_NAME = "dense_final/BiasAdd"
OUTPUT_NCS_NAME = "dense_final/BiasAdd"
SHAVES = 12 #Each NCS has 12 shave cores. Use them all

#Hardcoded image input dimensions
INPUT_WIDTH = 32
INPUT_HEIGHT = 32

#Globals for creating graphs
#Convolutional layer filter sizes in pixels
FILTER_SIZES    =   [5, 5, 5]
#Number of filters in each convolutional layer
NUM_FILTERS     =   [16, 24, 32]
#Number of neurons in fully connected (dense) layers. Final layer is added
#on top of this
FC_SIZES        =   [256, 48]

#Converts to frequency domain and applies a frequency cutoff on a numpy array
#representing an image. Cutoff: <1 for low freq, >200 for high freq
def apply_frequency_cutoff(img_matrix, cutoff):
    #Convert to freq domain
    img_matrix =  np.fft.fft2(img_matrix)
    img_matrix *= 255.0 / img_matrix.max() #Scale to [0, 255]

    #Apply cutoff to real part of freq (<1 for low freq, >200 for high freq)
    cutoff = np.abs(img_matrix) < cutoff
    img_matrix[cutoff] = 0

    #Back to spacial domain
    img_matrix =  np.fft.ifft2(img_matrix)
    img_matrix =  np.abs(img_matrix)        #Complex->real
    img_matrix =  np.log10(img_matrix)      #Take log scale
    img_matrix *= 255.0 / img_matrix.max()  #Scale to [0, 255]
    #img_matrix = 255 - img_matrix           #Invert

    return img_matrix

#Uniquely saves a figure of the imgage represented by the supplied array
#in the output folder
def save_array_as_fig(img_array, name, start_x=0, start_y=0):
    #Create graph to ensure that block was read correctly
    fig = plt.figure(name, figsize=(15, 15), dpi=80)  #dims*dpi = res

    #Bounds
    w = img_array.shape[0]
    h = img_array.shape[1]

    #Plot
    ax = plt.gca()

    ax.set_xticks(np.arange(0, w, math.floor(w/10)))  #Ticks
    ax.set_xticks(np.arange(0, w, 1), minor = True)
    #labels
    ax.set_xticklabels(np.arange(start_x, start_x + w, math.floor(w/10)))

    ax.set_yticks(np.arange(0, h, math.floor(h/10)))
    ax.set_yticks(np.arange(0, h, 1), minor = True)
    #labels
    ax.set_yticklabels(np.arange(start_y, start_y + h, math.floor(h/10)))

    #Colourisation mapped to [0,1]
    plt.imshow( np.uint8(img_array), cmap="Greys_r",
                vmin=0, vmax=255,
                interpolation='nearest')

    #Label and save
    fig.savefig("output/" + name)

    #Explicity close figure for memory usage
    plt.close(fig)

#Transforms the standard uint8 1 channel image gotten from the Kakadu SDK into
#a three channel RGB (redundant channels) half precision float (float16) image
#such that it is compatible with the Movidius NCS architecture
def make_compatible(image_data,
                    save_image=False,
                    width=INPUT_WIDTH,
                    height=INPUT_HEIGHT,
                    duplicate_channels=True):
    #Reshape to placeholder dimensions
    output = np.reshape(image_data, (width, height))

    #Cast to 8-bit unsigned integer
    output = np.uint8(output)

    #Output an image if required while uint8
    if save_image:
        save_array_as_fig(output, 'test')

    #Now cast
    output = np.float16(output)

    #Add two more channels to get RBG if required
    if duplicate_channels:
        output = np.dstack([output]*3)

    #Give it back
    return output

#Converts socket recieved bytes to integers
def byte_string_to_int_array(bytes):
    data = []
    i = 0
    while i < len(bytes):
        data.append(struct.unpack('I', bytes[i:(i + 4)]))
        i = i + 4
    return data

#Facilitates the saving of an area of input data for comparison with evaluation
#process
def save_data_as_comparison_image(image_data, x, w, y, h, f):
    #Make compatible
    image_data = make_compatible(   image_data, width=w, height=h,
                                    duplicate_channels=False)

    #Create name
    name = "original-" + str(x) + "-" + str(y)  + "-" + str(w) + "-" + \
            str(h) + "-" + str(f)

    #Save array as fig
    save_array_as_fig(image_data, name, start_x=x, start_y=y)

#Boots up a concurrently running client that keeps the trainable graph in memory
#to speed up training time. Updates depending on argument at the end of its run
def run_training_client(graph_name, port, optimise_and_save, batch_size, total_units):
    #Connect to specified port
    print("Connecting client to port " + str(port))
    sock = socket.socket()
    sock.connect(('', port))
    sock.setblocking(0) #Throw an exception when out of data to read (non-blocking)
    timeout = 0.5       #Timeout before cutting recv'ing loop in seconds

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating,
    #by creating an identical grah and restoring weights
    print(  "Loading graph '" + graph_name + "' for " + \
            ("training" if optimise_and_save == 1 else "validation"))
    saver = new_training_graph(graph_name)
    restore_model(graph_name, sess, saver)

    #Prepare to track information
    t_pos = 0   #Correct classified a galaxy
    f_pos = 0   #Guessed galaxy when should have been noise
    t_neg = 0   #Correctly classified noise
    f_neg = 0   #Guessed noise when should have been galaxy

    #Recieve supervised data. Take 4 times as many as recieving uint32s
    #(kdu_uint32s) and convert
    recieving = True
    image_batch = []
    label_batch = []
    batch_num = 0
    units_num = 0
    while recieving:
        #When ready then recv
        ready = select.select([sock], [], [], timeout)
        if ready[0]:
            #If INPUT_WIDTH*INPUT_HEIGHT*4 units are recieved then this is an image
            image_bytes = sock.recv((INPUT_WIDTH*INPUT_HEIGHT)*4)
            image_input = byte_string_to_int_array(image_bytes)
            if len(image_input) == INPUT_WIDTH*INPUT_HEIGHT:
                #Make the image tensorflow graph compatible
                image_batch.append(make_compatible(image_input, False))
            else:
                #No data found in socket, stop recving
                print("Error: image data not recv'd correctly, finishing early")
                break
        else:
            if len(image_batch) != 0:
                print("No data found in socket, remaining batch had " + str(len(image_batch)) + " units:")
            recieving = False

        #When ready then recv
        if recieving:   #Check if still reciving (image could've finished)
            ready = select.select([sock], [], [], timeout)
            if ready[0]:
                #If 4 units are recieved then this is a label
                label_bytes = sock.recv(1*4)
                label_input = byte_string_to_int_array(label_bytes)
                if len(label_input) == 1:
                    #Make the label tensorflow compatible
                    label_one_hot = [0, 0]
                    #Label input is a tuple in a list, hence this monstrosity
                    label_one_hot[label_input[0][0]] = 1
                    label_batch.append(label_one_hot)
                else:
                    #No data found in socket, stop recving
                    print("Error: label data not recv'd correctly, finishing early")
                    break
            else:
                print("No label data found in socket, ending")
                recieving = False

        #Increment if a label and image came trhough (one unit)
        if recieving:
            units_num = units_num + 1

        #If at end of data feed or if the batch is full then feed to graph
        batch_ready =   len(image_batch) == batch_size and \
                        len(label_batch) == batch_size;
        if batch_ready or (not recieving and len(image_batch) != 0):
            #Turn recieved info into a feeder dictionary
            feed_dict = {
                            'images:0': image_batch,
                            'labels:0': label_batch,
                            'is_training:0': optimise_and_save
                        }

            #If training then train
            if optimise_and_save == 1:
                #Feed into the network so it can 'learn' by running the adam optimiser
                sess.run('Adam', feed_dict=feed_dict)

            #What is the prediction for each image? (as prob [0,1])
            preds = sess.run('predictor:0', feed_dict=feed_dict)

            #Redundant for clarity here
            batch_t_pos, batch_t_neg, batch_f_pos, batch_f_neg = 0, 0, 0, 0
            for i in range(len(image_batch)):
                pred_gal = preds[i][1] > 0.5    #Softmax over classes [NSE, GAL]
                is_gal = label_batch[i][1] == 1 #One hot encoding

                #Count types of failure
                if pred_gal and is_gal:
                    batch_t_pos += 1
                elif pred_gal and not is_gal:
                    batch_f_pos += 1
                elif not pred_gal and is_gal:
                    batch_f_neg += 1
                elif not pred_gal and not is_gal:
                    batch_t_neg += 1

            #Increment running totals
            t_pos += batch_t_pos
            f_pos += batch_f_pos
            f_neg += batch_f_neg
            t_neg += batch_t_neg
            batch_num += 1

            #Print running results
            print("-units           = " + str(units_num) + "/" + str(total_units) + " (" + "{0:.4f}".format(100*units_num/total_units) + "% of units fed)")
            #Prints the global step
            print("-global_step     = " + str(sess.run('global_step:0', feed_dict=feed_dict)))
            #Prints current learning rate
            print("-alpha           = " + str(sess.run('alpha:0', feed_dict=feed_dict)))
            #Prints current loss
            print("-loss            = " + str(sess.run('loss:0', feed_dict=feed_dict)))

            #Print tabulated gal data
            tableGal = Texttable()
            tableGal.set_precision(4)
            tableGal.set_cols_width([11, 7, 7, 7, 7])
            tableGal.set_cols_align(['r', 'r', 'r', 'r', 'r'])
            tableGal.add_rows([
                ['GAL_PREDs', 'T', 'F', 'T+F', '%'],

                ['BATCH_' + str(batch_num),
                batch_t_pos,
                batch_f_pos,
                batch_t_pos + batch_f_pos,
                "{0:.4f}".format(100*batch_t_pos/(batch_t_pos + batch_f_pos)) \
                    if (batch_t_pos + batch_f_pos != 0) else "-"],

                ['SESSION',
                t_pos,
                f_pos,
                t_pos + f_pos,
                "{0:.4f}".format(100*t_pos/(t_pos + f_pos)) \
                    if (t_pos + f_pos != 0) else "-"]
            ])
            print(tableGal.draw())

            #Print tabulated nse data
            tableNse = Texttable()
            tableNse.set_precision(4)
            tableNse.set_cols_width([11, 7, 7, 7, 7])
            tableNse.set_cols_align(['r', 'r', 'r', 'r', 'r'])
            tableNse.add_rows([
                ['NSE_PREDs', 'T', 'F', 'T+F', '%'],

                ['BATCH_' + str(batch_num),
                batch_t_neg,
                batch_f_neg,
                batch_t_neg + batch_f_neg,
                "{0:.4f}".format(100*batch_t_neg/(batch_t_neg + batch_f_neg)) \
                    if (batch_t_neg + batch_f_neg != 0) else "-"],

                ['SESSION',
                t_neg,
                f_neg,
                t_neg + f_neg,
                "{0:.4f}".format(100*t_neg/(t_neg + f_neg)) \
                    if (t_neg + f_neg != 0) else "-"]
            ])
            print(tableNse.draw())

            #Print tabulated summary
            tableSum = Texttable()
            tableSum.set_precision(4)
            tableSum.set_cols_width([11, 7, 7, 7, 7])
            tableSum.set_cols_align(['r', 'r', 'r', 'r', 'r'])
            tableSum.add_rows([
                ['ALL_PREDs', 'T', 'F', 'T+F', '%'],

                ['BATCH_' + str(batch_num),
                batch_t_pos + batch_t_neg,
                batch_f_pos + batch_f_neg,
                batch_t_neg + batch_f_neg + batch_t_pos + batch_f_pos,
                "{0:.4f}".format(100*(batch_t_pos + batch_t_neg)/len(image_batch)) ],

                ['SESSION',
                t_pos + t_neg,
                f_pos + f_neg,
                t_neg + f_neg + t_pos + f_pos,
                "{0:.4f}".format(100*(t_pos + t_neg)/units_num) ]
            ])
            print(tableSum.draw())
            print("")


            #Ready for new batch
            image_batch = []
            label_batch = []

    #Only optimise & save the graph if in training mode
    if optimise_and_save == 1:
        #Save the slightly more trained graph if in training mode
        print("Saving training modifications made to graph in this run")
        save_model(graph_name, sess, saver)

    #Close tensorflow session
    sess.close()

#Softmax is considered a large typed operation on the ncs, so a version
#has been implemented here
def softmax(arr):
    """Compute softmax values for each sets of scores in x."""
    e_arr = np.exp(arr - np.max(arr))
    return e_arr / e_arr.sum()

#Helper to process a probability map to highlight regions that most likely
#have galaxies in them
def post_process_prob_map(prob_map):
    #Get dimensions
    width   = prob_map.shape[0]
    height  = prob_map.shape[1]
    depth   = prob_map.shape[2]

    #Ignore poorly sampled edges
    ignore_offset = 8
    prob_map[0:ignore_offset,:,:] = 0
    prob_map[(width-ignore_offset):(width),:,:] = 0
    prob_map[:,0:ignore_offset,:] = 0
    prob_map[:,(height-ignore_offset):(height),:] = 0

    #Apply threshold for clarity
    threshold = prob_map < 0.4
    #prob_map[threshold] = 0

    #Go through entire prob map and zero out high prediction pixels
    #that don't have a sufficient number of nearby high prediction pixels

    #Cluster (with min radius) pixel groups

    #Write bounding boxes for pixel groups

    #Increase prob for high prob areas that bleed through freq frames
    if depth >= 3:
        #Obviously cannot do on frequencu frames on edge (no adjacent)
        for f in range(1, depth-1):
            pass

    #Combat coordinate system change by flipping and rotating
    prob_map = np.rot90(prob_map, axes=[0,1])
    prob_map = np.flip(prob_map, axis=0)    #Constant time (uses views)

    #Return the altered map
    return prob_map

#Helper to plot an evaluation probability map
def plot_prob_map(prob_map, start_x, start_y, start_f):
    #Save the probability map to output in 2d componetn slices
    for f in range(prob_map.shape[2]):
        #dims*dpi = res
        fig = plt.figure("component-" + str(f), figsize=(15, 15), dpi=80)

        #Bounds
        w = prob_map.shape[0]
        h = prob_map.shape[1]

        #Plot
        ax = plt.gca()
        ax.set_aspect('auto', adjustable='box')    #Aspect ratio

        ax.set_xticks(np.arange(0, w, math.floor(w/10)))  #Ticks
        ax.set_xticks(np.arange(0, w, 1), minor = True)
        #labels
        ax.set_xticklabels(np.arange(start_x, start_x + w, math.floor(w/10)))

        ax.set_yticks(np.arange(0, h, math.floor(h/10)))
        ax.set_yticks(np.arange(0, h, 1), minor = True)
        #labels
        ax.set_yticklabels(np.arange(start_y, start_y + h, math.floor(h/10)))

        #Colourisation mapped to [0,1], as this is a probability map
        plt.imshow( prob_map[:,:,f], cmap="Greys_r", vmin=0.0, vmax=1.0,
                    interpolation='nearest')

        #Label and save
        name = "probmap-" + str(start_x) + "-" + str(start_y) + "-" + str(w) + \
                "-" + str(h) + "-" + str(start_f + f)
        fig.savefig("output/" + name)

        #Explicity close figure for memory usage
        plt.close(fig)

#Recieves a unit and evaluates it using the graph. This function is used for
#development (comparisons) and not designed for use in production. Expect
#unreliability
def run_evaluation_client_for_cpu(  graph_name,        #Graph to evaluate on
                                    port,              #Port to stream from
                                    region_width,      #Width of region to evaluate
                                    region_height,     #Height of region to evaluate
                                    region_depth,      #Depth of region to evaluate
                                    units_per_component,   #How many images at each frequency
                                    start_x,
                                    start_y,
                                    start_f):
    sock = socket.socket()
    sock.connect(('', port))
    sock.setblocking(0) #Throw an exception when out of data to read (non-blocking)
    timeout = 0.5       #How many seconds to wait before finishing

    #A session is required
    sess = tf.Session()

    #Initialise no placeholder architecture and restore weights into empty
    #graph
    saver = new_evaluation_graph(graph_name)
    restore_model(graph_name, sess, saver)

    #Print a picture to put on the tensorboard fridge
    writer = tf.summary.FileWriter('logs', sess.graph)
    writer.close()

    #Data will need to be stored in a heat map - allocate memory for this
    #data structure. Probability aggregate is the sum of each predictions made
    #that include that pixel. Samples are the number of predictions made for
    #that pixel. This allows normalised probability map as output
    prob_ag_map = np.zeros(
        shape=(region_width, region_height, region_depth),
        dtype=float
    )
    sample_map = np.zeros(
        shape=(region_width, region_height, region_depth),
        dtype=float #For later convienience
    )

    #For tracking time and progress
    evaluation_start = datetime.now()
    count = 0
    expected = units_per_component*region_depth #Images per freq frame*freq frame for total expected images

    #Load image data from socket while there is image data to load
    recieving = True
    while recieving:
        #When ready then recv location of image in evaluation space
        image_loc = None
        ready = select.select([sock], [], [], timeout)
        if ready[0]:
            #If 3*4 units are recieved then this is an image's location
            image_loc_bytes = sock.recv(3*4)
            image_loc = byte_string_to_int_array(image_loc_bytes)
            if len(image_loc) != 3 or not image_loc:
                #No data found in socket, stop recving
                print("Error: image location data not recv'd correctly, finishing early")
                break

        #When ready then recv image data
        image_input = None
        ready = select.select([sock], [], [], timeout)
        if ready[0]:
            #If INPUT_WIDTH*INPUT_HEIGHT*4 units are recieved then this is an image
            image_bytes = sock.recv((INPUT_WIDTH*INPUT_HEIGHT)*4)
            image_input = byte_string_to_int_array(image_bytes)
            if len(image_input) != INPUT_WIDTH*INPUT_HEIGHT or not image_input:
                #No data found in socket, stop recving
                print("Error: image data not recv'd correctly, finishing early")
                break
        else:
            #Announce finish and time taken (remove timeout)
            evaluation_duration = datetime.now() - evaluation_start - timedelta(seconds=timeout)
            print("\r100.000% complete (" + str(evaluation_duration) + ")")
            print("No image data found in socket. Ending recv'ing loop")
            recieving = False

        #If something was recieved from socket
        if recieving:
            #Make the recv'd data graph compatible
            image_input = make_compatible(image_input, False)
            image_tlx = image_loc[0][0]
            image_brx = image_tlx + INPUT_WIDTH
            image_tly = image_loc[1][0]
            image_bry = image_tly + INPUT_HEIGHT
            image_f   = image_loc[2][0]

            #Create a unitary feeder dictionary
            feed_dict_eval = {
                                'images:0': [image_input]
                            }

            #Get the prediction that it is a galaxy
            #(the 1th element of the one hot encoding)
            output = sess.run(OUTPUT_CPU_NAME + ':0', feed_dict=feed_dict_eval)[0]
            pred = softmax(output)[1]
            count += 1
            print("\r{0:.4f}".format(100*count/expected) + "% complete", end="")

            #Write information into heatmap. Likelihood is simply added onto
            #heat map at each pixel
            prob_ag_map[image_tlx:image_brx,
                        image_tly:image_bry,
                        image_f] += pred
            sample_map[image_tlx:image_brx,
                        image_tly:image_bry,
                        image_f] += 1.0

    #Announce progress
    print("Normalising prediction map")

    #Normalise the probability by dividing the aggregate prob by the amount
    #of predictions/samples made at that pixel. Handle divide by zeroes which may
    #occur along edges
    prob_map = np.divide(   prob_ag_map, sample_map,
                            out=np.zeros_like(prob_ag_map),
                            where=sample_map!=0)

    #Process
    prob_map = post_process_prob_map(prob_map)

    #Plot data or visualisation
    print("Plotting 2D component prediction map(s)")
    plot_prob_map(prob_map, start_x, start_y, start_f)

    #Close tensorflow session, no reason to save
    sess.close()

#Must occur pre NCS usage, creates a version of the .meta file without any
#training placeholders (otherwise will fail)
def compile_for_ncs(graph_name):
    #A session is required
    sess = tf.Session()

    #Initialise no placeholder architecture and restore weights into empty
    #graph
    saver = new_evaluation_graph(graph_name)
    restore_model(graph_name, sess, saver)

    #Save without placeholders
    saver.save(sess, GRAPHS_FOLDER + "/" + graph_name + "/" + "ncs")

    #Finish session
    sess.close()

    #Compile graph with subprocess for NCS
    #Example: 'mvNCCompile graphs/test-graph/test-graph.meta -in=images -on=dense_final/BiasAdd -o=./graphs/test-graph/test-graph-for-ncs.graph'
    subprocess.call([
        'mvNCCompile',
        (GRAPHS_FOLDER + "/" + graph_name + "/" + "ncs.meta"),
        "-s=" + str(SHAVES),
        "-in=" + INPUT_NAME,
        "-on=" + OUTPUT_NCS_NAME,
        "-o=" + GRAPHS_FOLDER + "/" + graph_name + "/" + "ncs.graph",
        "-is", str(INPUT_WIDTH), str(INPUT_HEIGHT)
    ]
    #,stdout=open(os.devnull, 'wb')  #Suppress output
    );

#Manages te inferences on one NCS
def run_NCS_parallel(   device_number, graph_handle, queue, utilisation,
                        num_inferenced, num_expected,
                        prob_ag_map, sample_map):

    #Has processing started
    started = False

    #Iterate over data
    while num_inferenced.value != num_expected:
        #Get data from shared queue
        try:
            #Get data while blocking briefly
            data = queue.get(True, 1)
        except:
            #Check if queue is empty, as NCS' are unreliable and some units
            #may not have been processed
            if started and queue.empty:
                break

            #Otherwise its likely that another thread beat this thread to last
            #datum in the queue, continue into finish
            continue

        #Track the number of images processed by this device
        utilisation.value += 1

        #Once one object is found in the queue processing has started
        if not started:
            started = True

        #Attempt to load data
        try:
            #Load into tensor
            graph_handle.LoadTensor(data[0], "")
        except:
            pass

        #Attempt to inference data
        try:
            #Get inference
            output, image_loc_string = graph_handle.GetResult()

            #Get location of the processed area through the userobj string
            loc = data[1]

            #One hot encoding, want probability of class 1
            #(galaxy). Note ncsdk doesn't support the predictor
            #layer which softmaxes the last dense layer to give
            #class probability, so manual
            #softmax must be done in its place
            pred = softmax(output)[1]

            #Write information into heatmap. Likelihood is
            #simply added onto heat map at each pixel
            prob_ag_map[loc[0]:loc[1],
                        loc[2]:loc[3],
                        loc[4]] += pred
            sample_map[ loc[0]:loc[1],
                        loc[2]:loc[3],
                        loc[4]] += 1.0

        except:
            pass

        #Increment number inferenced
        num_inferenced.value += 1

        #Progress report
        print(  "\r{0:.4f}".format(100*(num_inferenced.value/num_expected)) \
                + "% complete", end="")

#Boots up one NCS, loads a compiled version of the graph onto it and begins
#running inferences on it. Supports inferencing a 3d area that must be supplied
def run_evaluation_client_for_ncs(  graph_name,        #Graph to evaluate on
                                    port,              #Port to stream from
                                    region_width,      #Width of region to evaluate
                                    region_height,     #Height of region to evaluate
                                    region_depth,      #Depth of region to evaluate
                                    units_per_component,  #Needed to calculate expected units
                                    start_x,
                                    start_y,
                                    start_f):
    #Ensure there is at least one NCS
    device_name_list = mvnc.EnumerateDevices()
    num_devices = len(device_name_list)
    if num_devices == 0:
        print("No devices found, exiting")
        sys.exit()

    #Compile a copy of the evaluation graph for running on the ncs
    compile_for_ncs(graph_name)

    #Load the compiled graph
    graph_file = None;
    filepath = GRAPHS_FOLDER + "/" + graph_name + "/" + "ncs.graph"
    with open(filepath, mode='rb') as f:
        #Read it in
        graph_file = f.read()
    if graph_file == None:
        print("Error: evaluation graph compiled for ncs could not be read, exiting")
        sys.exit()

    #Data will need to be stored in a heat map - allocate memory for this
    #data structure. Wrap the raw IPC shared memory in a numpy array for ease
    #of use
    prob_ag_map = np.zeros(
        shape=(region_width, region_height, region_depth),
        dtype=float
    )
    sample_map = np.zeros(
        shape=(region_width, region_height, region_depth),
        dtype=float #For later convienience
    )

    #Images per freq frame*freq frame for total expected images
    num_expected   = units_per_component*region_depth
    num_received   = 0
    num_inferenced = Value('i', 0)

    #Open each device and allocate it a graph
    print(  str(num_devices) + \
            " device(s) found, initialising them for inferencing task")
    device_handles = []
    graph_handles = []
    for d in range(num_devices):
        #Get device
        device_handles.append(mvnc.Device(device_name_list[d]))

        #Open device
        device_handles[d].OpenDevice()

        #Print device's optimisations
        opt_list = device_handles[d].GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
        #print(opt_list)

        #Allocate the compiled graph onto the device and get a reference
        #for later deallocation
        graph_handles.append(device_handles[d].AllocateGraph(graph_file))

        #Set iterations as 1 and confirm with print
        graph_handles[d].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
        it = graph_handles[d].GetGraphOption(mvnc.GraphOption.ITERATIONS)
        #print(it)

        #Set graph options so that calls to LoadTensor() block until complete
        graph_handles[d].SetGraphOption(mvnc.GraphOption.DONT_BLOCK, 0)

    #Set up each device's thread
    #Prepare to run each NCS in a parallel process with enqueued data
    threads = []
    queue = Queue()
    utilisation = []    #For tracking each device's utilisation
    for d in range(num_devices):
        #Count the number of units processed by each device
        utilisation.append(Value('i', 0))

        #Create the process
        t = threading.Thread(                                          \
            target=run_NCS_parallel,                                   \
            args=(( d, graph_handles[d], queue, utilisation[d],        \
                    num_inferenced, num_expected,                      \
                    prob_ag_map, sample_map, )))
        threads.append(t)

    #Start all processes
    for t in range(len(threads)):
        threads[t].start()

    #Begin recv'ing data and piping it off to children
    #Child, connect to socket
    print("Connecting NCS manager to port " + str(port))
    sock = socket.socket()
    sock.connect(('', port))
    sock.setblocking(0) #Throw an exception when out of data to read (non-blocking)
    timeout = 0.5       #How many seconds to wait before timing out recv

    #Begin inference timer and intialise other metrics
    receiving_start     = datetime.now()
    NCS_start           = datetime.now()

    #Begin loading data from C++ server into parallelised NCS'
    while(num_received != num_expected):
        #Read an image and its location from the socket
        #When ready then recv location of image in evaluation space
        image_loc = None
        ready = select.select([sock], [], [], timeout)
        if ready[0]:
            #If 3*4 units are recieved then this is an image's location
            image_loc_bytes = sock.recv(3*4)
            image_loc = byte_string_to_int_array(image_loc_bytes)

        #When ready then recv
        image_input = None
        ready = select.select([sock], [], [], timeout)
        if ready[0]:
            #If INPUT_WIDTH*INPUT_HEIGHT*4 units are recieved then this is an image
            image_bytes = sock.recv((INPUT_WIDTH*INPUT_HEIGHT)*4)
            image_input = byte_string_to_int_array(image_bytes)

        #Error check
        if image_loc == None or image_input == None:
            print("Error: image or location not recv'd correctly, exiting")
            sys.exit()

        #Track receptions
        num_received += 1

        #Make the recv'd data graph compatible
        image_input = make_compatible(image_input, False)
        #Explicitly make into '1' batch tensor, NCS won't take it as a list
        image_input = np.reshape(image_input, (1, INPUT_WIDTH, INPUT_HEIGHT, 3))
        image_tlx   = image_loc[0][0]
        image_brx   = image_tlx + INPUT_WIDTH
        image_tly   = image_loc[1][0]
        image_bry   = image_tly + INPUT_HEIGHT
        image_f     = image_loc[2][0]
        image_loc = [image_tlx, image_brx, image_tly, image_bry, image_f]

        #Place into queue
        queue.put([image_input, image_loc])

    #All images recieved from C++ server
    receiving_duration = datetime.now() - receiving_start
    print(  "\nAll image data recieved, waiting for " + str(queue.qsize()) + \
            " images to be inferenced")

    #Wait for all children to terminate
    for t in range(len(threads)):
        threads[t].join()

    #Inferencing now complete, calculate metrics for later reporting
    NCS_duration =  datetime.now() - NCS_start - \
                    timedelta(seconds=timeout)

    #Deallocate graphs and close devices
    print("\nDeallocating network graphs and closing device(s)")
    for d in range(num_devices):
        #Finished recieving, deallocate the graph from the device
        graph_handles[d].DeallocateGraph()

        #Close opened device
        device_handles[d].CloseDevice()

    #Print metrics
    print("Time spent (in parallel):")
    print("\t-receiving data: " + str(receiving_duration))
    print("\t-inferencing:    " + str(NCS_duration))
    print("Device utilisation:")
    for d in range(num_devices):
        print("\t-" + device_name_list[d] + " - " + str(utilisation[d].value))

    #Normalise the probability by dividing the aggregate prob by the amount
    #of predictions/samples made at that pixel. Handle divide by zeroes which may
    #occur along edges
    print("Processing and plotting probability map")
    prob_map = np.divide(   prob_ag_map, sample_map,
                            out=np.zeros_like(prob_ag_map),
                            where=sample_map!=0) #Ignore poorly sampled edges

    #Run post processing to enhance galactic probabilites
    prob_map = post_process_prob_map(prob_map)

    #Plot data or visualisation
    plot_prob_map(prob_map, start_x, start_y, start_f)

#Plots the convolutional filter-weights/kernel for a given layer using matplotlib
def plot_conv_weights(graph_name, scope, start_conv, final_conv, suffix):
    #A session is required
    sess = tf.Session()

    #Initialise no placeholder architecture and restore weights into empty
    #graph
    saver = new_evaluation_graph(graph_name)
    restore_model(graph_name, sess, saver)

    #Get the weights
    for j in range(start_conv, final_conv + 1):
        w = None
        for v in tf.all_variables():
            if v.name == scope + str(j) + '/kernel:0':
                w = sess.run(v)

        #Get the lowest and highest values for the weights to scale colour intensity
        #across filters
        w_min = np.min(w)
        w_max = np.max(w)

        #Number of filters used in the conv. layer
        num_filters = w.shape[3]

        #Create figure with a grid of sub-plots (4xY)
        fig, axes = plt.subplots(4, math.ceil(num_filters/4))

        #Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            #Only plot the valid filter-weights.
            if i < num_filters:
                #See new_conv_layer() for details on the format
                #of this 4-dim tensor
                img = w[:, :, 0, i] #Get first depth

                #Plot image
                ax.imshow(img, vmin=w_min, vmax=w_max,
                          interpolation='nearest', cmap='bone')

            #Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        fig.savefig("output/" + scope + str(j) + "_kernel_" + str(suffix))

    #Explicity close figure for memory usage
    plt.close(fig)

    #Close tensorflow session (no need to save)
    sess.close()

#Initialises a fresh graph and stores it for later training
def new_graph(id,             #Unique identifier for saving the graph
              filter_sizes,   #Filter dims for each convolutional layer (kernals)
              num_filters,    #Number of filters for each convolutional layer
              fc_sizes,       #Number of neurons in fully connected layers
              training_graph):  #The Movidius NCS' are picky and won't resolve unknown
                              #placeholders. For loading onto the NCS these, and other
                              #training structures (optimizer, dropout, batch normalisation)
                              #all must go - only keep the inference structures

    #Create computational graph to represent the neural net:
    print("Creating new graph: '" + str(id) + "'")
    print("\t*Structure details:")

    #Movidius NCS requires an RGB 'colour' image, even though our inputs are
    #grayscale
    channels = 3

    #Placeholders serve as variable input to the graph (can be changed when run)
    #Following placeholder takes 30x30 grayscale images as tensors
    #(must be float32 for convolution and must be 4D for tensorflow)
    images = tf.placeholder(tf.float32, shape=[None, INPUT_WIDTH, INPUT_HEIGHT, channels], name='images')
    print("\t\t" + '{:20s}'.format("-Image placeholder ") + " : " + str(images))

    if training_graph:
        #and supervisory signals which are boolean (is or is not a galaxy)
        labels = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
        print("\t\t" + '{:20s}'.format("-Label placeholder ") + " : " + str(labels))

        #Controls batch normalisation behaviour when training/validating
        is_training = tf.placeholder(tf.bool, name='is_training')

    #This layer will be transformed along the way
    layer = images

    #Create as many convolution layers as required if valid
    if len(num_filters) != len(filter_sizes):
        print("Error: " + str(len(num_filters)) + " filters requested but only " + str(len(filter_sizes)) + " sizes given")
        return

    for i in range(len(num_filters)):
        #These convolutional layers sequentially take inputs and create/apply filters
        #to these inputs to create outputs. They create weights and biases that will
        #be optimised during graph execution. They also down-sample (pool) the image
        #after doing so. Filters are created in accordance with the arguments to this
        #function
        layer = tf.layers.conv2d(
            inputs=layer,
            filters=num_filters[i],
            kernel_size=filter_sizes[i],
            strides=1,
            padding='SAME',
            use_bias=True,
            bias_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.0005),
            activation=tf.nn.relu,
            trainable=True,
            name="conv_" + str(i)
        )

        print("\t\t" + '{:20s}'.format("-Convolutional ") + str(i) + ": " + str(layer))

        '''
        #Batch norm must be implemented differently for NCS (always not training)
        if training_graph:
            layer = tf.layers.batch_normalization(
                inputs=layer,
                training=is_training,
                momentum=0.9,
                fused=True,
                name="conv_bn_" + str(i)
            )
        else:
            layer = tf.layers.batch_normalization(
                inputs=layer,
                training=False,
                momentum=0.9,
                fused=True,
                name="conv_bn_" + str(i)
            )

        print("\t\t" + '{:20s}'.format("-Batch normal ") + str(i) + ": " + str(layer))
        '''

    #Fully connected layers only take 1D tensors so above output must be
    #flattened from 4D to 1D
    num_features = (layer.get_shape())[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features], name="flattener")
    print("\t\t" + '{:20s}'.format("-Flattener ")  + " : " + str(layer))

    for i in range(len(fc_sizes)):
        #These fully connected layers create new weights and biases and matrix
        #multiply the weights with the inputs, then adding the biases. They then
        #apply a ReLU function before returning the layer. These weights and biases
        #are learned during execution
        layer = tf.layers.dense(
            inputs=layer,    #Will be auto flattened
            units=fc_sizes[i],
            activation=tf.nn.relu,
            use_bias=True,
            bias_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.0005),
            trainable=True,
            name="dense_" + str(i)
        )
        print("\t\t" + '{:20s}'.format("-Dense ") + str(i) + ": " + str(layer))

        #Batch norm must be implemented differently for NCS (always not training)
        if training_graph:
            layer = tf.layers.batch_normalization(
                inputs=layer,
                training=is_training,
                momentum=0.9,
                fused=True,
                name="dense_bn_" + str(i)
            )
        else:
            layer = tf.layers.batch_normalization(
                inputs=layer,
                training=False,
                momentum=0.9,
                fused=True,
                name="dense_bn_" + str(i)
            )

        print("\t\t" + '{:20s}'.format("-Batch normal  ") + str(i) + ": " + str(layer))

    #Dropout 50% for max regularization (equal prob dist for subnets)
    if training_graph:
        layer = tf.nn.dropout(layer, 0.5, name="dropout")
        print("\t\t" + '{:20s}'.format("-Dropout ") + " : " + str(layer))

    #The final layer is a neuron for each class
    layer = tf.layers.dense(
        inputs=layer,
        units=2,
        activation=None,    #Note no ReLU
        use_bias=True,
        bias_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.0005),
        trainable=True,
        name="dense_final"
    )
    print("\t\t" + '{:20s}'.format("-Dense (final)") + " : " + str(layer))

    #Create special operations for batch normalisation (moving mean & variance)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    #The following structures are only required when training
    if training_graph:
        #Final fully connected layer suggests prediction (these structures are added to
        #collections for ease of access later on). This gets the most likely prediction.
        #Note that the tf.nn.softmax layer is considered to operate only on large dtypes
        #by the ncsdk and may be less performant
        print("\t*Prediction details:")
        prediction = tf.nn.softmax(layer, name='predictor')
        print("\t\t" + '{:20s}'.format("-Predictor") + " : " + str(prediction))

        #Backpropogation details only required when training
        print("\t*Backpropagation details:")

        #Cross entropy (+ve and approaches zero as the model output
        #approaches the desired output
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=layer,
            labels=labels,
            name='sigmoid_cross_entropy'
        )
        print("\t\t" + '{:20s}'.format("-Cross entropy") + " : " + str(cross_entropy))

        #Loss function is the mean of the cross entropy
        loss = tf.reduce_mean(cross_entropy, name='loss')
        print("\t\t" + '{:20s}'.format("-Loss function") + " : " + str(loss))

        #Decaying learning rate for bolder retuning
        #at the beginning of the training run and more finessed tuning at end
        global_step = tf.Variable(0, trainable=False, name="global_step")   #Incremented per batch
        init_alpha = 0.001
        decay_base = 1      #alpha = alpha*decay_base^(global_step/decay_steps)
        decay_steps = 64
        alpha = tf.train.exponential_decay( init_alpha,
                                            global_step, decay_steps, decay_base,
                                            name='alpha')
        print("\t\t" + '{:20s}'.format("-Learning rate") + " : " + str(alpha), end="")
        print(" [" + str(init_alpha) + "*" + str(decay_base) + "^(batch_no/" + str(decay_steps) + ")]")

        #Optimisation function to Optimise cross entropy will be Adam optimizer
        #(advanced gradient descent)
        with tf.control_dependencies(update_ops):
            optimiser = (
                tf.train.AdamOptimizer(learning_rate=init_alpha)
                .minimize(loss, global_step=global_step)    #Decay learning rate
                                                            #by incrementing global step
                                                            #once per batch
            )
        print("\t\t" + '{:20s}'.format("-Optimiser") + " : " + str(optimiser.name))

    #Create a saver for this graph
    return tf.train.Saver(tf.global_variables())

#Wraps the above to make a basic convolutional neural network for binary
#image classification
def new_training_graph(id):
    #Create a graph, if graph is any larger then network will
    #not be Movidius NCS compatible (reason unknown)
    saver = new_graph(id,      #Id/name
              filter_sizes=FILTER_SIZES,    #Convolutional layer filter sizes in pixels
              num_filters=NUM_FILTERS,      #Number of filters in each Convolutional layer
              fc_sizes=FC_SIZES,            #Number of neurons in fully connected layer
              training_graph=True)

    #Return the saver for saving
    return saver

#Wraps the above to make a convolutional neural network for binary image
#classification WITHOUT training graph structures
def new_evaluation_graph(id):
    #Create a graph, if graph is any larger then network will
    #not be Movidius NCS compatible (reason unknown)
    saver = new_graph(id,      #Id/name
              filter_sizes=FILTER_SIZES,    #Convolutional layer filter sizes in pixels
              num_filters=NUM_FILTERS,      #Number of filters in each Convolutional layer
              fc_sizes=FC_SIZES,            #Number of neurons in fully connected layer
              training_graph=False)

    #Return the saver for saving
    return saver

#Deletes a graph and its subfolders
def delete_graph(id):
    #Remove old if it exists (don't complain if it doesn't)
    shutil.rmtree(GRAPHS_FOLDER + "/" + id, ignore_errors=True)

#Reloads the variables from another graph into the current session's graph
def restore_model(id, sess, saver):
    #File is named id in id folder
    filepath = GRAPHS_FOLDER + "/" + id + "/" + "ckpt"

    #Restoring process. '.meta' file extension is implied
    saver.restore(sess, filepath)

#Saves the current model (graph and variables) to a supplied filepath
def save_model(id, sess, saver):
    #Remove old version
    delete_graph(id)

    #File is named id in id folder
    filepath = GRAPHS_FOLDER + "/" + id + "/" + "ckpt"

    #Saving operation
    saver.save(sess, filepath)
