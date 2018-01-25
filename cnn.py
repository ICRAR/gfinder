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
import subprocess                           #For compiling graphs
import multiprocessing as mp                #For NCS multiprocessing
import ctypes                               #For initialising shared memory arrays
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

#Names of inputs and outputs in tensorflow graph (for compiling for NCS)
#See 'new_graph' function for options
INPUT_NAME = "images"
OUTPUT_CPU_NAME = "dense_final/BiasAdd"
OUTPUT_NCS_NAME = "dense_final/BiasAdd"

#Hardcoded image input dimensions
INPUT_WIDTH = 32
INPUT_HEIGHT = 32

#Globals for creating graphs
#Convolutional layer filter sizes in pixels
FILTER_SIZES    =   [5, 5, 5, 5]
#Number of filters in each convolutional layer
NUM_FILTERS     =   [8, 12, 16, 20]
#Number of neurons in fully connected (dense) layers. Final layer is added
#on top of this
FC_SIZES        =   [2048, 256, 32]

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
def save_array_as_fig(img_array, name):
    #Create graph to ensure that block was read correctly
    fig = plt.figure(name, figsize=(15, 15), dpi=80)  #dims*dpi = res

    #Bounds
    w = img_array.shape[0]
    h = img_array.shape[1]

    #Plot
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')    #Aspect ratio

    ax.set_xticks(np.arange(0, w, math.floor(w/10)))  #Ticks
    ax.set_xticks(np.arange(0, w, 1), minor = True)

    ax.set_yticks(np.arange(0, h, math.floor(h/10)))
    ax.set_yticks(np.arange(0, h, 1), minor = True)

    #Colourisation mapped to [0,255]
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
def make_compatible(image_data, save_image):
    #Reshape to placeholder dimensions
    output = np.reshape(image_data, (INPUT_WIDTH, INPUT_HEIGHT))

    #Cast to 8-bit unsigned integer
    output = np.uint8(output)

    #Output an image if required while uint8
    if save_image:
        save_array_as_fig(output, 'test')

    #Now cast
    output = np.float16(output)/255.0

    #Add two more channels to get RBG
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

#Boots up a concurrently running client that keeps the trainable graph in memory
#to speed up training time. Updates depending on argument at the end of its run
def run_training_client(graph_name, port, optimise_and_save, batch_size, total_units):
    #Connect to specified port
    print("Connecting client to port " + str(port))
    sock = socket.socket()
    sock.connect(('', port))
    sock.setblocking(0) #Throw an exception when out of data to read (non-blocking)
    timeout = 0.5       #Timeout before cutting recv'ing loop in seconds

    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    print("Loading graph '" + graph_name + "' for " + ("training" if optimise_and_save == 1 else "validation"))
    saver = restore_model(graph_name, sess)

    #Prepare to track information
    t_pos = 0   #Correct classified a galaxy
    f_pos = 0   #Guessed galaxy when should have been noise
    t_neg = 0   #Correctly classified noise
    f_neg = 0   #Guessed noise when should have been galaxy

    #Recieve supervised data. Take 4 times as many as recieving uint32s (kdu_uint32s)
    #and convert
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
        batch_ready = len(image_batch) == batch_size and len(label_batch) == batch_size;
        if batch_ready or (not recieving and len(image_batch) != 0):
            #Turn recieved info into a feeder dictionary
            feed_dict = {
                            'images:0': image_batch,
                            'labels:0': label_batch
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
            print("-units     = " + str(units_num) + "/" + str(total_units) + " (" + "{0:.4f}".format(100*units_num/total_units) + "% of units fed)")
            #Prints current learning rate
            print("-alpha     = " + str(sess.run('alpha:0', feed_dict=feed_dict)))
            #Prints current loss
            print("-loss      = " + str(sess.run('loss:0', feed_dict=feed_dict)))

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
                                "{0:.4f}".format(100*batch_t_pos/(batch_t_pos + batch_f_neg)) if (batch_t_pos + batch_f_neg != 0) else "-"],

                                ['SESSION',
                                t_pos,
                                f_pos,
                                t_pos + f_pos,
                                "{0:.4f}".format(100*t_pos/(t_pos + f_neg)) if (t_pos + f_neg != 0) else "-"]
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
                                "{0:.4f}".format(100*batch_t_neg/(batch_t_neg + batch_f_pos)) if (batch_t_neg + batch_f_pos != 0) else "-"],

                                ['SESSION',
                                t_neg,
                                f_neg,
                                t_neg + f_neg,
                                "{0:.4f}".format(100*t_neg/(t_neg + f_pos)) if (t_neg + f_pos != 0) else "-"]
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
        update_model(graph_name, sess, saver)

    #Close tensorflow session
    sess.close()

#Softmax is considered a large typed operation on the ncs, so a version
#has been implemented here
def softmax(arr):
    """Compute softmax values for each sets of scores in x."""
    e_arr = np.exp(arr - np.max(arr))
    return e_arr / e_arr.sum()

#Helper to plot an evaluation probability map
def plot_prob_map(prob_map):
    #Save the probability map to output in 2d componetn slices
    for f in range(prob_map.shape[2]):
        fig = plt.figure("component-" + str(f), figsize=(15, 15), dpi=80)  #dims*dpi = res

        #Bounds
        w = prob_map.shape[0]
        h = prob_map.shape[1]

        #Plot
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')    #Aspect ratio

        ax.set_xticks(np.arange(0, w, math.floor(w/10)))  #Ticks
        ax.set_xticks(np.arange(0, w, 1), minor = True)

        ax.set_yticks(np.arange(0, h, math.floor(h/10)))
        ax.set_yticks(np.arange(0, h, 1), minor = True)

        #Colourisation mapped to [0,1], as this is a probability map
        plt.imshow( prob_map[:,:,f], cmap="bone", vmin=0.0, vmax=1.0,
                    interpolation='nearest')

        #Label and save
        fig.savefig("output/" + "component-" + str(f))

        #Explicity close figure for memory usage
        plt.close(fig)

#Recieves a unit and evaluates it using the graph
def run_evaluation_client_for_cpu(  graph_name,        #Graph to evaluate on
                                    port,              #Port to stream from
                                    region_width,      #Width of region to evaluate
                                    region_height,     #Height of region to evaluate
                                    region_depth,      #Depth of region to evaluate
                                    units_per_component):   #How many images at each frequency

    sock = socket.socket()
    sock.connect(('', port))
    sock.setblocking(0) #Throw an exception when out of data to read (non-blocking)
    timeout = 0.5       #How many seconds to wait before finishing

    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Create a graph for evaluation
    new_graph(id=graph_name,        #Id/name
              filter_sizes=FILTER_SIZES,  #Convolutional layer filter sizes in pixels
              num_filters=NUM_FILTERS, #Number of filters in each Convolutional layer
              fc_sizes=FC_SIZES,    #Number of neurons in fully connected layer
              for_training=False)   #Gets rid of placeholders and training structures
                                    #that aren't needed for NCS

    #Use this to load in weights from trained graph
    saver = tf.train.Saver(tf.global_variables())

    #A session is required
    sess = tf.Session()

    #Initialise no placeholder architecture
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #Load in weights from trained graph
    saver.restore(sess, GRAPHS_FOLDER + "/" + graph_name + "/" + graph_name)

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
    inference_start = datetime.now()
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
            inference_duration = datetime.now() - inference_start - timedelta(seconds=timeout)
            print("\r100.000% complete (" + str(inference_duration) + ")")
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
            feed_dict_eval = {  'images:0': [image_input]  }

            #Get the prediction that it is a galaxy
            #(the 1th element of the one hot encoding)
            output = sess.run(OUTPUT_CPU_NAME + ':0', feed_dict=feed_dict_eval)[0]
            pred = softmax(output)[1]
            #print(pred)
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

    #Plot data or visualisation
    print("Plotting 2D component prediction map(s)")
    plot_prob_map(prob_map)

    #Close tensorflow session
    sess.close()

#Must occur pre NCS usage, creates a version of the .meta file without any
#training placeholders (otherwise will fail)
def compile_for_ncs(graph_name):
    #Create a graph, if graph is any larger then network will
    #not be Movidius NCS compatible (reason unknown)
    new_graph(id=graph_name,      #Id/name
              filter_sizes=FILTER_SIZES,  #Convolutional layer filter sizes in pixels
              num_filters=NUM_FILTERS, #Number of filters in each Convolutional layer
              fc_sizes=FC_SIZES,    #Number of neurons in fully connected layer
              for_training=False)   #Gets rid of placeholders and training structures
                                    #that aren't needed for NCS

    #Prepare to save all this stuff
    saver = tf.train.Saver(tf.global_variables())

    #A session is required
    sess = tf.Session()

    #Initialise no placeholder architecture
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #Load in weights from trained graph
    saver.restore(sess, GRAPHS_FOLDER + "/" + graph_name + "/" + graph_name)

    #Save without placeholders
    saver.save(sess, GRAPHS_FOLDER + "/" + graph_name + "/" + graph_name + "-for-ncs")

    #Finish session
    sess.close()

    #Compile graph with subprocess for NCS
    #Example: 'mvNCCompile graphs/test-graph/test-graph.meta -in=images -on=dense_final/BiasAdd -o=./graphs/test-graph/test-graph-for-ncs.graph'
    subprocess.call([
        'mvNCCompile',
        (GRAPHS_FOLDER + "/" + graph_name + "/" + graph_name + "-for-ncs.meta"),
        "-in=" + INPUT_NAME,
        "-on=" + OUTPUT_NCS_NAME, #No predictor in eval graph because softmax broken
        "-o=" + GRAPHS_FOLDER + "/" + graph_name + "/" + graph_name + "-for-ncs.graph",
        "-is", str(INPUT_WIDTH), str(INPUT_HEIGHT)
    ]
    #,stdout=open(os.devnull, 'wb')  #Suppress output
    );

#Creates a process that manages an NCS (intakes new input data and writes
#output data to shared memory)
def run_ncs(device, device_index, graph_ref, child_end,
            shared_prob_map, shared_sample_map,
            w, h, d,
            shared_count, expected):
    #Reshape shared memory as a numpy array by first making 1d
    prob_ag_map_1d = np.frombuffer(shared_prob_map.get_obj())
    sample_map_1d = np.frombuffer(shared_sample_map.get_obj())

    #And then reshaping
    prob_ag_map = prob_ag_map_1d.reshape((w, h, d))
    sample_map = sample_map_1d.reshape((w, h, d))

    print(str(device_index) + " " + str(graph_ref))

    #Begin processing loop
    while True:
        #Read the pipe for new data, when it comes in, unpack it. There could be
        #some delay at the start while booting NCS'
        data = child_end.recv()

        print(str(device_index) + " data rec")
        #print(data)

        #Check if ending flag has been sent, in which case close
        if data[0] == "CLOSED":
            break

        #Unpack data into region bounds to aggregate prediction onto and
        #actual image data for inferencing
        image_tlx   = data[0]
        image_brx   = image_tlx + INPUT_WIDTH
        image_tly   = data[1]
        image_bry   = image_tly + INPUT_HEIGHT
        image_f     = data[2]
        image_input = data[3]
        image_name  = "image:" + str(image_tlx) + ":" + str(image_tly) + ":" + str(image_f)

        #Get the graph's prediction
        print(str(device_index) + " preload")
        if graph_ref.LoadTensor(image_input, image_name):
            print(str(device_index) + " postload")
            #Get output of graph
            output, userobj = graph_ref.GetResult()
            print(str(device_index) + " getres")


            #One hot encoding, want probability of class 1 (galaxy).
            #Note ncsdk doesn't support the predictor layer which softmaxes
            #the last dense layer to give class probability, so manual
            #softmax must be done in its place
            pred = softmax(output)[1]
            #print(pred)

            #Incremnt and report shared count
            shared_count.value += 1
            print("\r{0:.4f}".format(100*shared_count.value/expected) + "% complete", end="")

            #Write information into heatmap. Likelihood is simply added onto
            #heat map at each pixel
            prob_ag_map[image_tlx:image_brx,
                        image_tly:image_bry,
                        image_f] += pred
            sample_map[image_tlx:image_brx,
                        image_tly:image_bry,
                        image_f] += 1.0

        else:
            print("Error: cannot evaluate output of neural network on NCS:" + str(device_index))
            sys.exit()

    print(str(device_index) + " done")

    #End process
    sys.exit()

#Boots up one NCS, loads a compiled version of the graph onto it and begins
#running inferences on it. Supports inferencing a 3d area that must be supplied
def run_evaluation_client_for_ncs(  graph_name,        #Graph to evaluate on
                                    port,              #Port to stream from
                                    region_width,      #Width of region to evaluate
                                    region_height,     #Height of region to evaluate
                                    region_depth,      #Depth of region to evaluate
                                    units_per_component):   #Needed to calculate expected units

    #Compile a copy of the evaluation graph for running on the ncs
    compile_for_ncs(graph_name)

    #Load the compiled graph
    graph_file = None;
    filepath = GRAPHS_FOLDER + "/" + graph_name + "/" + graph_name + "-for-ncs.graph"
    with open(filepath, mode='rb') as f:
        #Read it in
        graph_file = f.read()
    if graph_file == None:
        print("Error: evaluation graph compiled for ncs could not be read, exiting")
        sys.exit()

    #Data will need to be stored in a heat map - allocate memory for this
    #data structure. Probability aggregate is the sum of each predictions made
    #that include that pixel. Samples are the number of predictions made for
    #that pixel. This allows normalised probability map as output. These
    #are initialised as shared memory arrays (1D) and reshaped in child procs
    print("Creating shared memory data structures for multiprocessing")
    shared_prob_map = mp.Array(ctypes.c_double, region_width*region_height*region_depth)
    shared_sample_map = mp.Array(ctypes.c_double, region_width*region_height*region_depth)

    #Reshape shared memory as a numpy array by first making 1d
    prob_ag_map_1d = np.frombuffer(shared_prob_map.get_obj())
    sample_map_1d = np.frombuffer(shared_sample_map.get_obj())

    #And then reshaping. Now this memory structure can be used as a numpy
    #array between processes
    prob_ag_map = prob_ag_map_1d.reshape((region_width, region_height, region_depth))
    sample_map = sample_map_1d.reshape((region_width, region_height, region_depth))

    #For tracking time taken
    inference_start = datetime.now()

    #Images per freq frame*freq frame for total expected images
    expected = units_per_component*region_depth
    shared_count = mp.Value('i', 0)

    #Ensure there is at least one NCS
    print("Finding and opening device(s)")
    device_name_list = mvnc.EnumerateDevices()
    num_devices = len(device_name_list)
    if num_devices == 0:
        print("none found! Exiting")
        sys.exit()

    #For whatever reason multiple allocation must be done like so - thanks ncsdk
    device_handles = []
    graph_handles = []
    for d in range(num_devices):
        #Get device
        device_handles.append(mvnc.Device(device_name_list[d]))

        #Open device
        device_handles[d].OpenDevice()

        #Allocate the compiled graph onto the device and get a reference
        #for later deallocation
        graph_handles.append(device_handles[d].AllocateGraph(graph_file))

        #Set graph options to allow for efficiency with multiple NCS'
        #this prevents LoadTensor() and GetResult() from blocking (they
        #will return immediately)
        graph_handles[d].SetGraphOption(mvnc.GraphOption.DONT_BLOCK, 1)

    #In the 'parent' process, create a new process for each NCS with a pipe
    #to put input data and a reference to the shared probability map
    '''
    pipes = []
    children = []
    for d in range(num_devices):
        #Create a pipe for this child and keep track of it
        parent_end, child_end = mp.Pipe()
        pipes.append(parent_end)

        #Begin the ncs child process
        child = mp.Process( target=run_ncs,
                            args=(  None, d,
                                    graph_handles[d], child_end,
                                    shared_prob_map, shared_sample_map,
                                    region_width, region_height, region_depth,
                                    shared_count, expected,))
        children.append(child)

    #Start your engines
    for c in children:
        c.start()
    '''

    #Begin recv'ing data and piping it off to children
    #Child, connect to socket
    print("Connecting NCS manager to port " + str(port))
    sock = socket.socket()
    sock.connect(('', port))
    sock.setblocking(0) #Throw an exception when out of data to read (non-blocking)
    timeout = 0.5       #How many seconds to wait before finishing

    #Begin inference timer
    inference_start = datetime.now()

    #Track which child should be sent data
    curr_child = 0

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

        #When ready then recv
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
            #No longer reciving, may still be inferencing
            recieving = False

        #If something was recieved from socket
        if recieving:
            #Make the recv'd data graph compatible
            image_input = make_compatible(image_input, False)
            #Explicitly make into '1' batch tensor, NCS won't take it as a list
            image_input = np.reshape(image_input, (1, INPUT_WIDTH, INPUT_HEIGHT, 3))
            image_tlx   = image_loc[0][0]
            image_brx   = image_tlx + INPUT_WIDTH
            image_tly   = image_loc[1][0]
            image_bry   = image_tly + INPUT_HEIGHT
            image_f     = image_loc[2][0]
            #CSV encoding location
            image_name  = str(image_tlx) + ":" +    \
                          str(image_brx) + ":" +    \
                          str(image_tly) + ":" +    \
                          str(image_bry) + ":" +    \
                          str(image_f)

            #Try to load data onto an NCS
            loaded = False
            while not loaded:
                #Go through all NCS' and attempt to load this new data
                for graph_ref in graph_handles:
                    #Try to load new data
                    loaded = graph_ref.LoadTensor(image_input, image_name)

                    #If it failed then try to get inference of what must be
                    #currently loaded onto device
                    if not loaded:
                        output, userobj = graph_ref.GetResult()

                        if userobj == None:
                            #If no inference then still inferencing, so try next NCS
                            continue
                        else:
                            #Otherwise process inference results. Get location
                            #of the processed area through the userobj string
                            loc = [int(x) for x in userobj.split(':')]

                            #One hot encoding, want probability of class 1 (galaxy).
                            #Note ncsdk doesn't support the predictor layer which softmaxes
                            #the last dense layer to give class probability, so manual
                            #softmax must be done in its place
                            pred = softmax(output)[1]

                            #Incremnt and report shared count
                            shared_count.value += 1
                            print("\r{0:.4f}".format(100*shared_count.value/expected) + "% complete", end="")

                            #Write information into heatmap. Likelihood is simply added onto
                            #heat map at each pixel
                            prob_ag_map[loc[0]:loc[1],
                                        loc[2]:loc[3],
                                        loc[4]] += pred
                            sample_map[ loc[0]:loc[1],
                                        loc[2]:loc[3],
                                        loc[4]] += 1.0

                    else:
                        #Otherwise it succeeded, so stop trying to load this image
                        #and go to next one
                        break

            '''
            #Get the graph's prediction
            if graph_handles[curr_child].LoadTensor(image_input, image_name):
                #Get output of graph
                output, userobj = graph_handles[curr_child].GetResult()

                #One hot encoding, want probability of class 1 (galaxy).
                #Note ncsdk doesn't support the predictor layer which softmaxes
                #the last dense layer to give class probability, so manual
                #softmax must be done in its place
                pred = softmax(output)[1]
                #print(pred)

                #Incremnt and report shared count
                shared_count.value += 1
                print("\r{0:.4f}".format(100*shared_count.value/expected) + "% complete", end="")

                #Write information into heatmap. Likelihood is simply added onto
                #heat map at each pixel
                prob_ag_map[image_tlx:image_brx,
                            image_tly:image_bry,
                            image_f] += pred
                sample_map[image_tlx:image_brx,
                            image_tly:image_bry,
                            image_f] += 1.0

            else:
                print("Error: cannot evaluate output of neural network on NCS:" + str(device_index))
                sys.exit()

            #Send it over to an NCS quick smart and get ready to send to next NCS
            #pipes[curr_child].send([image_tlx, image_tly, image_f, image_input])
            curr_child = 0 if curr_child == num_devices - 1 else curr_child + 1
            '''

    #If here then all of the data has been recv'd from the C++ kakadu wrapper,
    #so just wait for the last few inferences
    while(shared_count.value != expected):
        #Go through all NCS' and attempt to get inference results
        for graph_ref in graph_handles:
            #Try to get inference results and check if inference is yet complete
            output, userobj = graph_ref.GetResult()

            if userobj == None:
                #If no inference then still inferencing, so try next NCS
                continue
            else:
                #Otherwise process inference results. Get location
                #of the processed area through the userobj string
                loc = [int(x) for x in userobj.split(':')]

                #One hot encoding, want probability of class 1 (galaxy).
                #Note ncsdk doesn't support the predictor layer which softmaxes
                #the last dense layer to give class probability, so manual
                #softmax must be done in its place
                pred = softmax(output)[1]

                #Incremnt and report shared count
                shared_count.value += 1
                print("\r{0:.4f}".format(100*shared_count.value/expected) + "% complete", end="")

                #Write information into heatmap. Likelihood is simply added onto
                #heat map at each pixel
                prob_ag_map[loc[0]:loc[1],
                            loc[2]:loc[3],
                            loc[4]] += pred
                sample_map[ loc[0]:loc[1],
                            loc[2]:loc[3],
                            loc[4]] += 1.0

    '''
    #Close all pipes. This is janky but python3 multiprocessing pipes lack a
    #good closing method - can't use poll! Also can't use queues which have the
    #empty method since we're going for speed and queues are built on top of pipes
    for pipe in pipes:
        pipe.send(["CLOSED"])
        pipe.close()

    #Wait for NCS children (who may still be inferencing) to die
    for child in children:
        child.join()
    '''

    #Deallocate graphs and close devices
    for d in range(num_devices):
        #Finished recieving, deallocate the graph from the device
        graph_handles[d].DeallocateGraph()

        #Close opened device
        device_handles[d].CloseDevice()

    #Inferencing now complete
    inference_duration = datetime.now() - inference_start - timedelta(seconds=timeout)
    print("\r{0:.4f}".format(100*shared_count.value/expected) + "% complete (" + str(inference_duration) + ")")

    #Normalise the probability by dividing the aggregate prob by the amount
    #of predictions/samples made at that pixel. Handle divide by zeroes which may
    #occur along edges
    print("Normalising prediction map")
    prob_map = np.divide(   prob_ag_map, sample_map,
                            out=np.zeros_like(prob_ag_map),
                            where=sample_map!=0)

    #Plot data or visualisation
    print("Plotting 2D component prediction map(s)")
    plot_prob_map(prob_map)

#Plots the convolutional filter-weights/kernel for a given layer using matplotlib
def plot_conv_weights(graph_name, scope, start_conv, final_conv, suffix):
    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained
    saver = restore_model(graph_name, sess)

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
              for_training):  #The Movidius NCS' are picky and won't resolve unknown
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

    if for_training:
        #and supervisory signals which are boolean (is or is not a galaxy)
        labels = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
        print("\t\t" + '{:20s}'.format("-Label placeholder ") + " : " + str(labels))

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
            bias_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.05),
            activation=tf.nn.relu,
            trainable=True,
            name="conv_" + str(i)
        )
        print("\t\t" + '{:20s}'.format("-Convolutional ") + str(i) + ": " + str(layer))
        '''
        #Apply pooling
        layer = tf.layers.average_pooling2d(
            inputs=layer,
            pool_size=2,
            strides=1,
            padding='SAME',
            name="pooling_" + str(i)
        )
        print("\t\t" + '{:20s}'.format("-Average pooling ")  + str(i) + ": " + str(layer))
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
            bias_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.05),
            trainable=True,
            name="dense_" + str(i)
        )
        print("\t\t" + '{:20s}'.format("-Dense ") + str(i) + ": " + str(layer))

    #The final layer is a neuron for each class
    layer = tf.layers.dense(
        inputs=layer,
        units=2,
        activation=None,    #Note no ReLU
        use_bias=True,
        bias_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.05),
        trainable=True,
        name="dense_final"
    )
    print("\t\t" + '{:20s}'.format("-Dense (final)") + " : " + str(layer))

    #The following structures are only required when training
    if for_training:
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
        global_step = tf.Variable(0, trainable=False)   #Incremented per batch
        init_alpha = 0.0001
        decay_base = 1      #alpha = alpha*decay_base^(global_step/decay_steps)
        decay_steps = 64
        alpha = tf.train.exponential_decay( init_alpha,
                                            global_step, decay_steps, decay_base,
                                            name='alpha')
        print("\t\t" + '{:20s}'.format("-Learning rate") + " : " + str(alpha), end="")
        print(" (" + str(init_alpha) + "*" + str(decay_base) + "^(batch_no/" + str(decay_steps) + ")")

        #Optimisation function to Optimise cross entropy will be Adam optimizer
        #(advanced gradient descent)
        optimiser = (
            tf.train.AdamOptimizer(learning_rate=init_alpha)
            .minimize(loss, global_step=global_step)    #Decay learning rate
                                                        #by incrementing global step
                                                        #once per batch
        )
        print("\t\t" + '{:20s}'.format("-Optimiser") + " : " + str(optimiser.name))

#Wraps the above to make a basic convolutional neural network for binary
#image classification
def new_basic_training_graph(id):
    #Create a graph, if graph is any larger then network will
    #not be Movidius NCS compatible (reason unknown)
    new_graph(id,      #Id/name
              filter_sizes=FILTER_SIZES,    #Convolutional layer filter sizes in pixels
              num_filters=NUM_FILTERS,      #Number of filters in each Convolutional layer
              fc_sizes=FC_SIZES,            #Number of neurons in fully connected layer
              for_training=True)

    #Save it in a tensorflow session
    sess = tf.Session()
    save_model_as_meta(id, sess)
    sess.close()

#Restores the model (graph and variables) from a supplied filepath
def restore_model(id, sess):
    #Location is in graphs/id
    filepath = GRAPHS_FOLDER + "/" + id + "/" + id

    #Load from file
    saver = tf.train.import_meta_graph(filepath + ".meta")  #Graph structure
    saver.restore(sess, filepath)                       #Variables

    #Return the saver
    return saver

#Updates a model (used in training)
def update_model(id, sess, saver):
    #Remove old if it exists (don't complain if it doesn't)
    shutil.rmtree(GRAPHS_FOLDER + "/" + id)

    #File is named id in id folder
    filepath = GRAPHS_FOLDER + "/" + id + "/" + id

    #Saving operation
    saver.save(sess, filepath)                      #Variables
    saver.export_meta_graph(filepath + ".meta")     #Graph structure

#Saves the current model (graph and variables) to a supplied filepath
def save_model_as_meta(id, sess):
    #Remove old if it exists (don't complain if it doesn't)
    shutil.rmtree(GRAPHS_FOLDER + "/" + id, True)

    #File is named id in id folder
    filepath = GRAPHS_FOLDER + "/" + id + "/" + id

    #Initialise if required
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #Saving operation
    saver = tf.train.Saver(max_to_keep=1)           #Keep only one copy
    saver.save(sess, filepath)                      #Variables
    saver.export_meta_graph(filepath + ".meta")     #Graph structure

#Export's the current model (graph and variables) to a pb MVNCcompile-able
#serialized graph file
def save_model_as_pb(id, sess):
    #Announce
    print("Exporting model to: '" + GRAPHS_FOLDER + "/" + id + "/" + id + ".pb' ")

    #Get graph
    graph = sess.graph_def

    #Notify of placeholder names to use when compiling for evaluation
    #print("& including the following operations:")
    #ops = sess.graph.get_operations()
    #print(ops)

    #print(sess.graph.get_tensor_by_name("images:0"))

    #Export to specified file
    with open(GRAPHS_FOLDER + "/" + id + "/" + id + ".pb", 'wb') as f:
        f.write(graph.SerializeToString())
