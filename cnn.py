#! /usr/bin/env python3

#Visualise with: 'tensorboard --logdir='logs''

#Imports
import sys                          #For embedded python workaround
import os                           #For file reading and warning suppression

#Embedded python work around
if not hasattr(sys, 'argv'):
    sys.argv = ['']

import shutil                       #For directory deletion
import numpy as np                  #For transforming blocks
import matplotlib.pyplot as plt     #For visualisation
import math                         #For logs
import time                         #For debugging with catchup
import subprocess                   #For compiling graphs
import tensorflow as tf             #For deep learning
from mvnc import mvncapi as mvnc    #For Movidius NCS API

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

#Hardcoded image input dimensions
WIDTH = 32
HEIGHT = 32

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

    #Constrain axis proportions and plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow( img_array, cmap="Greys_r", vmin=0, vmax=255,
                interpolation='nearest')

    fig.savefig("output/" + name)

    #Explicity close figure for memory usage
    plt.close(fig)

#Transforms the standard uint8 1 channel image gotten from the Kakadu SDK into
#a three channel RGB (redundant channels) half precision float (float16) image
#such that it is compatible with the Movidius NCS architecture
def make_compatible(image_data, save_image):
    #Reshape to placeholder dimensions
    output = np.reshape(image_data, (WIDTH, HEIGHT))

    #Cast to 8-bit unsigned integer
    output = np.uint8(output)

    #Output an image if required while uint8
    if save_image:
        save_array_as_fig(output, 'test')

    #Now cast
    output = np.float16(output)

    #Add two more channels to get RBG
    output = np.dstack([output]*3)

    #Give it back
    return output

#Recieves a training unit and trains the graph on it
def use_supervised_batch(   image_data_batch,
                            label_batch,
                            graph_name,
                            optimise_and_save):
    batch_size = len(image_data_batch)

    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    saver = restore_model(graph_name, sess)

    #Convert to feedable format
    image_input = []
    label_input = []
    for i in range(0, batch_size):
        #Transform to tensorflow/Movidius NCS compatible and add to list of input
        image_input.append(make_compatible(image_data_batch[i], False))

        #Convert label input into array of scalar arrays
        label_input.append([label_batch[i]])

    #Create a unitary feeder dictionary
    feed_dict = {   'images:0': image_input,
                    'labels:0': label_input,
                    'is_training:0': update_model}

    #Only optimise & save the graph if in training mode
    if optimise_and_save == 1:
        #Feed into the network so it can 'learn' by running the adam optimiser
        sess.run('Adam', feed_dict=feed_dict)

        #Save the slightly more trained graph if in training mode
        update_model(graph_name, sess, saver)

    #Prints current loss
    print("\t-loss (mean sigmoid cross entropy of batch) = " + str(sess.run('loss:0', feed_dict=feed_dict)))
    #Prints current learning rate
    print("\t-alpha = " + str(sess.run('alpha:0', feed_dict=feed_dict)))

    #What is the prediction for each image? (as bool)
    preds = [x[0] > 0.5 for x in sess.run('predictor:0', feed_dict=feed_dict)]

    #Close tensorflow session
    sess.close()

    #Return the training predictions
    return preds

#Recieves a unit and evaluates it using the graph
def use_evaluation_unit_on_cpu( np_array,
                                graph_name):

    #Convert to tensorflow/Movidius NCS compatible
    image_input = np.reshape(   make_compatible(np_array, True),
                                (1, WIDTH, HEIGHT, 3))

    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    saver = restore_model(graph_name, sess)

    #Create a unitary feeder dictionary
    feed_dict_eval = {'images:0': image_input, 'is_training:0': False}

    #Get the prediction
    pred = sess.run('predictor:0', feed_dict=feed_dict_eval)[0]
    print(sess.run('predictor:0', feed_dict=feed_dict_eval))

    #Close tensorflow session
    sess.close()

    #Return result
    return pred

'''
#Loads a specified graph onto the connected NCS devices
def compile_and_allocate_graph_onto_ncs(graph_name):
'''

#Recieves a unit and evaluates it using the graph on 1 or more Movidius NCS'
def use_evaluation_unit_on_ncs( np_array,
                                graph_name):
    #Load the compiled graph
    #print("Loading compiled graph")
    graph_file = None;
    filepath = GRAPHS_FOLDER + "/" + graph_name + "/" + graph_name + "-for-ncs.graph"
    with open(filepath, mode='rb') as f:
        #Read it in
        graph_file = f.read()

    #Find 1 or more NCS'
    devicesNames = mvnc.EnumerateDevices()
    if len(devicesNames) > 0:
        #print("Enumerated " + str(len(devicesNames)) + " devices")

        #For every device allocate the compiled graph
        count = 0
        for deviceName in devicesNames:
            #Open the device
            #print("Opening device " + str(count))
            device = mvnc.Device(deviceName)
            device.OpenDevice()

            #Put the compiled graph onto the device and get a reference
            #for later deallocation
            #print("Allocating compiled graph onto device")
            graph_ref = device.AllocateGraph(graph_file)

            #Image input must conform to input placeholder dimensions
            image_input = np.reshape(   make_compatible(np_array, True),
                                        (1, WIDTH, HEIGHT, 3))
            print(image_input.shape)

            #Track the prediction
            pred = None
            if graph_ref.LoadTensor(image_input, "images"):
                #Get output of graph
                output, userobj = graph_ref.GetResult()

                #print("GALAXY" if output[0] > 0.5 else "NOISE")
                print(output.shape)
                print(output)
                pred = output[0]

            else:
                print("Error evaluating output of neural network, continue")
                continue

            #Deallocate the graph from the device
            #print("Deallocating compiled graph from device")
            graph_ref.DeallocateGraph()

            #Close opened device
            #print("Closing device " + str(count))
            device.CloseDevice()

            #TODO, don't exit immediately
            #Return the prediction -> 0 for noise, 1 for galaxy
            return pred

            count = count + 1

    else:
        print("No devices to enumerate, exiting")

'''
#Removes the allocated graph from the NCS and closes the device
def deallocate_graph_from_ncs():
'''

#Plots the convolutional filter-weights for a given layer using matplotlib
def plot_conv_weights(graph_name, layer_name):
    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    saver = restore_model(graph_name, sess)

    #Retrieve the weights (don't need a feed dict as nothing is being calculated)
    #A feed-dict is not necessary because nothing is calculated
    w = sess.run([v for v in tf.global_variables() if v.name == (layer_name + ":0")][0])

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
                      interpolation='nearest', cmap='Greys_r')

        #Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig("output/" + layer_name)

    #Explicity close figure for memory usage
    plt.close(fig)

    #Close tensorflow session (no need to save)
    sess.close()

#Helper function for creating filters as a tensorflow variable within each layer
def new_weights(shape, var_name):
    #Weights start as noise and are eventually learned
    return tf.Variable(tf.truncated_normal(shape, stddev=0.001), name=var_name)

#Helper function for creating biases as a tensorflow variable within each layer
def new_biases(length, var_name):
    #Biases start as noise and are eventually learned
    return tf.Variable(tf.constant(0.001, shape=[length]), name=var_name)

#Helper function for creating a new convolution layer in a graph
def new_conv_layer(prev_layer,         #the previous layer (input to this layer)
                   num_input_channels, #how many channels in input image
                   filter_size,        #width & height of each filter
                   num_filters,        #number of filters
                   conv_index):        #for naming

    #Stringify index
    conv_index = str(conv_index)

    #Create filters with a given shape to be optimised over graph execution
    #rank must be 4 for tensorflow conv2d
    weights = new_weights(  shape=[filter_size, filter_size, num_input_channels, num_filters],
                            var_name=("conv_weights_" + conv_index))

    #Create biases to be optimised over graph execution
    biases = new_biases(length=num_filters, var_name=("conv_biases_" + conv_index))

    #This layer is a 2D convolution with padding of zeroes to ensure that shape
    #remains consistent during convolution
    layer = tf.nn.conv2d(input=prev_layer,
                         filter=weights,
                         strides=[1, 1, 1, 1],    #X & Y pixel strides are args 2 & 3
                         padding='SAME',
                         name=('conv_2d_' + conv_index))

    #Add bias
    layer += biases

    #Apply pooling
    layer = tf.nn.max_pool(value=layer,         #Take the layer
                           ksize=[1, 2, 2, 1],  #2x2 max pooling over resolution
                           strides=[1, 2, 2, 1],
                           padding='SAME',      #Consistent shape
                           name=('max_pool_' + conv_index))

    #Apply activation function
    layer = tf.nn.relu(layer, name=('conv_' + conv_index))

    #Return the layer (and also the weights for inspection)
    return layer

#Helper function for flattening a layer before it is fed to a fully connected layer
def flatten_layer(layer):
    #Get the shape of the input layer
    layer_shape = layer.get_shape()

    #The shape of the input layer is assumed to be
    #[num_images, img_height, img_width, num_channels]

    #The number of features is: img_height * img_width * num_channels,
    #calculated here
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features], name='conv_flatten')

    #The shape of the flattened layer is now
    #[num_images, img_height * img_width * num_channels]

    #Return both the flattened layer and the number of features.
    return layer_flat, num_features

#Helper function for creating a new fully connected layer in a graph
def new_fc_layer(prev_layer,         #the previous layer (input to this layer)
                 num_inputs,         #number of images to be input
                 num_outputs,        #number to be output
                 final_fc_layer,
                 fc_index):          #for unique variable naming

    #Stringify index
    fc_index = str(fc_index)

    #Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs], var_name=("fc_weights_" + fc_index))
    biases = new_biases(length=num_outputs, var_name=("fc_biases_" + fc_index))

    #Calculate the layer as the matrix multiplication of
    #the input and weights, and then add the bias-values
    layer = tf.matmul(prev_layer, weights, name=('fc_matmul_' + fc_index)) + biases

    #Apply activation function if requested
    if not final_fc_layer:
        #Apply ReLU
        layer = tf.nn.relu(layer, name='fc_' + fc_index)
    else:
        #No ReLU, just name
        layer = tf.identity(layer, name='fc_' + fc_index)

    return layer

#Initialises a fresh graph and stores it for later training
def new_graph(id,             #Unique identifier for saving the graph
              filter_sizes,   #Filter dims for each convolutional layer (kernals)
              num_filters,    #Number of filters for each convolutional layer
              fc_sizes,       #Number of neurons in fully connected layers
              for_training):  #The Movidius NCS' are picky and won't resolve unknown
                              #placeholders. For loading onto the NCS these, and other
                              #training structures (optimizer, dropout, batch normalisation)
                              #all must go

    #Create computational graph to represent the neural net:
    print("Creating new graph: '" + id + "'")
    print("\t*Structure details:")

    #Movidius NCS requires an RGB 'colour' image
    channels = 3

    #INPUT
    #Placeholders serve as variable input to the graph (can be changed when run)
    #Following placeholder takes 30x30 grayscale images as tensors
    #(must be float32 for convolution and must be 4D for tensorflow)
    images = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, channels], name='images')
    print("\t\t-Placeholder '" + images.name + "': " + str(images))

    if for_training:
        #and supervisory signals which are boolean (is or is not a galaxy)
        labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
        print("\t\t-Placeholder '" + labels.name + "': " + str(labels))
        #Whether or not we are training (for batch normalisation)
        is_training = tf.placeholder(tf.bool, name='is_training')

    #CONVOLUTIONAL LAYERS
    #These convolutional layers sequentially take inputs and create/apply filters
    #to these inputs to create outputs. They create weights and biases that will
    #be optimised during graph execution. They also down-sample (pool) the image
    #after doing so. Filters are created in accordance with the arguments to this
    #function
    layer_conv0 = new_conv_layer(   prev_layer=images,
                                    num_input_channels=channels,
                                    filter_size=filter_sizes[0],
                                    num_filters=num_filters[0],
                                    conv_index=0)
    print("\t\t-Convolutional 0: " + str(layer_conv0))

    if for_training:
        #Apply batch normalisation after ReLU
        layer_conv0 = tf.layers.batch_normalization(layer_conv0, training=is_training)

    #layer 2 takes layer 1's output
    layer_conv1 = new_conv_layer(   prev_layer=layer_conv0,
                                    num_input_channels=num_filters[0],
                                    filter_size=filter_sizes[1],
                                    num_filters=num_filters[1],
                                    conv_index=1)
    print("\t\t-Convolutional 1: " + str(layer_conv1))

    if for_training:
        #Apply batch normalisation after ReLU
        layer_conv1 = tf.layers.batch_normalization(layer_conv1, training=is_training)

    #Fully connected layers only take 2D tensors so above output must be
    #flattened from 4d
    layer_flat, num_features = flatten_layer(layer_conv1)
    print("\t\t-Flattener 0: " + str(layer_flat))

    #FULLY CONNECTED LAYERS
    #These fully connected layers create new weights and biases and matrix
    #multiply the weights with the inputs, then adding the biases. They then
    #apply a ReLU function before returning the layer. These weights and biases
    #are learned during execution
    layer_fc0 = new_fc_layer(prev_layer=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_sizes[0],
                             final_fc_layer=False,
                             fc_index=0)
    print("\t\t-Fully connected 0: " + str(layer_fc0))

    if for_training:
        #Apply batch normalisation after ReLU
        layer_fc0 = tf.layers.batch_normalization(layer_fc0, training=is_training)

    layer_fc1 = new_fc_layer(prev_layer=layer_fc0,
                             num_inputs=fc_sizes[0],
                             num_outputs=fc_sizes[1],
                             final_fc_layer=False,
                             fc_index=1)
    print("\t\t-Fully connected 1: " + str(layer_fc1))

    if for_training:
        #Apply batch normalisation after ReLU
        layer_fc1 = tf.layers.batch_normalization(layer_fc1, training=is_training)

    layer_fc2 = new_fc_layer(prev_layer=layer_fc1,
                             num_inputs=fc_sizes[1],
                             num_outputs=1,
                             final_fc_layer=True,
                             fc_index=2)
    print("\t\t-Fully connected 2: " + str(layer_fc2))

    #PREDICTION DETAILS
    #Final fully connected layer suggests prediction (these structures are added to
    #collections for ease of access later on)
    print("\t*Prediction details:")

    #Softmax it to get a probability (Greater than 0.5 was used earlier to get
    #prediction as a boolean, but this is Unsupported by Movidius ncsdk at this time)
    prediction = tf.nn.sigmoid(layer_fc2, name='predictor')
    print("\t\t-Class prediction: " + str(prediction))


    if for_training:
        #Backpropogation details
        print("\t*Backpropagation details:")

        #COST FUNCTION
        #Cost function is cross entropy (+ve and approaches zero as the model output
        #approaches the desired output
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_fc2,
                                                                labels=labels,
                                                                name='sigmoid_cross_entropy')
        print("\t\t-Cross entropy: " + str(cross_entropy))
        cost = tf.reduce_mean(cross_entropy, name='loss')
        print("\t\t-Loss (reduced mean cross entropy): " + str(cost))

        #OPTIMISATION FUNCTION
        #Optimisation function will have a decaying learning rate for bolder retuning
        #at the beginning of the trainig run
        global_step = tf.Variable(0, trainable=False)   #Incremented per batch
        init_alpha = 0.1    #Ideally want to go down to 1e-4
        decay_base = 0.95   #alpha = alpha*decay_base^(global_step/decay_steps)
        decay_steps = 64    #With data set of 300000, this should get us to 0.0001
        alpha = tf.train.exponential_decay( init_alpha,
                                            global_step, decay_steps, decay_base,
                                            name='alpha')

        print("\t\t-Learning rate: " + str(alpha))

        #Optimisation function to Optimise cross entropy will be Adam optimizer
        #(advanced gradient descent)
        #Require the following extra ops due to batch normalisation
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimiser = (
                tf.train.AdamOptimizer(learning_rate=alpha)
                .minimize(cost, global_step=global_step)    #Decay learning rate
                                                            #by incrementing global step
                                                            #once per batch
            )

        print("\t\t-Optimiser: " + str(optimiser.name))

#Wraps the above to make a basic convolutional neural network for binary
#image classification
def new_basic_training_graph(id):
    #Create a graph, if graph is any larger then network will
    #not be Movidius NCS compatible (reason unknown)
    new_graph(id,      #Id/name
              filter_sizes=[5, 5],  #Convolutional layer filter sizes in pixels
              num_filters=[32, 32], #Number of filters in each Convolutional layer
              fc_sizes=[128, 48],    #Number of neurons in fully connected layer
              for_training=True)

    #Save it in a tensorflow session
    sess = tf.Session()
    save_model_as_meta(id, sess)
    sess.close()

#Must occur pre NCS usage, creates a version of the .meta file without any
#placeholders that aren't required for validation (otherwise will fail)
def compile_for_ncs(id):
    #Create a graph, if graph is any larger then network will
    #not be Movidius NCS compatible (reason unknown)
    new_graph(id,      #Id/name
              filter_sizes=[5, 5],  #Convolutional layer filter sizes in pixels
              num_filters=[32, 32], #Number of filters in each Convolutional layer
              fc_sizes=[128, 48],    #Number of neurons in fully connected layer
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
    saver.restore(sess, GRAPHS_FOLDER + "/" + id + "/" + id)

    #Save without placeholders
    saver.save(sess, GRAPHS_FOLDER + "/" + id + "/" + id + "-for-ncs")

    #Finish session
    sess.close()

    #Compile graph with subprocess for NCS
    #Example: 'mvNCCompile graphs/test-graph/test-graph.meta -in=images -on=predictor -o=./graphs/test-graph/test-graph-for-ncs.graph'
    subprocess.call([
        'mvNCCompile',
        (GRAPHS_FOLDER + "/" + id + "/" + id + "-for-ncs.meta"),
        "-in=images",
        "-on=predictor",
        "-s=12",
        "-o=" + GRAPHS_FOLDER + "/" + id + "/" + id + "-for-ncs.graph",
        "-is", str(WIDTH), str(HEIGHT)
    ],
    stdout=open(os.devnull, 'wb')); #Suppress output

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
