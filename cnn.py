#! /usr/bin/env python3

#Visualise with: 'tensorboard --logdir='logs''

#Imports
import sys              #For exiting

#Embedded python work around
if not hasattr(sys, 'argv'):
    sys.argv = ['']

#from mvnc import mvncapi as mvnc    #For Movidius NCS API
import os                           #For file reading and warning suppression
import shutil                       #For directory deletion
import numpy as np                  #For transforming blocks
import matplotlib.pyplot as plt     #For visualisation
import tensorflow as tf             #For deep learning
import math                         #For logs
import time                         #For debugging with catchup

#Global variables
GRAPHS_FOLDER = "./graphs"
LOGS_FOLDER = "./logs"

#Set logging level for Movidius NCS API
#mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
#Set non-verbose error checking for tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)
#Suppress binary compilation improvement suggestions befire import
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'  #1 filters all
                                        #2 filters warnings
                                        #3 filters none

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
    #Make sure there is a 0 pixel and a 255 pixel so colours are mapped correctly
    #on heatmap
    img_array[0][0][1] = 255;
    img_array[0][0][0] = 0;

    #Create graph to ensure that block was read correctly
    fig = plt.figure(name, figsize=(15, 15), dpi=80)  #dims*dpi = res

    #Constrain axis proportions and plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow(img_array[0], cmap="Greys_r")

    fig.savefig("output/" + name)

    #Explicity close figure for memory usage
    plt.close(fig)

#Recieves a training unit and trains the graph on it
def use_training_unit(  np_array,
                        width,
                        height,
                        isGalaxy,
                        rot,
                        graph_name):
    #Convert to uint8, [0, 255] is all that's needed
    np_array = np_array.astype(np.uint8)

    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    saver = restore_model(graph_name, sess)

    #Make image conform to placeholder dimensions
    image_input = np.reshape(np_array, (1, width, height))
    #Rotate image input as informed
    if rot == 90:
        image_input = np.rot90(image_input, k=1, axes=(1, 2))
    elif rot == 180:
        image_input = np.rot90(image_input, k=2, axes=(1, 2))
    elif rot == 270:
        image_input = np.rot90(image_input, k=3, axes=(1, 2))

    #Save a copy of the graph if required
    #save_array_as_fig(image_input, "test")

    #Make label conform to placeholder dimensions
    label_array = [0, 0]                            #One hot encoding
    label_array[isGalaxy] = 1
    label_input = np.reshape(label_array, (1, 2))

    #Create a unitary feeder dictionary
    feed_dict_train = {'images:0': image_input, 'labels:0': label_input}

    #Feed into the network so it can 'learn' by running the adam optimiser
    sess.run('Adam', feed_dict=feed_dict_train)

    #Check the training prediction
    #class_prob = tf.get_collection("class_prob")[0]
    #print("Probs:" + str(class_prob.eval(session=sess, feed_dict=feed_dict_train)))

    class_pred = tf.get_collection("class_pred")[0]
    pred = class_pred.eval(session=sess, feed_dict=feed_dict_train)
    #print("Pred: " + str(pred))

    class_true = tf.get_collection("class_true")[0]
    true = class_true.eval(session=sess, feed_dict=feed_dict_train)
    #print("True: " + str(true))

    #Save the slightly more trained graph
    update_model(graph_name, sess, saver)

    #Close tensorflow session
    sess.close()

    #Return the training success
    return true == pred

#Recieves a validation unit and runs prediction on it the graph on it
#Similar to using a training unit except model is not updated
def use_validation_unit(np_array,
                        width,
                        height,
                        isGalaxy,
                        rot,
                        graph_name):
    #Convert to uint8, [0, 255] is all that's needed
    np_array = np_array.astype(np.uint8)

    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    saver = restore_model(graph_name, sess)

    #Make image and conform to placeholder dimensions
    image_input = np.reshape(np_array, (1, width, height))
    #Rotate image input as informed
    if rot == 90:
        image_input = np.rot90(image_input, k=1, axes=(1, 2))
    elif rot == 180:
        image_input = np.rot90(image_input, k=2, axes=(1, 2))
    elif rot == 270:
        image_input = np.rot90(image_input, k=3, axes=(1, 2))

    #Save a copy of the graph if required
    #save_array_as_fig(image_input, "test")

    #Make labels conform to placeholder constraints
    label_array = [0, 0]                            #One hot encoding
    label_array[isGalaxy] = 1
    label_input = np.reshape(label_array, (1, 2))

    #Create a unitary feeder dictionary
    feed_dict_train = {'images:0': image_input, 'labels:0': label_input}

    #Check the training prediction
    #class_prob = tf.get_collection("class_prob")[0]
    #print("Probs:" + str(class_prob.eval(session=sess, feed_dict=feed_dict_train)))

    class_pred = tf.get_collection("class_pred")[0]
    pred = class_pred.eval(session=sess, feed_dict=feed_dict_train)
    #print("Pred: " + str(pred))

    class_true = tf.get_collection("class_true")[0]
    true = class_true.eval(session=sess, feed_dict=feed_dict_train)
    #print("True: " + str(true))

    #Close tensorflow session
    sess.close()

    #Return the training success
    return true == pred

#Recieves a unit and evaluates it using the graph
def use_evaluation_unit(np_array,
                        width,
                        height,
                        graph_name):
    #Convert to uint8, [0, 255] is all that's needed
    np_array = np_array.astype(np.uint8)

    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    saver = restore_model(graph_name, sess)

    #Make image and labels conform to placeholder dimensions
    image_input = np.reshape(np_array, (1, width, height))

    #Save a copy of image if required
    #save_array_as_fig(image_input, "test")

    #Create a unitary feeder dictionary
    feed_dict_eval = {'images:0': image_input}

    #Get the evaluation prediction. 0 -> noise, 1 -> galaxy
    class_pred = tf.get_collection("class_pred")[0]
    pred = class_pred.eval(session=sess, feed_dict=feed_dict_eval)

    #Close tensorflow session
    sess.close()

    #Return result
    return pred == 1    #True if predicted galaxy, false if predicted noise

#Loads a specified graph onto the connected NCS devices
def compile_and_load_graph_onto_ncs(graph_name):
    #Compile graph with
    #'mvNCCompile graphs/test-graph/test-graph.meta -in=images -on=fc_weights_1 -o=./graphs/test-graph/test-graph-compiled'
    print("PC LOAD LETTER")

#Recieves a unit and evaluates it using the graph on 1 or more Movidius NCS'
def use_evaluation_unit_on_ncs( np_array,
                                width,
                                height,
                                graph_name):
    print("PC LOAD LETTER")

#Plots the convolutional filter-weights for a given layer using matplotlib
def plot_conv_weights(graph_name):
    #Make sure graph structure is reset before opening session
    tf.reset_default_graph()

    #Begin a tensorflow session
    sess = tf.Session()

    #Load the graph to be trained & keep the saver for later updating
    saver = restore_model(graph_name, sess)

    #Retrieve the weights (don't need a feed dict as nothing is being calculated)
    weights = [v for v in tf.global_variables() if v.name == "conv_weights_0:0"][0]

    '''THE FOLLOWING IS TAKEN FROM HVASS LABRATORIES NOTEBOOK AT THE FOLLOWING:
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb'''
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = sess.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, 0, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='Greys_r')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig("output/weights")

    #Explicity close figure for memory usage
    plt.close(fig)


    #Close tensorflow session
    sess.close()

#Helper function for creating filters as a tensorflow variable within each layer
def new_weights(shape, var_name):
    #Weights start as noise and are eventually learned
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=var_name)

#Helper function for creating biases as a tensorflow variable within each layer
def new_biases(length, var_name):
    #Biases start as noise and are eventually learned
    return tf.Variable(tf.constant(0.05, shape=[length]), name=var_name)

#Helper function for creating a new convolution layer in a graph
def new_conv_layer(prev_layer,         #the previous layer (input to this layer)
                   num_inputs,         #number of images to be input
                   filter_size,        #width & height of each filter
                   num_filters,        #number of filters
                   conv_index):        #for naming

    #Stringify index
    conv_index = str(conv_index)

    #Create filters with a given shape to be optimised over graph execution
    #rank must be 4 for tensorflow
    weights = new_weights(shape=[filter_size, filter_size, num_inputs, num_filters], var_name=("conv_weights_" + conv_index))

    #Create biases to be optimised over graph execution
    biases = new_biases(length=num_filters, var_name=("conv_biases_" + conv_index))

    #This layer is a 2D convolution with padding of zeroes to ensure that shape
    #remains consistent during convolution
    layer = tf.nn.conv2d(input=prev_layer,
                         filter=weights,
                         strides=[1, 1, 1, 1],    #X & Y pixel strides are args 2 & 3
                         padding='SAME')

    #Add bias
    layer += biases

    #Apply pooling
    layer = tf.nn.max_pool(value=layer,         #Take the layer
                           ksize=[1, 2, 2, 1],  #2x2 max pooling over resolution
                           strides=[1, 2, 2, 1],
                           padding='SAME')      #Consistent shape

    #Apply ReLU
    layer = tf.nn.relu(layer)

    #Return the layer (and also the weights for inspection)
    return layer, weights

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
    layer_flat = tf.reshape(layer, [-1, num_features])

    #The shape of the flattened layer is now
    #[num_images, img_height * img_width * num_channels]

    #Return both the flattened layer and the number of features.
    return layer_flat, num_features

#Helper function for creating a new fully connected layer in a graph
def new_fc_layer(prev_layer,         #the previous layer (input to this layer)
                 num_inputs,         #number of images to be input
                 num_outputs,        #number to be output
                 use_ReLU,
                 fc_index):          #for unique variable naming

    #Stringify index
    fc_index = str(fc_index)

    #Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs], var_name=("fc_weights_" + fc_index))
    biases = new_biases(length=num_outputs, var_name=("fc_biases_" + fc_index))

    #Calculate the layer as the matrix multiplication of
    #the input and weights, and then add the bias-values
    layer = tf.matmul(prev_layer, weights) + biases

    #Apply ReLU
    if use_ReLU:
        layer = tf.nn.relu(layer)

    return layer

#Initialises a fresh graph and stores it for later training
def new_graph(id,             #Unique identifier for saving the graph
              filter_sizes,   #Filter dims for each convolutional layer (kernals)
              num_filters,    #Number of filters for each convolutional layer
              fc_size):       #Number of neurons in fully connected layer

    #Create computational graph to represent the neural net:
    print("Creating new graph: '" + id + "'")
    print("\t*Structure details:")

    #INPUT
    #Placeholders serve as variable input to the graph (can be changed when run)
    #Following placeholder takes single freqeuncy components as tensors
    #(must be float32 for convolution and must be 4D for tensorflow)
    images = tf.placeholder(tf.float32, shape=[None, 100, 100], name='images')
    images_4d = tf.reshape(images, shape=[-1, 100, 100, 1])
    print("\t\t-Placeholder '" + images.name + "': " + str(images))

    #and supervisory signals which are true/false (galaxy or not)
    labels = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
    print("\t\t-Placeholder '" + labels.name + "': " + str(labels))

    #for later comparison
    class_true = tf.argmax(labels, axis=1, name='class_true')
    tf.add_to_collection("class_true", class_true)

    #CONVOLUTIONAL LAYERS
    #These convolutional layers sequentially take inputs and create/apply filters
    #to these inputs to create outputs. They create weights and biases that will
    #be optimised during graph execution. They also down-sample (pool) the image
    #after doing so. Filters are created in accordance with the arguments to this
    #function
    layer_conv0, weights_conv0 = new_conv_layer(prev_layer=images_4d,
                                                num_inputs=1,
                                                filter_size=filter_sizes[0],
                                                num_filters=num_filters[0],
                                                conv_index=0)
    print("\t\t-Convolutional 0: " + str(layer_conv0))

    #layer 2 takes layer 1's output
    layer_conv1, weights_conv1 = new_conv_layer(prev_layer=layer_conv0,
                                                num_inputs=num_filters[0],
                                                filter_size=filter_sizes[1],
                                                num_filters=num_filters[1],
                                                conv_index=1)
    print("\t\t-Convolutional 1: " + str(layer_conv1))

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
                             num_outputs=fc_size,
                             use_ReLU=True,
                             fc_index=0)
    print("\t\t-Fully connected 0: " + str(layer_fc0))

    layer_fc1 = new_fc_layer(prev_layer=layer_fc0,
                             num_inputs=fc_size,
                             num_outputs=2,
                             use_ReLU=False,
                             fc_index=1)
    print("\t\t-Fully connected 1: " + str(layer_fc1))

    #PREDICTION DETAILS
    #Final fully connected layer suggests prediction (these structures are added to
    #collections for ease of access later on)
    print("\t*Prediction details:")

    #Normalise it to get a probability
    class_prob = tf.nn.softmax(layer_fc1)
    print("\t\t-Class probabilities: " + str(class_prob))
    tf.add_to_collection("class_prob", class_prob)

    #Greatest probability of the classes (not a galaxy, is a galaxy) is the prediction
    class_pred = tf.argmax(class_prob, axis=1)
    print("\t\t-Class prediction: " + str(class_pred))
    tf.add_to_collection("class_pred", class_pred)

    #Backpropogation details
    print("\t*Backpropagation details:")

    #COST FUNCTION
    #Cost function is cross entropy (+ve and approaches zero as the model output
    #approaches the desired output
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc1, labels=labels)
    cost = tf.reduce_mean(cross_entropy)
    print("\t\t-Cost function: " + str(cross_entropy))

    #OPTIMISATION FUNCTION
    #Optimisation function to Optimise cross entropy will be Adam optimizer
    #(advanced gradient descent)
    alpha = 1e-4
    optimiser = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)
    print("\t\t-Optimiser: alpha=" + str(alpha) + ", " + str(optimiser.name))


#Wraps the above to make a very basic convolutional neural network
def new_basic_graph(id):
    #Create a graph
    new_graph(id,      #Id
              filter_sizes=[5, 5],  #Convolutional layer filter sizes in pixels
              num_filters=[16, 36], #Number of filters in each Convolutional layer
              fc_size=128)          #Number of neurons in fully connected layer

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
