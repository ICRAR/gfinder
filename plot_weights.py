from cnn import *

#Main function
if __name__ == "__main__":
    #Plot both conv layer filters
    plot_conv_weights("test-graph", "conv_", 0, 7, sys.argv[1])
