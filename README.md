# gfinder
gfinder is a program that facilitates the training, validation and evaluation of convolutional neural networks using JPEG2000 formatted data for the purpose of finding faint galaxies in wavelet transformed data. Parallelisation using Intel Movidius Neural Compute Sticks (NCS) is supported to speed up the inferencing process.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites
* C++ (ISO/IEC 9899:1999 or greater)
* Python3 (3.5.2 or greater)
* [Kakadu SDK](http://kakadusoftware.com/) (v7_8-01265L or greater)
* [NCSDK](https://github.com/Movidius/ncsdk) (NCSDK 1.12.00 or greater)
* Ubuntu (16.04 LTS or greater)

### Installing
1. Clone this repository to your local machine:
```
git clone https://github.com/Isaac-Ronald-Ward/gfinder
```
2. Enter the directory:
```
cd gfinder
```
3. Copy your compiled version of the Kakadu SDK into the 'libs' folder. If for some reason this step cannot be completed, see step 4.
```
cp /absolute/path/to/compiled/Kakadu/SDK ./libs
```
4. Edit the supplied Makefile on line 34 to point to the location of Python3 on your local machine (this is required for the embedded Python3 operations the program uses). Line 10 can also be edited to point to the compiled Kakadu SDK in the case that step 3 could not be completed. For example:
```
KDU_PATH = /absolute/path/to/compiled/Kakadu/SDK
...
PY_PATH = /usr/include/python3.5m
```
5. Make and attempt to display the program's usage string:
```
make && ./gfinder -u
```
Expected output:
```
Trains, validates and evaluates convolutional neural networks for the purpose of finding faint galaxies in JPEG2000 formatted SIDCs.
Arguments:
	-c,	the component range (inclusive) to use: '-c start_component_index,final_component_index'
	-d,	the discard levels (DWT) that should be applied to input (default 0): '-d discard_level'. Use the '-m' argument to print the available discard levels in the file (Note that this will decrease the output image's width and height each by a factor of 2^discard_level)
	-e,	whether or not to evaluate the input using a given graph and the region to evaluate: '-e x,w,y,h'
	-f,	the input file to use: '-f filepath'
	-g,	the name of the graph to use: '-g graph_name'
	-h,	prints help message
	-m,	prints more information about input JPEG2000 formatted data
	-n,	whether or not to evaluate the input using attached Intel Movidius Neural Compute Sticks
	-o,	the region of the input file that should be output as a low quality PNG: '-o x,w,y,h'
	-p,	the port to stream data from C++ decompressor to Python3 graph manipulator on (usually 10000 or greater): '-p port_number'
	-q,	the quality level to limit decompressing to (default maximum): '-q quality_level'. Use the '-m' argument to print the available quality levels in the file (Note that this does not affect the output image's dimensions, only its appearance)
	-t,	whether or not to train on the supplied input file
	-u,	prints usage statement
	-v,	whether or not to validate supplied graph's unit inferencing capabilities
	-x,	the filepath to an evaluation result that should be cross checked for differences with actual galaxy locations in input file: '-x filepath' (specifying this parameter will scan the entire input file's metadata tree, regardless of component range arguments supplied to gfinder. (Ensure that the resolution level used to generate the supplied evaluation result is matched))
```

## Pretrained graphs
'gfinder' comes with 9 pretrained graphs that have been trained on components 0-799 of each dingo.XX000.jpx file at varying quality layers. They are named accordingly and can be found in the 'graphs' folder. Creating a new graph with the same name as these graphs will overwrite them.

## Usage
The three main functions of gfinder are training, validating and evaluating a convolutional neural network on JPEG2000 formatted data.
* Begin by creating a new training graph called test graph (note that this script will overwrite CNNs with the same name):
```
python3 new_training_graph.py test-graph
```
* gfinder can now use this graph (defaultly saved to the 'graphs' folder) for training, an operation which will output training statistics to the 'output' folder:
```
./gfinder -f /data/dingo.00000.with_catalogue.jpx -g test-graph -t -r 0 -c 0,799 -p 10000
```
* the trained graph can then be used for validation (note that the components used for validation are independant to the components used for training and evaluation):
```
./gfinder -f /data/dingo.00000.with_catalogue.jpx -g test-graph -v -r 0 -c 800,899 -p 10000
```
* a trained graph can also evaluate a supplied region in a given .jpx file:
```
./gfinder -f /data/dingo.00000.with_catalogue.jpx -g test-graph -e 0,0,1800,1800 -r 0 -c 994,994 -p 10000
```
* results will be printed to console and outputted to the 'output' and 'results' folder. These results can be checked for difference against the original file to ensure that the found galaxies exist in the original file within a given seperation through components (galaxies may be visible across multiple components but are only labelled in the component in which they are most intense):
```
./gfinder -f /data/dingo.00000.with_catalogue.jpx -g test-graph -d ./results/file-dingo.00000.with_catalogue.jpx_comp-994-994_locs-0-0-1800-1800.dat
```
* For further explanation of command line arguments see the program's usage string:
```
./gfinder -u
```
* The evaluation process can be sped up if one or more NCS' are connected to the local machine using a powered USB hub if the '-n' argument is provided. It is recommended that a terminal window is opened with the command 'dmesg -w' running as the NCS' are connected to ensure a successful connection.
* In general, creating and training a new CNN involves running the new_training_graph script

## Advanced usage
The structure of the CNN that gfinder will create/train/validate/evaluate can be altered by changing the source code in 'cnn.py' and 'src/main.cpp'. In cnn.py, lines 43 - 55 can be changed to alter the CNN's:
* input dimensions (note any changes to this parameter must be reflected in the main.cpp file on lines 57 and 58 - remember to recompile with 'make'):
```
INPUT_WIDTH       = ...
INPUT_HEIGHT      = ...
```
* number of convolutional layers (and the number and size of filters in them - note that the length of 'FILTER_SIZES' and 'NUM_FILTERS' must match!):
```
FILTER_SIZES      = [...]
NUM_FILTERS       = [...]
```
* number of fully connected layers (and the number of neurons in them):
```
FC_SIZES          =   [...]
```
* filter initialisation bias:
```
BIAS_STD_DEV_INIT = ...
```

More advanced changes can be made by altering the 'new_graph' function but this is only recommended for users with an understanding of the TensorFlow framework.

Once changes are made, the gfinder program will be unable to load CNNs created with different parameters, as the NCS doens't support loading CNN structures from TensorFlow's '.meta' files, so the structure of the CNN must be generated in the program before restoring the CNN's weights (It is recommended that the user keep track of the parameters used by each CNN if they choose to alter them).

## License
TODO

##### Author
[**Isaac Ronald Ward**](https://github.com/Isaac-Ronald-Ward), contact: isaacronaldward@gmail.com

##### Acknowledgments
* [**Dr. Slava Kitaeff**](https://github.com/skitaeff) (KDU compilation, Skuareview compilation, supervision)
* [**Jurek Tadek Malarecki**](https://github.com/jtmalarecki) (JPEG2000 data formatting, JPEG2000 metadata embedding)
* [**Movidius**](https://github.com/Movidius/ncsdk)
* [**TensorFlow**](https://github.com/tensorflow/tensorflow)
* [**Kakadu**](http://kakadusoftware.com/)
* [**International Centre for Radio Astronomy Research**](https://www.icrar.org/)
