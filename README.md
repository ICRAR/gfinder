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
	-d,	the filepath to an evaluation result that should be checked for differences with actual galaxy locations in input file: '-d filepath' (specifying this parameter will scan the entire input file's metadata tree, regardless of component range arguments supplied to gfinder. (Ensure that the resolution level used to generate the supplied evaluation result is matched)
	-e,	whether or not to evaluate the input using a given graph and the region to evaluate: '-e x,w,y,h'
	-f,	the input file to use: '-f filepath'
	-g,	the name of the graph to use: '-g graph_name'
	-h,	prints help message
	-m,	prints more information about input JPEG2000 formatted data
	-n,	whether or not to evaluate the input using attached Intel Movidius Neural Compute Sticks
	-p,	the port to stream data from C++ decompressor to Python3 graph manipulator on (usually 10000 or greater): '-p port_number'
	-r,	the resolution level (DWT) to use the input at (default 0): '-r resolution_level'
	-t,	whether or not to train on the supplied input file
	-u,	prints usage statement
	-v,	whether or not to validate supplied graph's unit inferencing capabilities
```

## Usage
The three main functions of gfinder are training, validating and evaluating a convolutional neural network on JPEG2000 formatted data.
* Begin by creating a new training graph called test graph:
```
python3 new_training_graph test-graph
```
* gfinder can now use this graph (defaultly saved to the 'graphs' folder) for training, an operation which will output training statistics to the 'output' folder:
```
./gfinder -f /data/dingo.00000.with_catalogue.jpx -g test-graph -t -r 0 -c 0,799 -p 10000
```
* the trained graph can then be used for validation:
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

## License
TODO

##### Author
[**Isaac Ronald Ward**](https://github.com/Isaac-Ronald-Ward)

##### Acknowledgments
* Dr Slava Kitaeff (KDU compilation, Skuareview compilation, supervision)
* JT Malarecki (JPEG2000 data formatting, JPEG2000 metadata embedding)
* International Centre for Radio Astronomy Research
