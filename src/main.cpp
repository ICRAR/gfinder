//Compile with makefile provided

//For example:
//  ./gfinder -f /media/izy/irw_ntfs_0/dingo.00000.with_catalogue.jpx -g test-graph -t -r 0 -c 0,799 -p 10000
//  ./gfinder -f /media/izy/irw_ntfs_0/dingo.00000.with_catalogue.jpx -g test-graph -v -r 0 -c 800,899 -p 10000
//  ./gfinder -f /media/izy/irw_ntfs_0/dingo.00000.with_catalogue.jpx -g test-graph -e 28,1489,400,400 -r 0 -c 994,994 -p 10000

//C++ standard includes
#include <iostream>     //For cout
#include <sstream>      //For parsing command line arguments
#include <string>       //For xtoy conversions
#include <algorithm>    //For min
#include <math.h>       //For ceil, pow
#include <stdlib.h>     //For atoi and atof
#include <vector>       //For vectors
#include <unistd.h>     //For getopt

//IPC/networking & process management includes
#include <sys/types.h>
#include <sys/socket.h> //For sockets
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>   //For wait
#include <errno.h>      //Gonna need it ...

//KDU includes
#include "jpx.h"
#include "kdu_messaging.h"
#include "kdu_utils.h"
#include "kdu_region_decompressor.h"

//Python includes
//Suppresses warnings when loading numpy array objects
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>               //For embedded python (must be included after KDU)
#include <numpy/arrayobject.h>    //For creating numpy arrays in C++

//Namespaces (get specifics so as to not pollute)
using namespace kdu_supp; //Also includes the `kdu_core' namespace

//Don't want to use namespace due to fear of collisions but don't want to write
//'std::' everywhere
using std::cout;
using std::min;
using std::max;
using std::vector;

//Input sizes of images (in pixels) to be fed to graph
//Must reflect changes in Python file's globals
const int INPUT_WIDTH = 32;
const int INPUT_HEIGHT = 32;

//Number of images to feed per batch (minibatch)
const int BATCH_SIZE = 1;

//Global command line variables, use getopt to get the following arguments:
char *JPX_FILEPATH    = NULL;
bool IS_TRAIN         = false;
bool IS_VALIDATE      = false;
bool IS_EVALUATE      = false;
char *GRAPH_NAME      = NULL;
int START_COMPONENT_INDEX = 0;
int FINAL_COMPONENT_INDEX = 0;
int RESOLUTION_LEVEL  = -1;
int LIMIT_RECT_X      = 0;
int LIMIT_RECT_Y      = 0;
int LIMIT_RECT_W      = 0;
int LIMIT_RECT_H      = 0;
int PORT_NO           = -1;
bool NCS_EVALUATION   = false;


//----------------------------------------------------------------------------//
// Set up KDU messaging                                                       //
//----------------------------------------------------------------------------//
class kdu_stream_message : public kdu_message {
  public: // Member classes
    kdu_stream_message(FILE *dest)
      { this->dest = dest; }
    void start_message()
      { fprintf(dest,"-------------\n"); }
    void put_text(const char *string)
      { fprintf(dest,"%s",string); }
    void flush(bool end_of_message=false)
      { fflush(dest); }
  private: // Data
    FILE *dest;
};

static kdu_stream_message cout_message(stdout);
static kdu_stream_message cerr_message(stderr);
static kdu_message_formatter pretty_cout(&cout_message);
static kdu_message_formatter pretty_cerr(&cerr_message);

//----------------------------------------------------------------------------//
// Internal structs                                                           //
//----------------------------------------------------------------------------//

struct label{
  int tlx; //RA or x-axis of top left vertex of rectangle
  int tly; //DEC or y-axis of top left vertex of rectangle

  int brx; //RA or x-axis of bottom right vertex of rectangle
  int bry; //DEC or y-axis of bottom right vertex of rectangle

  int f; //FREQ or z-axis coordinate of rectangle

  bool isGalaxy;  //Does this label represent a galaxy
};

//----------------------------------------------------------------------------//
// Internal functions                                                         //
//----------------------------------------------------------------------------//

//Loads label data from the ROI container in a jpx file. Note the jpx source
//is taken as a refernce - this prevents a segmentation fault later when using
//the jpx source with kakadu tools
void load_labels_from_roid_container( jpx_source & jpx_src,
                                      vector<label> & labels)
{
  //Get a reference to the meta manager
  jpx_meta_manager meta_manager = jpx_src.access_meta_manager();
  if(!meta_manager.exists()){
    return;  //If file didn't have meta_manager then return
  }

  //Iterate over flattened metadata tree from the root node
  jpx_metanode root = meta_manager.access_root();

  //Try to find all nlst nodes (they will have roid nodes as their direct children)
  jpx_metanode child;
  do{
    //Get the label boxes
    child = root.get_next_descendant(child);
    if(!child.exists()){
      //Found our last child last iteration; continue into while loop termination
      continue;
    }

    //Throw away the uuid boxes, keep the lbl_ boxes (they will have lbl_, nlst
    //and roid children)
    switch(child.get_box_type()){
      case jp2_uuid_4cc:
      {
        //Skip uuid boxes
        continue;
        break;
      }

      case jp2_label_4cc:
      {
        //From a child of the metatree root the order is as follows
        //child->lbl_->nlst->roid (we need data out of the roid)

        //Get lbl_, nlst & roid boxes out of child lbl_ boxes
        //use get descendant rather than next descendant which will
        //skip every second descendant of the child (no idea why)
        jpx_metanode lbl_ = child.get_descendant(0);
        jpx_metanode nlst = lbl_.get_descendant(0);
        jpx_metanode roid = nlst.get_descendant(0);

        //Ensure roid box attained correctly
        if(roid.get_box_type() == jp2_roi_description_4cc){
          //If roid box obtained correctly then obtain single roi from it
          jpx_roi roi = roid.get_region(0);

          //Extract the quadrilateral represented by this roi
          kdu_coords v1, v2, v3, v4;  //v1 is top left, then clockwise
          roi.get_quadrilateral(v1, v2, v3, v4);

          //Begin constructing a supervision label for this discrete 3d spectral
          //imaging datacube
          label l;

          //Simple rectangles can be represented using top left (v1) and bottom
          //right (v3) vertices. Note that the bounding box for the galaxy labels
          //is VERY large (100x100 pixels) so is shrunk to 30x30 pixels for
          //decompression
          l.tlx = v1.get_x() + (100 - INPUT_WIDTH)/2;
          l.tly = v1.get_y() + (100 - INPUT_HEIGHT)/2;
          l.brx = v3.get_x() - (100 - INPUT_WIDTH)/2;
          l.bry = v3.get_y() - (100 - INPUT_HEIGHT)/2;

          //Also need the frequency location to triangulate the roi, which can
          //be found in nlst
          l.f = nlst.get_numlist_layer(0);

          //If a label is being read from the roid box then it holds a galaxy
          l.isGalaxy = true;

          //Only acknowledge if within component range (inclusive)
          if(l.f >= START_COMPONENT_INDEX && l.f <= FINAL_COMPONENT_INDEX){
            //"Your labels will make a fine addition to my ... collection"
            labels.push_back(l);

            /*
            //Also push back slightly translated labels (<10px each way for better
            //generalisation)
            int translation = 4;
            int x;
            int y;
            for(int i = 0; i < 4; i++){
              if(i == 0){
                x = -1; y = 0;
              }else if(i == 1){
                x = 1;  y = 0;
              }else if(i == 2){
                x = 0;  y = -1;
              }else if(i == 3){
                x = 0;  y = 1;
              }

              //Goes through four combinations (-1,0), (1,0), (0,-1), (0,1)
              label translated_l;
              translated_l.tlx  = l.tlx + translation*x;
              translated_l.tly  = l.tly + translation*y;
              translated_l.brx  = l.brx + translation*x;
              translated_l.bry  = l.bry + translation*y;
              translated_l.f    = l.f;
              translated_l.isGalaxy = l.isGalaxy;
              labels.push_back(translated_l);
            }
            */
          }
        }
        //Case closed
        break;
      }

      default:
      {
        cout << "Error: metanode child is neither type uuid or lbl_, skipping\n";
        break;
      }
    }
  }while(child.exists());
}

//A helper that checks if two labels (typically an existing galaxy and a generated
//noise label are overlapping)
bool labels_intersect(label a, label b){
  //tlx = left
  //brx = right
  //tly = top
  //bry = bottom
  return !( a.tlx > b.brx ||
            a.brx < b.tlx ||
            a.tly > b.bry ||
            a.bry < b.tly);
}

//Creates a set of false labels
void generate_false_labels( vector<label> & labels){
  //Range of components
  int range = FINAL_COMPONENT_INDEX - START_COMPONENT_INDEX + 1; //+1 because inclusive
  //Required number of noise labels to be found
  int req = labels.size();
  //The width and height of galaxy labels
  int w = labels[0].brx - labels[0].tlx;
  int h = labels[0].bry - labels[0].tly;

  //If not a single free component was found then return
  if(range == 0){
    cout << "Error: cannot find noise labels over a component range of zero\n";
  }else{
    //Calculate the number of noise labels to be found per component
    double labels_per_component = ceil((double)req/(double)range);

    //Go over every component and generate the required labels randomly from
    //areas that don't have a galaxy in them
    int noise_labels_generated = 0;
    for(int i = START_COMPONENT_INDEX; i <= FINAL_COMPONENT_INDEX; i++){
      //If enough galaxies have been found then finish
      if(noise_labels_generated >= req){
        break;
      }

      //Find all true labels in this component
      vector<label> galaxies;
      for(int j = 0; j < labels.size(); j++){
        if(labels[j].f == i){
          galaxies.push_back(labels[j]);
        }
      }

      //A random number generator will be required
      srand(0); //Seed

      //In each component i, find the correct number of labels (calculated earlier)
      //ensuring that they don't intersect with the galaxies in this label
      for(int j = 0; j < labels_per_component; j++){
        //TODO: don't do trial and error method and scutinuse overlap more carefully,
        //perhaps within a component tolerance (galaxies bleed through components).
        //Sort galaxies vector to reduce from O(n^2 + n)
        //to O(nlogn + n). Don't hardcode 3600x3600 resolution

        //Generate a random noise label
        label noise;
        noise.tlx       = rand()%3599;
        noise.tly       = rand()%3599;
        noise.brx       = noise.tlx + w;
        noise.bry       = noise.tly + h;
        noise.isGalaxy  = false;
        noise.f         = i;

        //Does the noise label overlap with any galaxy label? If so then retry
        bool overlap = false;
        for(int k = 0; k < galaxies.size(); k++){
          if(labels_intersect(galaxies[k], noise)){
            overlap = true;
            break;
          }
        }

        //If there was no overlap then add, otherwise try again
        if(overlap){
          j--;
        }else{
          labels.push_back(noise);
          noise_labels_generated++;
        }
      }
    }
  }
}

//Prints various codestream statistics
void print_statistics(kdu_codestream codestream){
  //Announce various statistics
  cout << "File statistics:\n";

  //Print dwt levels
  cout << "\t-Minimum DWT   : " << codestream.get_min_dwt_levels() << " (levels)\n";

  //Print tiling
  kdu_dims tiles;
  codestream.get_valid_tiles(tiles);
  cout << "\t-Tiling        : " << tiles.access_size()->get_x() << "x" << tiles.access_size()->get_y() << " (tiles)\n";

  //Print number of components in image
  cout << "\t-Components    : " << codestream.get_num_components() << "\n";

  //Print dimensions
  kdu_dims dims;
  codestream.get_dims(0, dims, false);
  cout << "\t-Dimensions[0] : " << dims.access_size()->get_x() << "x" << dims.access_size()->get_y() << " (px)\n";

  //Print area
  cout << "\t-Area[0]       : " << dims.area() << " (px)\n";

  //Print bit-depth
  cout << "\t-Bit-depth[0]  : " << codestream.get_bit_depth(0, false, false) << "\n";

  //Print codestream comments if they exist
  kdu_codestream_comment comment;
  comment = codestream.get_comment(comment);
  int count = 0;
  while(comment.exists()){
    cout << "\t-Comment[" << count++ << "]    : " << comment.get_text() << "\n";
    comment = codestream.get_comment(comment);
  }
}

//Directly writes to low level structures in the file's codestream in order
//to edit the appearance of the decoded image with respect to frequency
void apply_frequency_limits(kdu_codestream codestream){
  //From the codestream, access the tile (always one codestream & one tile)
  kdu_dims main_tile_indices; main_tile_indices.size.x = main_tile_indices.size.y = 1;
  codestream.open_tiles(main_tile_indices, false, NULL);
  kdu_tile main_tile = codestream.access_tile(kdu_coords(0, 0), true, NULL);

  //From the tile, access a tile component
  int num_components_in_tile  = main_tile.get_num_components();
  for(int c = 450; c < 451; c++){    //1000 components  //TODO
    cout << "In component " << c << "\n";
    kdu_tile_comp tile_comp     = main_tile.access_component(c);

    //From a tile component, access all resolutions
    int num_resolutions_in_comp = tile_comp.get_num_resolutions();
    for(int r = 1; r < num_resolutions_in_comp; r++){ //6 resolutions
      cout << "In resolution layer " << r << "\n";

      //Get's the rth resolution level
      kdu_resolution res          = tile_comp.access_resolution(r);

      //From a resolution, access a subband
      //min_idx will hold 0 if lowest resolution layer, else 1 after call to res
      int min_idx = -1;
      //1 if lowest res, otherwise higher res integers returned
      int num_subbands_in_res     = res.get_valid_band_indices(min_idx);
      //min_idx should be 1 at this point

      //Iterate over all valid subband indices
      for(int s = 1; s <= num_subbands_in_res; s++){   //3 subbands
        //Open this iteration's subband
        kdu_subband subband         = res.access_subband(s);

        //From a subband, open all code blocks
        kdu_dims valid_code_blocks;  subband.get_valid_blocks(valid_code_blocks);
        for(int i = 0; i < valid_code_blocks.size.x; i++){
          for(int j = 0; j < valid_code_blocks.size.y; j++){  //57x57 blocks
            //Open block
            kdu_block *code_block   = subband.open_block(kdu_coords(i, j));

            //From a code block, edit attributes that will affect
            //following decompression
            //recall orientation member defines band level:
              //LL_BAND = 0
              //HL_BAND = 1
              //LH_BAND = 2
              //HH_BAND = 3

            /*
            //cout << code_block->orientation;
            //cout << code_block->num_passes << "\n";
            code_block->set_num_passes(1);

            cout << (code_block->vflip ? "T" : "F") << "\n";
            code_block->vflip = true;
            cout << (code_block->vflip ? "T" : "F") << "\n";

            //cout << code_block->num_passes << "\n";
            */

            //Close code block
            subband.close_block(code_block);
          }
        }
      }
    }
  }

  //Close tiles (all other objects were accessed, so don't require closing)
  codestream.close_tiles(main_tile_indices, NULL);
}

//Initialises interpreter and numpy arrays for passing blocks to python
void* init_embedded_python(){
  //Initialises interpreter
  Py_Initialize();

  //So modules can be found
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");

  //Init numpy arrays
  import_array();

  return (void*) 1;
}

//Finalises embeeded python (closing interpreter)
void end_embedded_python(){
  Py_Finalize();
}

//Called when training a graph is specified. Note a reference to the jpx source
//is taken - this prevents segmentation fault (same with codestream)
void train(vector<label> labels, kdu_codestream codestream, kdu_thread_env & env){

  //Set up a sockets
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if(sockfd < 0){
    cout << "Error: couldn't establish socket for server in C++ process\n";
    return;
  }

  //Set up server using socket
  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = htons(INADDR_ANY);
  server_addr.sin_port = htons(PORT_NO);

  //Bind server to socket
  if((bind(sockfd, (struct sockaddr*)&server_addr,sizeof(server_addr))) < 0){
    cout << "Error: couldn't bind connection to already established socket\n";
    return;
  }

  //Fork the python training program (loads up the graph)
  int status;
  pid_t pid = fork();
  if(pid == 0){
    //Child, create the python training process here (requires graph name)
    //Get filename
    PyObject* py_name   = PyUnicode_FromString((char*)"cnn");
    PyErr_Print();

    //Import file as module
    PyObject* py_module = PyImport_Import(py_name);
    PyErr_Print();

    //Get function name from module depending on if validating or training
    PyObject* py_func;
    py_func = PyObject_GetAttrString(py_module, (char*)"run_training_client");
    PyErr_Print();

    //Call function with the graph to train on and the port to listen for
    //training data on
    PyObject_CallObject(py_func, Py_BuildValue("(s, i, i, i, i)",
      GRAPH_NAME,
      PORT_NO,
      (IS_TRAIN ? 1 : 0),  //Whether or not to update graph
      BATCH_SIZE,
      labels.size()
    ));
    PyErr_Print();

    //Child process now finished, destroy
    exit(EXIT_SUCCESS);

  }else if (pid > 0){
    //Parent, wait for 1 python client to connect here
    listen(sockfd, 1);
    socklen_t size = sizeof(server_addr);
    int server = accept(sockfd, (struct sockaddr *)&server_addr, &size);
    if(server < 0){
      cout << "Error: couldn't accept client connection\n";
    }else{
      cout << "Training client successfully connected, begin streaming decompressed data\n";
    }

    //Training data is currently a block of positives followed by a block of
    //negatives. It should be random
    std::random_shuffle(labels.begin(), labels.end());

    //Track the number of training units fed thus far and the number expected to be fed
    int units_expected = labels.size();

    //Decompress areas given by labels with a given tolerance (labels mark only the
    //frequency point at which the galaxy is strongest, but typically they are still
    //in previous and further frequency frames)
    for(int l = 0; l < labels.size(); l++){
      //Construct a region from the label data
      kdu_dims region;
      region.access_pos()->set_x(labels[l].tlx);
      region.access_size()->set_x(labels[l].brx - labels[l].tlx);
      region.access_pos()->set_y(labels[l].tly);
      region.access_size()->set_y(labels[l].bry - labels[l].tly);

      //Decompress over labeled frames at the correct
      //spacial coordinates using kakadu decompressor
      kdu_region_decompressor decompressor;

      int component_index = labels[l].f;

      //TODO: variable
      int discard_levels = 0;

      //Get safe expansion factors for the decompressor
      //Safe upper bounds & minmum product returned into the following variables
      double min_prod;
      double max_x;
      double max_y;
      decompressor.get_safe_expansion_factors(
        codestream, //The codestream being decompressed
        NULL,  //Codestream's channel mapping (null because specifying index)
        component_index,    //Check over all components
        discard_levels,     //DWT levels to discard (resolution reduction)
        min_prod,
        max_x,
        max_y
      );

      kdu_dims component_dims;  //Holds expanded component coordinates (post discard)
      kdu_dims incomplete_region;  //Holds the region that is incomplete after processing run
      kdu_dims new_region;  //Holds the region that is rendered after a processing run

      //Get the expansion factors for expansion TODO: non 1x expansion
      kdu_coords scale_num;
      scale_num.set_x(1); scale_num.set_y(1);
      kdu_coords scale_den;
      scale_den.set_x(1); scale_den.set_y(1);

      //Get the size of the complete component (after discard levels decrease
      //in resolution is applied) on the rendering canvas
      component_dims = decompressor.get_rendered_image_dims(
        codestream, //The codestream being decompressed
        NULL,       //Codestream's channel mapping (null because specifying index)
        component_index,  //Component being decompressed
        discard_levels,   //DWT levels to discard (resolution reduction)
        scale_num,
        scale_den
      );

      //Should a region not fully be included in the image then skip to the next label
      if( region.pos.x < component_dims.pos.x ||
          region.pos.x + region.size.x > component_dims.pos.x + component_dims.size.x ||
          region.pos.y < component_dims.pos.y ||
          region.pos.y + region.size.y > component_dims.pos.y + component_dims.size.y){
            continue;
      }

      //Create a buffer to send the data across into
      int bufsize = INPUT_WIDTH*INPUT_HEIGHT;
      kdu_uint32 buffer[bufsize];

      //Loop decompression to ensure that amount of DWT discard levels doesn't
      //exceed tile with minimum DWT levels
      do{
        //Set up decompressing run
        //cout << "Starting decompression\n";
        decompressor.start(
          codestream, //The codestream being decompressed
          NULL,  //Codestream's channel mapping (null because specifying index)
          component_index,  //Component being decompressed
          discard_levels,   //DWT levels to discard (resolution reduction)
          INT_MAX,    //Max quality layers
          region,      //Region to decompress
          scale_num,  //Expansion factors
          scale_den,
          &env        //Multi-threading environment
        );

        //Decompress until buffer is filled (block is fully decompressed)
        //cout << "Processing\n";
        incomplete_region = component_dims;
        while(
          decompressor.process(     //Buffer to write into:
            (kdu_int32 *) buffer,
            region.pos,             //Buffer origin
            region.size.x,          //Row gap
            256000,                 //Suggesed increment
            0,                      //Max pixels in region
            incomplete_region,
            new_region
          )
        );
        //Finalise decompressing run
        //cout << "Finishing decompression\n";
        decompressor.finish();

      //Render until there is no incomplete region remaining
      }while(!incomplete_region.is_empty());

      //Send the now complete image data
      int image_transmitted = send(server, &buffer, sizeof(buffer), 0);
      if(image_transmitted != bufsize*4){
        cout << "Error: image " << l << " was not sent completely over socket, "
          << "len=" << image_transmitted << "\n";
        return;
      }

      //And the label (1 -> galaxy, 0 -> noise)
      kdu_uint32 label_int = (labels[l].isGalaxy ? 1 : 0);
      int label_transmitted = send(server, &label_int, sizeof(label_int), 0);
      if(label_transmitted != 1*4){
        cout << "Error: label " << l << " was not sent completely over socket, "
          << "len=" << label_transmitted << "\n";
        return;
      }
    }

    //Close the socket in the parent and prevent further transmissions. The
    //python process can still recv though
    if(shutdown(sockfd, SHUT_WR) != 0){
      cout << "Error, couldn't shutdown socket in parent\n";
    }

    //Wait for death of child
    while(wait(&status) != pid);

  }else{
    //Fork failure
    cout << "Error: 'fork()' failed when creating python training process\n";
    exit(EXIT_FAILURE);
  }
}

//Called when evaluating an image is specified
void evaluate(kdu_codestream codestream, kdu_thread_env & env){
  //Quick error check, not point traversing region too small for window
  if(LIMIT_RECT_W < INPUT_WIDTH || LIMIT_RECT_H < INPUT_HEIGHT){
    cout << "Error: defined region ("
      << LIMIT_RECT_W << "x" << LIMIT_RECT_H
      << ") is smaller than evaluation sliding window "
      << "of dimensions (" << INPUT_WIDTH << "x" << INPUT_HEIGHT << ")\n";
    return;
  }

  //To evaluate, a sliding window over the image at the required components is
  //used to evalate on every possible region of the input image/component. Each
  //possible region is fed to the neural network graph to test if it holds a
  //galaxy. The result is crunched to find the area that most likely holds
  //a galaxy based on the results for nearby regions
  int stride_x = 8;
  int stride_y = 8;

  //For tracking progress
  int steps_x = floor((LIMIT_RECT_W - INPUT_WIDTH)/stride_x) + 1;
  int steps_y = floor((LIMIT_RECT_H - INPUT_HEIGHT)/stride_y) + 1;
  int evaluation_units_fed = 0;
  int units_per_component = steps_x*steps_y;

  //Set up a sockets
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if(sockfd < 0){
    cout << "Error: couldn't establish socket for server in C++ process\n";
    return;
  }

  //Set up server using socket
  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = htons(INADDR_ANY);
  server_addr.sin_port = htons(PORT_NO);

  //Bind server to socket
  if((bind(sockfd, (struct sockaddr*)&server_addr,sizeof(server_addr))) < 0){
    cout << "Error: couldn't bind connection to already established socket\n";
    return;
  }

  //Fork the python evaluation program
  int status;
  pid_t pid = fork();
  if(pid == 0){
    //Child, create the python training process here (requires graph name)
    //Get filename
    PyObject* py_name   = PyUnicode_FromString((char*)"cnn");
    PyErr_Print();

    //Import file as module
    PyObject* py_module = PyImport_Import(py_name);
    PyErr_Print();

    //Get function name from module depending on if validating or training
    PyObject* py_func;
    if(NCS_EVALUATION){
      py_func = PyObject_GetAttrString(py_module, (char*)"run_evaluation_client_for_ncs");
    }else{
      py_func = PyObject_GetAttrString(py_module, (char*)"run_evaluation_client_for_cpu");
    }
    PyErr_Print();

    //Call function with the graph to train on and the port to listen for
    //training data on
    PyObject_CallObject(py_func, Py_BuildValue("(s, i, i, i, i, i, i, i, i)",
      GRAPH_NAME,
      PORT_NO,
      LIMIT_RECT_W,
      steps_x,
      stride_x,
      LIMIT_RECT_H,
      steps_y,
      stride_y,
      (FINAL_COMPONENT_INDEX - START_COMPONENT_INDEX + 1)
    ));
    PyErr_Print();

    //Child process now finished, destroy
    exit(EXIT_SUCCESS);

  }else if (pid > 0){
    //Parent, wait for some python client to connect here
    int clients_required = 1;
    listen(sockfd, clients_required);
    socklen_t size = sizeof(server_addr);
    int server = accept(sockfd, (struct sockaddr *)&server_addr, &size);
    if(server < 0){
      cout << "Error: couldn't accept client connection\n";
    }else{
      cout << clients_required << " evaluation client(s) successfully connected,"
        << "begin streaming decompressed data\n";
    }

    //Announce plan
    cout << "Image will be fed in " << units_per_component
      << " regions per component to '" << GRAPH_NAME << "' network\n";

    //Decompress using a sliding window
    //spacial coordinates using kakadu decompressor
    kdu_region_decompressor decompressor;

    for(int c = START_COMPONENT_INDEX; c <= FINAL_COMPONENT_INDEX; c++){
      //For consistency
      int component_index = c;
      for(int x_win = 0; x_win < steps_x; x_win++){
        for(int y_win = 0; y_win < steps_y; y_win++){
          //Construct a region from the label data
          int x = LIMIT_RECT_X + x_win*stride_x;
          int y = LIMIT_RECT_Y + y_win*stride_y;
          kdu_dims region;
          region.access_pos()->set_x(x);
          region.access_size()->set_x(INPUT_WIDTH);
          region.access_pos()->set_y(y);
          region.access_size()->set_y(INPUT_HEIGHT);

          //TODO: variable
          int discard_levels = 0;

          //Get safe expansion factors for the decompressor
          //Safe upper bounds & minmum product returned into the following variables
          double min_prod;
          double max_x;
          double max_y;
          decompressor.get_safe_expansion_factors(
            codestream, //The codestream being decompressed
            NULL,  //Codestream's channel mapping (null because specifying index)
            component_index,    //Check over all components
            discard_levels,     //DWT levels to discard (resolution reduction)
            min_prod,
            max_x,
            max_y
          );

          kdu_dims component_dims;  //Holds expanded component coordinates (post discard)
          kdu_dims incomplete_region;  //Holds the region that is incomplete after processing run
          kdu_dims new_region;  //Holds the region that is rendered after a processing run

          //Get the expansion factors for expansion TODO: non 1x expansion
          kdu_coords scale_num;
          scale_num.set_x(1); scale_num.set_y(1);
          kdu_coords scale_den;
          scale_den.set_x(1); scale_den.set_y(1);

          //Get the size of the complete component (after discard levels decrease
          //in resolution is applied) on the rendering canvas
          component_dims = decompressor.get_rendered_image_dims(
            codestream, //The codestream being decompressed
            NULL,       //Codestream's channel mapping (null because specifying index)
            component_index,  //Component being decompressed
            discard_levels,   //DWT levels to discard (resolution reduction)
            scale_num,
            scale_den
          );

          //Should a region not fully be included in the image then skip to the next
          if( region.pos.x < component_dims.pos.x ||
              region.pos.x + region.size.x > component_dims.pos.x + component_dims.size.x ||
              region.pos.y < component_dims.pos.y ||
              region.pos.y + region.size.y > component_dims.pos.y + component_dims.size.y){
                break;
          }

          //Create a buffer to send the data across into
          int bufsize = INPUT_WIDTH*INPUT_HEIGHT;
          kdu_uint32 buffer[bufsize];

          //Loop decompression to ensure that amount of DWT discard levels doesn't
          //exceed tile with minimum DWT levels
          do{
            //Set up decompressing run
            //cout << "Starting decompression\n";
            decompressor.start(
              codestream, //The codestream being decompressed
              NULL,  //Codestream's channel mapping (null because specifying index)
              component_index,  //Component being decompressed
              discard_levels,   //DWT levels to discard (resolution reduction)
              INT_MAX,    //Max quality layers
              region,      //Region to decompress
              scale_num,  //Expansion factors
              scale_den,
              &env        //Multi-threading environment
            );

            //Decompress until buffer is filled (block is fully decompressed)
            //cout << "Processing\n";
            incomplete_region = component_dims;
            while(
              decompressor.process(
                (kdu_int32 *) buffer,   //Buffer to write into
                region.pos,             //Buffer origin
                region.size.x,          //Row gap
                256000,                 //Suggesed increment
                0,                      //Max pixels in region
                incomplete_region,
                new_region
              )
            );
            //Finalise decompressing run
            //cout << "Finishing decompression\n";
            decompressor.finish();

          //Render until there is no incomplete region remaining
          }while(!incomplete_region.is_empty());

          //Send the now complete image data
          int image_transmitted = send(server, &buffer, sizeof(buffer), 0);
          if(image_transmitted != bufsize*4){
            cout << "Error: image " << evaluation_units_fed + 1
              << " (" << x << ", " << y << ", "  << c
              << ") was not sent completely over socket, "
              << "len=" << image_transmitted << "\n";
            return;
          }

          //Track
          evaluation_units_fed++;
        }
        //If done a full line then increment into next column
      }
      //Outside of sliding window loops
    }
    //Outside of component loop

    //Close the socket in the parent and prevent further transmissions. The
    //python process can still recv though
    if(shutdown(sockfd, SHUT_WR) != 0){
      cout << "Error, couldn't shutdown socket in parent\n";
    }

    //Wait for death of child (python evaluation process)
    while(wait(&status) != pid);

  }else{
    //Fork failure
    cout << "Error: 'fork()' failed when creating python training process\n";
    exit(EXIT_FAILURE);
  }
}

//Called if 'h' or 'u' was called at the command line
void print_usage(){
  cout << "Usage: TODO\n";
}

//----------------------------------------------------------------------------//
// Main                                                                       //
//----------------------------------------------------------------------------//
int main(int argc, char **argv){
  //For getopt
  int index;
  int arg;

  //Run getopt loop
  while((arg = getopt(argc, argv, "f:tve:g:c:r:p:nuh")) != -1){
    switch(arg){
      case 'f': //Filepath to image
        JPX_FILEPATH = optarg;
        break;
      case 't': //Train?
        IS_TRAIN = true;
        break;
      case 'v': //Validate?
        IS_VALIDATE = true;
        break;
      case 'e': //Evaluate?
        {
          IS_EVALUATE = true;

          //Parse rectangle as comma delimited string if evaluating
          vector<int> limit_rect;
          std::stringstream ss_limit_rect(optarg);
          int limit_rect_args;
          while(ss_limit_rect >> limit_rect_args){
            limit_rect.push_back(limit_rect_args);
            if(ss_limit_rect.peek() == ','){
              ss_limit_rect.ignore();
            }
          }
          //Error check rectangle values
          //TODO
          if(limit_rect.size() != 4){
            //Should only be a start and end index (inclusive)
            cout << "Error: limiting rectangle should be given in form 'x,y,w,h'\n";
            return -1;
          }

          //Set globals
          LIMIT_RECT_X = limit_rect[0];
          LIMIT_RECT_Y = limit_rect[1];
          LIMIT_RECT_W = limit_rect[2];
          LIMIT_RECT_H = limit_rect[3];

          break;
        }
      case 'g': //Name of graph to use
        GRAPH_NAME = optarg;
        break;
      case 'c': //Component range
        {
          //Parse component range as comma delimited string
          vector<int> range;
          std::stringstream ss_range(optarg);
          int comps;
          while(ss_range >> comps){
            range.push_back(comps);
            if(ss_range.peek() == ','){
              ss_range.ignore();
            }
          }
          //Error check component range
          if(range.size() != 2){
            //Should only be a start and end index (inclusive)
            cout << "Error: inclusive component index range should be in the form 'a,b'\n";
            return -1;
          }
          int start_component_index = range[0];
          int final_component_index = range[1];
          if(final_component_index < start_component_index){
            cout << "Error: inclusive component index range's start index is after end index\n";
            return -1;
          }

          //Set global vars
          START_COMPONENT_INDEX = start_component_index;
          FINAL_COMPONENT_INDEX = final_component_index;
          break;
        }
      case 'r': //Resolution level to use
        RESOLUTION_LEVEL = atoi(optarg);  //Convert to int
        break;
      case 'p': //Port number to IPC on;
        PORT_NO = atoi(optarg);
        break;
      case 'n': //Whether or not to use a plugged in NCS (otherwise will use CPU)
        NCS_EVALUATION = true;
        break;
      case 'u':
        print_usage();
        exit(EXIT_SUCCESS);
        break;
      case 'h':
        print_usage();
        exit(EXIT_SUCCESS);
        break;
      case '?':
        //getopt handles error print statements
        return -1;
        break;
      default:
        //Something has gone horribly wrong
        cout << "Error: getopt failed to recognise command line arguments\n";
        return -1;
        break;
    }
  }

  //Error check arguments
  int typeFlagCount = 0;
  typeFlagCount += (IS_TRAIN) ? 1 : 0;
  typeFlagCount += (IS_VALIDATE) ? 1 : 0;
  typeFlagCount += (IS_EVALUATE) ? 1 : 0;
  if(typeFlagCount != 1){
    cout << "Error: training, validation and evaluation sessions are mutally exclusive; please pick one option\n";
    return -1;
  }

  //Announce KDU core version and prepare error output
  cout << "KDU: " << kdu_get_core_version() << "\n";
  kdu_customize_warnings(&pretty_cout);
  kdu_customize_errors(&pretty_cerr);

  //Initialise numpy arrays for converting blocks into tensors
  cout << "Creating embedded python3 environment\n";
  init_embedded_python();

  //Create a jp2_family_src that takes a filepath as a parameter
  cout << "Creating jp2 family source from file: '" << JPX_FILEPATH << "'\n";
  jp2_family_src jp2_fam_src;
  jp2_fam_src.open(JPX_FILEPATH);

  //Create jpx_source that can load an opened jp2_family_src
  cout << "Creating jpx source from jp2 family source\n";
  jpx_source jpx_src;
  //True as second arg allows -1 check rather than going through KDU errors
  jpx_src.open(&jp2_fam_src, true);

  //Create a multi-threaded processing environment
  cout << "Creating multi-threading environment\n";
  kdu_thread_env env;
  env.create();
  int num_threads = kdu_get_num_processors();
  for(int nt=1; nt < num_threads; nt++){
    if(!env.add_thread()){
      num_threads = nt; //Unable to create all the threads requested, take as
                        //many as possible
    }
  }

  //Declare codestream interface to the file
  cout << "Creating codestream interface from jp2_source\n";
  kdu_core::kdu_codestream codestream;
  //jpx files may have multiple codestreams, access a single codestream and feed to codestream create function
  codestream.create(jpx_src.access_codestream(0).open_stream());
  //Set fussy to write errors with prejudice
  codestream.set_fussy();
  //Persistence is desirable as it permits multiple accesses to the same information
  //so new regions or resolutions of interest can be decoded at will (blocks)
  codestream.set_persistent();

  //Check that the user specified components aren't out of range now that
  //codestream is initialised
  if( codestream.get_num_components() - 1 < FINAL_COMPONENT_INDEX
      || START_COMPONENT_INDEX < 0){
    cout << "Error: specified component range [" << START_COMPONENT_INDEX << ", "
      << FINAL_COMPONENT_INDEX << "] is outside of input file's component range [0, "
      << codestream.get_num_components() - 1 << "], exiting\n";
      return -1;
  }

  //Print statistics
  //print_statistics(codestream);

  //TODO check resolution level is correct

  //Begin timing

  //Split on training/validation/evaluating
  if(IS_TRAIN || IS_VALIDATE){
    //Begin by getting labels in decompressor feedable format
    vector<label> labels;
    load_labels_from_roid_container(jpx_src, labels);
    int num_true_labels = labels.size();
    cout << num_true_labels << " galaxy labels found in inclusive component range ["
      << START_COMPONENT_INDEX << ", " << FINAL_COMPONENT_INDEX << "]\n";

    //Now add some false labels. False labels won't be where a galaxy is spatially,
    //so get 100x100's that aren't in the galaxy components. Get as many false labels
    //as true labels
    generate_false_labels(labels);
    cout << labels.size() - num_true_labels << " noise labels generated\n";

    //Note jpx_src required for metadata reads and codestream required for
    //image decompression. The final argument specifies if the model should
    //be update (which it should during training)
    train(labels, codestream, env);
  }
  if(IS_EVALUATE){
    //Don't actually need any labels here, just get started
    evaluate(codestream, env);
  }

  //Destroy references
  //Retract multi-threading environment before destroying codestream
  cout << "Closing multi-threading environment\n";
  env.destroy();
  cout << "Closing codestream\n";
  codestream.destroy();

  //Close references
  cout << "Closing jpx source\n";
  jpx_src.close();
  cout << "Closing jp2 family source\n";
  jp2_fam_src.close();

  //Close python interpreter
  cout << "Closing embedded Python3 environment\n";
  end_embedded_python();

  //Exit with success
  return 0;
}
