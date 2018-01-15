//Compile with makefile provided

//For example:  ./gfinder -f /mnt/shared-storage/dingo.00000.with_catalogue.jpx -g test-graph -t -r 0 -c 0,899
//              ./gfinder -f /mnt/shared-storage/dingo.00000.with_catalogue.jpx -g test-graph -v -r 0 -c 900,993
//              ./gfinder -f /mnt/shared-storage/dingo.00000.with_catalogue.jpx -g test-graph -e 165,1640,35,40 -r 0 -c 994,994

//C++ standard includes
#include <iostream>     //For cout
#include <iomanip>      //For pretty printing
#include <sstream>      //For parsing command line arguments
#include <string>       //For xtoy conversions
#include <algorithm>    //For min
#include <math.h>       //For ceil, pow
#include <stdlib.h>     //For atoi and atof
#include <vector>       //For vectors
#include <unistd.h>     //For getopt

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
                                      vector<label> & labels,
                                      int start_component_index,
                                      int final_component_index)
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
          if(l.f >= start_component_index && l.f <= final_component_index){
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
void generate_false_labels( vector<label> & labels, int start_component_index,
                            int final_component_index)
{
  //Range of components
  int range = final_component_index - start_component_index + 1; //+1 because inclusive
  //Required number of noise labels to be found
  int req = 4*labels.size();
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
    for(int i = start_component_index; i <= final_component_index; i++){
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

//Converts a series of kdu_uint32 arrays and labels into a supervised batch that
//can be fed into python
PyObject *get_supervised_batch( vector<kdu_uint32*> image_data_batch,
                                vector<bool> label_batch,
                                char *graph_name,
                                bool update_model)
{
  //A supervised batch is a list of 1D image data arrays, a list of labels, and
  //a graph name

  //Create a list of 1D image data arrays as a Python list
  PyObject* image_data_batch_py = PyList_New(image_data_batch.size());
  npy_intp len = INPUT_WIDTH*INPUT_HEIGHT;  //Length of one 1D array

  //Create a list of booleans as a Python list
  PyObject* label_batch_py = PyList_New(label_batch.size());

  for(int i = 0; i < image_data_batch.size(); i++){
    //Set the value for the image data
    PyList_SetItem(image_data_batch_py, i, Py_BuildValue("O",
      PyArray_SimpleNewFromData(
        1,
        &len,
        NPY_UINT32,
        (void *)image_data_batch[i]
      ))
    );

    //Sets the value for the boolean (which is passed as a int)
    PyList_SetItem(label_batch_py, i,
      Py_BuildValue("i", (label_batch[i] ? 1 : 0))
    );
  }

  //Create as a batch that will be passed to the python program
  PyObject* supervised_batch = Py_BuildValue("(O, O, s, i)",
    image_data_batch_py,
    label_batch_py,
    graph_name,
    (update_model ? 1 : 0)
  );

  return supervised_batch;
}

//Pretty prints the results of a batch into a table
void print_results_table( int t_pos, int f_pos, int t_neg, int f_neg,
                          int correct, int incorrect,
                          int digits)
{
  //Field names are read as: was (T = correct/F = incorrect)
  //because prediction was (+ = gal/ - = noise)
  cout << "\t\t+";
  cout << std::setfill('-') << std::setw(14) << "+";
  for(int i = 0; i < 3; i++){
      cout << std::setfill('-') << std::setw(9) << "+";
  }
  cout << "\n";

  cout << "\t\t| MARK | PRED | "
    << std::setfill(' ') << std::setw(digits) << std::left << "GALAXY" << " | "
    << std::setfill(' ') << std::setw(digits) << std::left << "NOISE" << " | "
    << std::setfill(' ') << std::setw(digits) << std::left << "TOTALS" << " |\n";
  cout << std::resetiosflags(std::ios::adjustfield);

  cout << "\t\t+";
  cout << std::setfill('-') << std::setw(14) << "+";
  for(int i = 0; i < 3; i++){
      cout << std::setfill('-') << std::setw(9) << "+";
  }
  cout << "\n";

  std::ostringstream c_oss;
  c_oss << " +" << correct << " (CORRECT)";
  cout << "\t\t|     CORRECT | "
    << std::setfill(' ') << std::setw(digits) << std::left << t_pos << " | "
    << std::setfill(' ') << std::setw(digits) << std::left << t_neg << " | "
    << std::setfill(' ') << std::setw(digits) << std::left << t_pos + t_neg << " | "
    << std::setfill(' ') << std::setw(17) << std::left << c_oss.str();
  cout << std::resetiosflags(std::ios::adjustfield) << "| \n";

  cout << "\t\t+";
  cout << std::setfill('-') << std::setw(14) << "+";
  for(int i = 0; i < 3; i++){
      cout << std::setfill('-') << std::setw(9) << "+";
  }
  cout << std::setfill(' ') << std::setw(18) << std::right << " " << "|--> "
    << "BATCH ACCURACY: "
    << 100*(double)(correct)/(double)(incorrect + correct) << "%\n";

  std::ostringstream inc_oss;
  inc_oss << " +" << incorrect << " (INCORRECT)";
  cout << "\t\t|   INCORRECT | "
    << std::setfill(' ') << std::setw(digits) << std::left << f_pos << " | "
    << std::setfill(' ') << std::setw(digits) << std::left << f_neg << " | "
    << std::setfill(' ') << std::setw(digits) << std::left << f_pos + f_neg << " | "
    << std::setfill(' ') << std::setw(17) << std::left << inc_oss.str();
  cout << std::resetiosflags(std::ios::adjustfield) << "| \n";

  cout << "\t\t+";
  cout << std::setfill('-') << std::setw(14) << "+";
  for(int i = 0; i < 3; i++){
      cout << std::setfill('-') << std::setw(9) << "+";
  }
  cout << "\n";

  cout << "\t\t|      TOTALS | "
    << std::setfill(' ') << std::setw(digits) << std::left << t_pos + f_pos << " | "  //Galaxy guesses
    << std::setfill(' ') << std::setw(digits) << std::left << f_neg + t_neg << " | "  //Noise guesses
    << std::setfill(' ') << std::setw(digits) << std::left << t_pos + t_neg + f_pos + f_neg << " | "
    << " SESSION ACCURACY: " << 100*(double)(t_pos + t_neg)/(double)(t_pos + t_neg + f_pos + f_neg) << "%\n";
  cout << std::resetiosflags(std::ios::adjustfield);

  cout << "\t\t+";
  cout << std::setfill('-') << std::setw(14) << "+";
  for(int i = 0; i < 3; i++){
      cout << std::setfill('-') << std::setw(9) << "+";
  }
  cout << "\n";
}

//Trains the nueral net on an image batch and label batch and returns the results
//to the terminal
void feed_batch_and_print_results(vector<kdu_uint32*> image_data_batch,
                                  vector<bool> label_batch,
                                  int & t_pos, int & f_pos, int & t_neg, int & f_neg,
                                  int units_expected, bool updateModel,
                                  int digits, char *graph_name)
{
  //Call the function in python to load the training unit as a tensor
  //Get filename
  PyObject* py_name   = PyUnicode_FromString((char*)"cnn");
  PyErr_Print();

  //Import file as module
  PyObject* py_module = PyImport_Import(py_name);
  PyErr_Print();

  //Get function name from module depending on if validating or training
  PyObject* py_func;
  py_func = PyObject_GetAttrString(py_module, (char*)"use_supervised_batch");
  PyErr_Print();

  //Call function with batch data
  PyObject* py_result;
  py_result = PyObject_CallObject(py_func,
    get_supervised_batch(image_data_batch, label_batch, graph_name, updateModel)
  );
  PyErr_Print();

  //Use the prediction results to track successes and failures
  vector<int> preds;
  int correct = 0;
  int incorrect = 0;
  for(int i = 0; i < image_data_batch.size(); i++){
    if(py_result != NULL){
      //PyObject_IsTrue returns 1 if py_result is true and 0 if it is false
      preds.push_back(PyObject_IsTrue(PyList_GetItem(py_result, i)));
    }else{
      cout << "Error: embedded python3 training function returning incorrectly\n";
    }

    //Track true/false positives/negatives
    if(preds[i] == 1){
      //Predicton was a galaxy (pos)
      if(label_batch[i]){
        //Was a galaxy (T)
        t_pos++;
        correct++;
      }else{
        //Was noise (F)
        f_pos++;
        incorrect++;
      }
    }else{
      //Predicton was noise (neg)
      if(label_batch[i]){
        //Was a galaxy (F)
        f_neg++;
        incorrect++;
      }else{
        //Was noise (T)
        t_neg++;
        correct++;
      }
    }
  }

  //Announce and track the number of training units fed
  cout << "\t-" << t_pos + f_pos + t_neg + f_neg << "/" << units_expected
    << ((updateModel) ? " training units " : " validation units ") << "fed to network:\n";
  print_results_table(t_pos, f_pos, t_neg, f_neg, correct, incorrect, digits);
}

//Called when training a graph is specified. Note a reference to the jpx source
//is taken - this prevents segmentation fault (same with codestream)
void train( vector<label> labels, kdu_codestream codestream, char *graph_name,
            kdu_thread_env & env,
            int start_component_index, int final_component_index,
            int resolution_level, bool updateModel)
{

  //Training data is currently a block of positives followed by a block of
  //negatives. It should be random
  std::random_shuffle(labels.begin(), labels.end());

  //Track the number of training units fed thus far and the number expected to be fed
  int units_expected = labels.size();
  int digits = units_expected > 0 ? (int) log10 ((double) units_expected) + 1 : 1;
  digits = (digits < 6) ? 6 : digits; //To hold the word 'galaxy'

  //More detailed tracking (false/true negatives/positives
  //where positive is galaxy and negative is noise)
  int f_pos = 0;
  int f_neg = 0;
  int t_pos = 0;
  int t_neg = 0;

  //Operate in batches (only recalc weights after batch change)
  vector<kdu_uint32*> image_data_batch;
  vector<bool> label_batch;

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

    //Will hold data of decompressed region
    image_data_batch.push_back(NULL);

    //For managing and allocating decompressor image buffers
    image_data_batch[image_data_batch.size() - 1] = new kdu_uint32[(size_t) region.area()];

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
          (kdu_int32 *) image_data_batch[image_data_batch.size() - 1],
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

    //Add the label to the batch
    label_batch.push_back(labels[l].isGalaxy);

    //Optimise over a batch (reduces noise in the cost function)
    if(image_data_batch.size() == 128){
      //Feed in the batch
      feed_batch_and_print_results( image_data_batch, label_batch,
                                    t_pos, f_pos, t_neg, f_neg,
                                    units_expected, updateModel,
                                    digits, graph_name);

      //Finished with batch data
      image_data_batch.clear();
      label_batch.clear();
    }
  }

  //If there are any leftovers in the batch then feed them into the neural net
  if(image_data_batch.size() != 0){
    //Feed in the batch
    feed_batch_and_print_results( image_data_batch, label_batch,
                                  t_pos, f_pos, t_neg, f_neg,
                                  units_expected, updateModel,
                                  digits, graph_name);

    //Finished with batch data
    image_data_batch.clear();
    label_batch.clear();
  }

  //At this point all valid training units have been fed into the network
  cout << t_pos + f_pos + t_neg + f_neg
    << "/" << units_expected
    << ((updateModel) ? " training" : " validation")
    << " units found in image bounds and fed into network\n";
}

//Converts 1D kdu_uint32 to 1D numpy array with dimension info for passing to python
//for evaluation
PyObject *get_evaluation_unit(kdu_uint32 array[],
                              char *graph_name)
{
  //Dimension of array
  npy_intp dim = INPUT_WIDTH*INPUT_HEIGHT;

  //Build value that can be passed to python function. Value is a 1 dimensional
  //w x h array of uint32s with the dimensions and spacing info appended as a tuple
  PyObject* evaluation_unit = Py_BuildValue("(O, s)",
    PyArray_SimpleNewFromData (
                                1,
                                &dim,
                                NPY_UINT32,
                                (void *)array
                              ),
    graph_name
  );

  //Allocate memory for a 1 x size numpy array of uint32s and input buffer data
  return evaluation_unit;
}

//Called when evaluating an image is specified
void evaluate(  kdu_codestream codestream, char *graph_name,
                kdu_thread_env & env,
                int start_component_index, int final_component_index,
                int resolution_level,
                int limit_rect_x, int limit_rect_y,
                int limit_rect_w, int limit_rect_h)
{

  //To evaluate, a sliding window over the image at the required components is
  //used to evalate on every possible region of the input image/component. Each
  //possible region is fed to the neural network graph to test if it holds a
  //galaxy. The result is crunched to find the area that most likely holds
  //a galaxy based on the results for nearby regions
  int stride_x = 5;
  int stride_y = 5;
  int image_width = limit_rect_x + limit_rect_w;
  int image_height = limit_rect_y + limit_rect_h;

  //For tracking progress
  int cols_fed = 0;
  int cols_expected = limit_rect_w/stride_x;
  int rows_expected = limit_rect_h/stride_y;
  int evaluation_units_fed = 0;
  int evaluation_units_expected = cols_expected*rows_expected;

  //Announce plan
  cout << "Image will be fed in " << evaluation_units_expected
    << " regions to " << graph_name << " network\n";

  //Decompress using a sliding window
  //spacial coordinates using kakadu decompressor
  kdu_region_decompressor decompressor;
  for(int c = start_component_index; c <= final_component_index; c++){
    //For consistency
    int component_index = c;
    for(int x = limit_rect_x; x < image_width; x += stride_x){
      for(int y = limit_rect_y; y < image_height; y += stride_y){
        //Construct a region from the label data
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

        //Will hold data of decompressed region
        kdu_uint32 *buffer = NULL;

        //For managing and allocating decompressor image buffers
        buffer = new kdu_uint32[(size_t) region.area()];

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

        //Finished decompressing here, pass it to the network graph for evaluation
        //Pass kdu_uint32 as numpy array into python3 tensorflow.
        PyObject* evaluation_unit = get_evaluation_unit(
          buffer,                         //Data
          graph_name                      //Graph to train on
        );

        //Call the function in python to load the training unit as a tensor
        //Get filename
        PyObject* py_name   = PyUnicode_FromString((char*)"cnn");
        PyErr_Print();
        //Import file as module
        PyObject* py_module = PyImport_Import(py_name);
        PyErr_Print();
        //Get function name from module
        PyObject* py_func   = PyObject_GetAttrString(py_module,
                                (char*)"use_evaluation_unit_on_ncs");
        PyErr_Print();
        //Call function with numpy aray
        PyObject* py_result;
        py_result = PyObject_CallObject(py_func, evaluation_unit);
        PyErr_Print();
        //Use the results to track successes
        int prediction = -1;
        if(py_result != NULL){
          //PyObject_IsTrue returns 1 if py_result is true and 0 if it is false
          prediction = PyObject_IsTrue(py_result);
        }else{
          cout << "Error: embedded python3 evaluation function returning incorrectly\n";
        }

        //Announce and track the number of training units fed
        evaluation_units_fed++;
        cout << "\t-evaluation unit " << evaluation_units_fed << "/"
          << evaluation_units_expected <<  " fed to network, prediction: "
          << ((prediction == 1) ? "galaxy\n" : "not galaxy\n");
      }
      cols_fed++;
      cout << "\t\t-col " << cols_fed << "/"
        << cols_expected <<  " fed to network\n";
    }
    //Outside of for loops
  }
  //Outside of component loop

}

//----------------------------------------------------------------------------//
// Main                                                                       //
//----------------------------------------------------------------------------//
int main(int argc, char **argv){
  //Use getopt to get the following arguments:
  char *jpx_filepath    = NULL;
  bool isTrain          = false;
  bool isValidate       = false;
  bool isEval           = false;
  char *graph_name      = NULL;
  char *component_range = NULL;
  int resolution_level  = -1;
  char *rect_string     = NULL;

  //For getopt
  int index;
  int arg;

  //Run getopt loop
  while((arg = getopt(argc, argv, "f:tve:g:c:r:")) != -1){
    switch(arg){
      case 'f': //Filepath to image
        jpx_filepath = optarg;
        break;
      case 't': //Train?
        isTrain = true;
        break;
      case 'v': //Validate?
        isValidate = true;
        break;
      case 'e': //Evaluate?
        isEval = true;
        rect_string = optarg; //When evaluating, the rectangle to use
        break;
      case 'g': //Name of graph to use
        graph_name = optarg;
        break;
      case 'c': //Component range
        component_range = optarg;
        break;
      case 'r': //Resolution level to use
        resolution_level = atoi(optarg);  //Convert to int
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
  typeFlagCount += (isTrain) ? 1 : 0;
  typeFlagCount += (isValidate) ? 1 : 0;
  typeFlagCount += (isEval) ? 1 : 0;
  if(typeFlagCount != 1){
    cout << "Error: training, validation and evaluation sessions are mutally exclusive; please pick one option\n";
    return -1;
  }

  //Parse component range as comma delimited string
  vector<int> range;
  std::stringstream ss_range(component_range);
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

  //Parse rectangle as comma delimited string if evaluating
  int limit_rect_x = -1;
  int limit_rect_y = -1;
  int limit_rect_w = -1;
  int limit_rect_h = -1;
  if(isEval){
    vector<int> limit_rect;
    std::stringstream ss_limit_rect(rect_string);
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

    limit_rect_x = limit_rect[0];
    limit_rect_y = limit_rect[1];
    limit_rect_w = limit_rect[2];
    limit_rect_h = limit_rect[3];
  }

  //Announce KDU core version and prepare error output
  cout << "KDU: " << kdu_get_core_version() << "\n";
  kdu_customize_warnings(&pretty_cout);
  kdu_customize_errors(&pretty_cerr);

  //Initialise numpy arrays for converting blocks into tensors
  cout << "Creating embedded python3 environment\n";
  init_embedded_python();

  //Create a jp2_family_src that takes a filepath as a parameter
  cout << "Creating jp2 family source from file: '" << jpx_filepath << "'\n";
  jp2_family_src jp2_fam_src;
  jp2_fam_src.open(jpx_filepath);

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
  if( codestream.get_num_components() - 1 < final_component_index
      || start_component_index < 0){
    cout << "Error: specified component range [" << start_component_index << ", "
      << final_component_index << "] is outside of input file's component range [0, "
      << codestream.get_num_components() - 1 << "], exiting\n";
      return -1;
  }

  //Print statistics
  print_statistics(codestream);

  //TODO check resolution level is correct

  //Begin timing


  //Split on training/validation/evaluating
  if(isTrain || isValidate){
    //Begin by getting labels in decompressor feedable format
    vector<label> labels;
    load_labels_from_roid_container(jpx_src, labels,
                                    start_component_index,
                                    final_component_index);
    int num_true_labels = labels.size();
    cout << num_true_labels << " galaxy labels found in inclusive component range ["
      << start_component_index << ", " << final_component_index << "]\n";

    //Now add some false labels. False labels won't be where a galaxy is spatially,
    //so get 100x100's that aren't in the galaxy components. Get as many false labels
    //as true labels
    generate_false_labels(labels, start_component_index, final_component_index);
    cout << labels.size() - num_true_labels << " noise labels generated\n";

    //Note jpx_src required for metadata reads and codestream required for
    //image decompression. The final argument specifies if the model should
    //be update (which it should during training)
    train(labels, codestream, graph_name, env,
          start_component_index, final_component_index,
          resolution_level, isTrain);
  }
  if(isEval){
    //Don't actually need any labels here, just get started
    evaluate( codestream, graph_name, env,
              start_component_index, final_component_index,
              resolution_level,
              limit_rect_x, limit_rect_y,
              limit_rect_w, limit_rect_h);
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
  cout << "Exiting successfully\n";
  return 0;
}
