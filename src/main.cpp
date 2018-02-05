//Compile with makefile provided

//For example:
//  ./gfinder -f /media/izy/irw_ntfs_0/dingo_SN_3/dingo.00000.with_catalogue.jpx -g test-graph -t -r 0 -c 0,799 -p 10000
//  ./gfinder -f /media/izy/irw_ntfs_0/dingo_SN_3/dingo.00000.with_catalogue.jpx -g test-graph -v -r 0 -c 800,899 -p 10000
//  ./gfinder -f /media/izy/irw_ntfs_0/dingo_SN_3/dingo.00000.with_catalogue.jpx -g test-graph -e 28,1489,400,400 -r 0 -c 994,994 -p 10000
//  ./gfinder -f /media/izy/irw_ntfs_0/dingo_master/dingo.00000.with_catalogue.jpx -d ./results/file-dingo.00000.with_catalogue.jpx_comp-904-906_locs-0-400-1400-400.dat


//C++ standard includes
#include <iostream>     //For cout
#include <sstream>      //For parsing command line arguments and other strings
#include <fstream>      //For reading files
#include <string>       //For xtoy conversions
#include <string.h>     //For strlen
#include <algorithm>    //For min
#include <math.h>       //For ceil, pow
#include <stdlib.h>     //For atoi and atof
#include <vector>       //For vectors
#include <unistd.h>     //For getopt
#include <limits>       //For infinity
#include <complex>      //For abs

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
//'std::' everywhere:
using std::cout;
using std::min;
using std::max;
using std::vector;
using std::string;
using std::ifstream;
using std::getline;
using std::istringstream;
using std::ostringstream;
using std::sort;
using std::abs;
using std::numeric_limits;

//Input sizes of images (in pixels) to be fed to graph
//Must reflect changes in Python file's globals
const int INPUT_WIDTH = 32;
const int INPUT_HEIGHT = 32;

//Number of images to feed per batch (minibatch)
const int BATCH_SIZE = 32;

//Global command line variables, use getopt to get the following arguments:
char *JPX_FILEPATH        = NULL;
bool IS_TRAIN             = false;
bool IS_VALIDATE          = false;
bool IS_EVALUATE          = false;
bool IS_CHECK             = false;
bool PRINT_MORE           = false;
char *RESULTS_FILEPATH    = NULL;
char *GRAPH_NAME          = NULL;
int START_COMPONENT_INDEX = 0;
int FINAL_COMPONENT_INDEX = 0;
int RESOLUTION_LEVEL      = -1;
int LIMIT_RECT_X          = 0;
int LIMIT_RECT_Y          = 0;
int LIMIT_RECT_W          = 0;
int LIMIT_RECT_H          = 0;
int PORT_NO               = -1;
bool NCS_EVALUATION       = false;


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

  bool is_galaxy;  //Does this label represent a galaxy

  //To string function for printing
  string to_string(){
    ostringstream total;

    ostringstream xss;
    xss << "x=[" << tlx << ", " << brx << "]";
    total << xss.str();
    for(int i = strlen(xss.str().c_str()); i < 16; i++){
      total << " ";
    }

    ostringstream yss;
    yss << "y=[" << tly << ", " << bry << "]";
    total << yss.str();
    for(int i = strlen(yss.str().c_str()); i < 16; i++){
      total << " ";
    }

    ostringstream fss;
    fss << "f=" << f;
    total << fss.str();
    for(int i = strlen(fss.str().c_str()); i < 8; i++){
      total << " ";
    }

    ostringstream gss;
    gss << "is_galaxy=" << ((is_galaxy) ? "true" : "false");
    total << gss.str();

    return total.str();
  }
};

//----------------------------------------------------------------------------//
// Internal functions                                                         //
//----------------------------------------------------------------------------//

//Comparator for sorting vectors of labels
bool label_frequency_comparator(label a, label b){
  return a.f < b.f;
}

//A helper that checks if two labels (typically an existing galaxy and a generated
//noise label) are overlapping. Ignores the frequency of the labels
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

//Loads label data from the ROI container in a jpx file. Note the jpx source
//is taken as a refernce - this prevents a segmentation fault later when using
//the jpx source with kakadu tools
int load_labels_from_roid_container( jpx_source & jpx_src,
                                      vector<label> & labels)
{
  //Get a reference to the meta manager
  jpx_meta_manager meta_manager = jpx_src.access_meta_manager();
  if(!meta_manager.exists()){
    return -1;  //If file didn't have meta_manager then return
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
          l.is_galaxy = true;

          //Only acknowledge if within component range (inclusive)
          if(l.f >= START_COMPONENT_INDEX && l.f <= FINAL_COMPONENT_INDEX){
            //"Your labels will make a fine addition to my ... collection"
            labels.push_back(l);
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

  //Return the number of labels found
  return labels.size();
}

//Augments the labels supplied by translating them a given distance which is
//randomly chosen between min and max supplied. Warning: the label will be
//tagged the same as what it is copying (galaxy/not galaxy) so do not translate
//the galaxy out of bounds. Returns number of generated labels and adds new
//labels to original array
int generate_translated_labels(vector<label> & labels,
                                int min_trans, int max_trans, int copies)
{
  //Count the number generated for returning
  int num_generated = 0;

  //To prevent infinite loops when copying
  int orig_size = labels.size();

  //Seed RNG
  srand(time(NULL));

  //Every label will have a given number of translated copies
  for(int l = 0; l < orig_size; l++){
    label orig_label = labels[l];

    //Go through labels and add i copies
    for(int i = 0; i < copies; i++){
      //Randomly generate bounded x and y translation
      int x_trans = rand()%max_trans + min_trans;
      int y_trans = rand()%max_trans + min_trans;

      //Randomly choose direction
      x_trans = (rand()%2 == 0) ? x_trans*-1 : x_trans;
      y_trans = (rand()%2 == 0) ? y_trans*-1 : y_trans;

      //Create a new label that is the old label translated
      label new_label;

      //Translate relative to old
      new_label.tlx = orig_label.tlx + x_trans;
      new_label.tly = orig_label.tly + y_trans;
      new_label.brx = orig_label.brx + x_trans;
      new_label.bry = orig_label.bry + y_trans;
      new_label.f   = orig_label.f;
      new_label.is_galaxy = orig_label.is_galaxy;

      //Add new label to label list
      labels.push_back(new_label);

      num_generated++;
    }
  }

  return num_generated;
}

//Creates a set of false labels. Returns how many it creates and adds new labels
//to original array
int generate_noise_labels(vector<label> & labels){
  int num_generated = 0;

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
    for(int i = START_COMPONENT_INDEX; i <= FINAL_COMPONENT_INDEX; i++){
      //If enough galaxies have been found then finish
      if(num_generated >= req){
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
        noise.is_galaxy  = false;
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
          num_generated++;
        }
      }
    }
  }

  return num_generated;
}

//Loads galaxy locations as labels from a supplied filepath and pushes them
//to supplied-by-reference vector 'results'
bool load_results_from_file(char * filepath, vector<label> & results){
  //File looks like this:
  //  #file_name:       dingo.00000.with_catalogue.jpx
  //  #component range: 904,906
  //  #bounds:          0,400,1400,400
  //  #galaxy count:    6
  //  #galaxy locations (x, y, f):
  //  73	85	904
  //  ...

  //Open the file
  string line;
  ifstream f(filepath);
  if(f.is_open()){
    //The file was successfully opened, get lines
    int count = 0;
    while(getline(f, line)){
      //First 5 lines treated differently
      if(count < 5){
        //Do nothing
      }else{
        //Read in the results, line is (x,y,f)
        istringstream iss(line);
        string item;

        //Read into label
        label l;

        iss >> item;
        l.tlx = atoi(item.c_str()) - INPUT_WIDTH/2;
        l.brx = l.tlx + INPUT_WIDTH;

        iss >> item;
        l.tly = atoi(item.c_str()) - INPUT_HEIGHT/2;
        l.bry = l.tly + INPUT_HEIGHT;

        iss >> item;
        l.f   = atoi(item.c_str());

        l.is_galaxy = true; //Obviously must be a galaxy

        //Add it back
        results.push_back(l);
      }

      count++;
    }

    //Now close the file
    f.close();
  }else{
    //Couldn't open the file
    cout << "Error: cannot open file at '" << filepath << "'\n";
    return false;
  }

  //Return success
  return true;
}

//Checks if results is equal to labels within a given tolerance and reports
//results
void check_evaluation_results(vector<label> labels, vector<label> results){
  //Track accuracy
  int successes = 0;  //When a result is in the actual labels
  //Track the min seperation in frequency between labels and results
  int total_freq_dist = 0;

  //Results are sorted in frequency, but labels should be sorted for convienience
  sort(labels.begin(), labels.end(), label_frequency_comparator);

  //Iterate over the results vector and make sure there is a matching
  //entry in labels. Find the label that is closest in frequency
  for(int r = 0; r < results.size(); r++){
    //Currently no intersection so min_freq_dist is infinite and no result
    int min_freq_dist = numeric_limits<int>::max();
    bool found_match = false;

    for(int l = 0; l < labels.size(); l++){
      //Ensure each result intersects within a given tolerance with at least one
      //label. Results may intersect with many labels across frequencies - always
      //take the closest
      if(labels_intersect(labels[l], results[r])){
        //A match has definitely been found
        found_match = true;

        //There was an intersection update if it was closer
        int freq_dist = abs(labels[l].f - results[r].f);
        min_freq_dist = (freq_dist < min_freq_dist) ? freq_dist : min_freq_dist;

        //If freq_dist is zero then this is a 3D intersection and we're done
        //for this loop
        if(min_freq_dist == 0){
          break;
        }
      }
    }

    //Track totals but don't sum inifities - that would ruin the metric
    total_freq_dist += (found_match) ? min_freq_dist : 0;
    successes += (found_match) ? 1 : 0;
  }

  //Report results depending on if there's a divide by zero
  if(successes == 0){
    cout << "No labels found that match results in file provided - perhaps "
      << "the wrong file paths were provided\n";
  }else{
    cout << successes << "/" << results.size() << " predicted galaxy locations "
      << "exist in the file's metadata ("
      << 100*(double)successes/(double)results.size()
      << "% accuracy) with an average separation in frequency of "
      << (double)total_freq_dist/(double)successes << " components\n";
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
//to edit the appearance of the decoded image with respect to frequency.
//TODO: doesn't augment underlying image data (may need to write out to new
//file - slow!)
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

//Simply saves a region of the input datacube as specified by parameters
bool save_data_as_image(kdu_codestream codestream, kdu_thread_env & env,
                        int x, int y, int w, int h, int f)
{
  //Construct a region from the given data
  kdu_dims region;
  region.access_pos()->set_x(x);
  region.access_size()->set_x(w);
  region.access_pos()->set_y(y);
  region.access_size()->set_y(h);

  //Decompress over labeled frames at the correct
  //spacial coordinates using kakadu decompressor
  kdu_region_decompressor decompressor;

  int component_index = f;

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

  //Should a region not fully be included in the image then cause error
  //skip to the next label
  if( region.pos.x < component_dims.pos.x ||
      region.pos.x + region.size.x > component_dims.pos.x + component_dims.size.x ||
      region.pos.y < component_dims.pos.y ||
      region.pos.y + region.size.y > component_dims.pos.y + component_dims.size.y){
        cout << "Error: cannot build comparison image if requested bounds exceed"
          << " codestream bounds\n";
        return false;
  }

  //Create a buffer to send the data across into. Do it on the heap here because
  //can be large (3600x3600)
  int bufsize = w*h;
  kdu_uint32* buffer = new kdu_uint32[bufsize];

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

  //Call python to recieve buffer and plot it
  //Get filename
  PyObject* py_name   = PyUnicode_FromString((char*)"cnn");
  PyErr_Print();

  //Import file as module
  PyObject* py_module = PyImport_Import(py_name);
  PyErr_Print();

  //Get function name from module depending on if validating or training
  PyObject* py_func;
  py_func = PyObject_GetAttrString(py_module, (char*)"save_data_as_comparison_image");
  PyErr_Print();

  //For converting array
  npy_intp dim = w*h;

  //Parameters to be passed to function
  PyObject* params = Py_BuildValue("(O, i, i, i, i, i, s)",
    PyArray_SimpleNewFromData(
      1,
      &dim,
      NPY_UINT32,
      (void *)buffer
    ),
    x,
    w,
    y,
    h,
    f,
    JPX_FILEPATH
  );
  PyErr_Print();

  //Call function with the graph to train on and the port to listen for
  //training data on
  PyObject_CallObject(py_func, params);
  PyErr_Print();

  //Delete the buffer as it was allocated on the heap
  delete[] buffer;

  //Return success
  return true;
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
    PyObject_CallObject(py_func, Py_BuildValue("(s, i, i, i, i, s)",
      GRAPH_NAME,
      PORT_NO,
      (IS_TRAIN ? 1 : 0),  //Whether or not to update graph
      BATCH_SIZE,
      labels.size(),
      JPX_FILEPATH
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
      kdu_uint32 label_int = (labels[l].is_galaxy ? 1 : 0);
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
  int stride_x = INPUT_WIDTH/2;
  int stride_y = INPUT_HEIGHT/2;

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

    //Call function with the graph to eval on and the port to listen for
    //training data on
    PyObject_CallObject(py_func, Py_BuildValue("(s, i, i, i, i, i, i, i, i, s)",
      GRAPH_NAME,
      PORT_NO,        //Where to get data
      LIMIT_RECT_W,   //Following so python knows how big to make eval heatmap
      LIMIT_RECT_H,
      (FINAL_COMPONENT_INDEX - START_COMPONENT_INDEX + 1),
      units_per_component,
      LIMIT_RECT_X,
      LIMIT_RECT_Y,
      START_COMPONENT_INDEX,
      JPX_FILEPATH
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
      cout << "Evaluation client successfully connected, "
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

          //Create a buffer to send the data across into. Do this on the stack
          //because data unlikely to be greater than 32x32
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

          //Send the coordinates that this image are corresponds to in the evalulation
          //area in form x,y,f
          kdu_uint32 loc[3] = {
            x - LIMIT_RECT_X,
            y - LIMIT_RECT_Y,
            c - START_COMPONENT_INDEX
          };
          int loc_transmitted = send(server, &loc, sizeof(loc), 0);
          if(loc_transmitted != 3*4){
            cout << "Error: position (" << x << ", "
              << y << ", " << c << ") was not sent completely over socket, "
              << "len=" << loc_transmitted << "\n";
            return;
          }

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

    //Compare prob map to original for each component AFTER inferencing
    //is completed
    for(int f = START_COMPONENT_INDEX; f <= FINAL_COMPONENT_INDEX; f++){
      save_data_as_image( codestream, env,
                          LIMIT_RECT_X, LIMIT_RECT_Y,
                          LIMIT_RECT_W, LIMIT_RECT_H,
                          f);
    }

  }else{
    //Fork failure
    cout << "Error: 'fork()' failed when creating python training process\n";
    exit(EXIT_FAILURE);
  }
}

//Called if 'h' or 'u' was called at the command line
void print_usage(){
  cout << "Trains, validates and evaluates convolutional neural networks "
    << "for the purpose of finding faint galaxies in JPEG2000 formatted SIDCs.\n";

  cout << "Arguments:\n"
    << "\t-c,\tthe component range (inclusive) to use: '-c start_component_index,final_component_index'\n"
    << "\t-d,\tthe filepath to an evaluation result that should be checked for differences with actual galaxy locations in input file: '-d filepath' (specifying this parameter will scan the entire input file's metadata tree, regardless of component range arguments supplied to gfinder)\n"
    << "\t-e,\twhether or not to evaluate the input using a given graph and the region to evaluate: '-e x,w,y,h'\n"
    << "\t-f,\tthe input file to use: '-f filepath'\n"
    << "\t-g,\tthe name of the graph to use: '-g graph_name'\n"
    << "\t-h,\tprints help message\n"
    << "\t-m,\tprints more information about input JPEG2000 formatted data\n"
    << "\t-n,\twhether or not to evaluate the input using attached Intel Movidius Neural Compute Sticks\n"
    << "\t-p,\tthe port to stream data from C++ decompressor to Python3 graph manipulator on (usually 10000 or greater): '-p port_number'\n"
    << "\t-r,\tthe resolution level to use the input at (default 0): '-r resolution_level'\n"
    << "\t-t,\twhether or not to train on the supplied input file\n"
    << "\t-u,\tprints usage statement\n"
    << "\t-v,\twhether or not to validate supplied graph's unit inferencing capabilities\n";
}

//----------------------------------------------------------------------------//
// Main                                                                       //
//----------------------------------------------------------------------------//
int main(int argc, char **argv){
  //For getopt
  int index;
  int arg;

  //Run getopt loop
  while((arg = getopt(argc, argv, "f:tve:g:c:r:p:nuhd:m")) != -1){
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
            cout << "Error: inclusive component index range should be in the form 'start_component_index,final_component_index'\n";
            return -1;
          }
          int start_component_index = range[0];
          int final_component_index = range[1];
          if(final_component_index < start_component_index){
            cout << "Error: inclusive component index range's start index is after the final index\n";
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
      case 'u': //Print usage
        print_usage();
        exit(EXIT_SUCCESS);
        break;
      case 'h': //Print help statement
        print_usage();
        exit(EXIT_SUCCESS);
        break;
      case 'd': //Check evaluation results with truth
        IS_CHECK = true;
        RESULTS_FILEPATH = optarg;
        break;
      case 'm': //Prints more information about input file
        PRINT_MORE = true;
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
  int type_flag_count = 0;
  type_flag_count += (IS_TRAIN) ? 1 : 0;
  type_flag_count += (IS_VALIDATE) ? 1 : 0;
  type_flag_count += (IS_EVALUATE) ? 1 : 0;
  type_flag_count += (IS_CHECK) ? 1 : 0;
  if(type_flag_count != 1){
    cout << "Error: training, validation, evaluation and checking sessions are mutally exclusive; please pick one option\n";
    return -1;
  }

  //Announce KDU core version and prepare error output
  cout << "KDU: " << kdu_get_core_version() << "\n";
  kdu_customize_warnings(&pretty_cout);
  kdu_customize_errors(&pretty_cerr);

  //Initialise numpy arrays for converting blocks into tensors
  cout << "Creating embedded Python3 environment\n";
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
  cout << "Creating multi-threading environment ";
  kdu_thread_env env;
  env.create();
  int num_threads = kdu_get_num_processors();
  for(int nt=1; nt < num_threads; nt++){
    if(!env.add_thread()){
      num_threads = nt; //Unable to create all the threads requested, take as
                        //many as possible
    }
  }
  cout << "(" << num_threads << " threads)\n";

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

  //Print statistics if required to
  if(PRINT_MORE){
    print_statistics(codestream);
  }

  //TODO check resolution level is correct

  //Split on training/validation/evaluating
  if(IS_TRAIN || IS_VALIDATE){
    //Begin by getting labels in decompressor feedable format
    vector<label> labels;
    cout << load_labels_from_roid_container(jpx_src, labels)
      << " galaxy labels found in inclusive component range ["
      << START_COMPONENT_INDEX << ", " << FINAL_COMPONENT_INDEX << "]\n";

    //Get translation translation labels,
    //args=orig_arr, min_trans, max_trans, copies
    cout << generate_translated_labels(labels, 0, 12, 8)
      << " translated labels generated\n";

    //Get as many false labels as true labels
    cout << generate_noise_labels(labels) << " noise labels generated\n";

    //Announce total number of labels
    cout << labels.size() << " labels in total\n";

    //Begin training
    train(labels, codestream, env);
  }
  if(IS_EVALUATE){
    //Don't actually need any labels here, just get started
    evaluate(codestream, env);
  }
  if(IS_CHECK){
    //Begin by getting labels in of ENTIRE file to check against
    START_COMPONENT_INDEX = 0;
    FINAL_COMPONENT_INDEX = codestream.get_num_components() - 1;
    vector<label> labels;
    cout << load_labels_from_roid_container(jpx_src, labels)
      << " labels found in file at '" << JPX_FILEPATH << "'\n";

    //Also load the galaxy locations of evaluation result as labels
    vector<label> results;
    if(load_results_from_file(RESULTS_FILEPATH, results)){
      //If unsuccessful read then end, otherwise announce find and compare
      cout << results.size()
        << " results found in file at '"
        << RESULTS_FILEPATH << "'\n";

      //Check if evaluation result is correct within a given tolerance
      check_evaluation_results(labels, results);
    }
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
