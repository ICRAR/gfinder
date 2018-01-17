#Compiler
CXX = g++

#Compile time flags
CFLAGS = -g

#From Dr. Slava Kitaeff's Skuareview implementation makefile
#KDU	--------------------------------------------------------------------------
#Where to find the KDU core system, apps and managed folders
KDU_PATH = ./libs/v7_8-01265L
KDU_APPS=$(KDU_PATH)/apps
KDU_INCLUDES = -I$(KDU_PATH)/managed/all_includes -pthread	#KDU requires pthread

#Assemble library flags (search in all places for platform independance)
KDU_LFLAGS_SIMPLE = -L$(KDU_PATH)/../bin -L$(KDU_PATH)/../lib
KDU_LFLAGS_TEMP = Linux-x86-64-gcc \
									Linux-x86-32-gcc \
									Linux-arm-64-gcc \
									Linux-arm-32-gcc \
									Mac-x86-64-gcc \
									Mac-x86-32-gcc \
									Mac-PPC-gcc \
									Mingw-x86-64-gcc \
									Solaris-gcc Win-x86-64
KDU_LFLAGS = $(patsubst %, -L$(KDU_PATH)/lib/%, $(KDU_LFLAGS_TEMP)) $(KDU_LFLAGS_SIMPLE)

# Use static linking to allow distribution for research purposes, previously libkdu_v78R.a libkdu_a78R.a
KDU_LIBS =  -lkdu_aux -lkdu

#-------------------------------------------------------------------------------
#Embedded Python	--------------------------------------------------------------
#Where to find python installation (which should include Python.h, otherwise must
#'sudo apt-get install python3-dev' or similar)
PY_PATH = /usr/include/python3.5m
PY_INCLUDES = -I$(PY_PATH)

PY_LIBS = -lpython3.5m

#-------------------------------------------------------------------------------

#Add all libraries to compile with reference to
LIBS = $(KDU_LFLAGS) $(KDU_LIBS) $(PY_LIBS)

#Add all include locations to compile with reference to
INCLUDES = $(KDU_INCLUDES) $(PY_INCLUDES)

#Directories to find .cpp & .o files
SRC_DIR = src
OBJ_DIR = obj

#Define the C++ source files (ADD TO HERE)
SRCS = main.cpp

#Define the C++ object files from the source files
OBJS = $(SRCS:.cpp=.o)

#Define the final 'main' file and the rule for building it
MAIN = gfinder

#Link all .o files into main file
all: $(OBJ_DIR)/$(OBJS)
		$(CXX) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJ_DIR)/$(OBJS) $(LIBS)

#Rule for compiling each .cpp seperately
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CFLAGS) $(INCLUDES) -c -fPIC $< -o $@

#For make clean, remove all files in object folder and remove binary
.PHONY: clean

clean:
	$(RM) $(OBJ_DIR)/*.o *~ $(MAIN)
