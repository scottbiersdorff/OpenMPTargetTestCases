SHELL = /bin/sh

EXE = test.ex
SRC = TestRedn.cpp

CXX=clang++
CXXFLAGS += -std=c++11
CXXFLAGS += -fopenmp
CXXFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_PATH} -I${CUDA_PATH}/include -ffp-contract=fast

#==========================
# Make the executable
#==========================
$(EXE): $(SRC)
	echo $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE)

#==========================
#remove all objs
#==========================
clean:
	/bin/rm -f *.o $(EXE)
