NVCC       = nvcc
# CXX     = g++
CXX       = $(NVCC)
NVCCFLAGS += -O3
CXXFLAGS  += -O3 --compiler-options -Wall
# CXXFLAGS  += -O3 --compiler-options -Wall -Wextra -pendatic
LDFLAGS   += $(NVCCFLAGS)

all: nbody-code

nbody-code: nbody.o nbody_bruteforce.o reader.o
	$(NVCC) $^ $(LDFLAGS) -o $@

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -Rf nbody-code *.o *~  gmon.out

