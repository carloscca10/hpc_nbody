CC=nvcc
LD=${CC}
DEBUG=-g
CFLAGS=-O3 $(DEBUG) # Removed -Wall, -Wextra, and -pedantic
LDFLAGS+=-lm $(DEBUG)

OBJS_NBODY=nbody.o nbody_bruteforce.o reader.o

all: nbody-code

nbody-code: $(OBJS_NBODY)
    $(LD) nbody_bruteforce.o reader.o nbody.o $(LDFLAGS) -o $@

nbody.o: nbody.c
    $(CC) $(CFLAGS) -c nbody.c -o nbody.o

nbody_bruteforce.o: nbody_bruteforce.cu
    $(CC) $(CFLAGS) -c nbody_bruteforce.cu -o nbody_bruteforce.o

reader.o: reader.c
    $(CC) $(CFLAGS) -c reader.c -o reader.o

clean:
    rm -Rf nbody-code *.o *~ gmon.out
