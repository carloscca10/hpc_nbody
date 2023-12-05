CC=nvcc
LD=${CC}
DEBUG=-g
CFLAGS=-O3 $(DEBUG) # Removed -Wall, -Wextra, and -pedantic
LDFLAGS+=-lm $(DEBUG)

OBJS_NBODY=nbody.o nbody_bruteforce.o reader.o

all: nbody-code

nbody-code: $(OBJS_NBODY)
	$(LD) $(OBJS_NBODY) $(LDFLAGS) -o $@

nbody.o: nbody.cu
	$(CC) $(CFLAGS) -c nbody.cu -o nbody.o

nbody_bruteforce.o: nbody_bruteforce.c
	$(CC) $(CFLAGS) -c nbody_bruteforce.c -o nbody_bruteforce.o

reader.o: reader.c
	$(CC) $(CFLAGS) -c reader.c -o reader.o

clean:
	rm -Rf nbody-code *.o *~ gmon.out
