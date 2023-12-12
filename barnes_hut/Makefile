CC=mpicc
LD=${CC}
DEBUG=-g
CFLAGS+=-Wall -Wextra -pedantic -pg $(DEBUG)
LDFLAGS+=-lm -pg $(DEBUG)

OBJS_NBODY=nbody.o reader.o nbody_barneshut.o

all: nbody-code

nbody-code: $(OBJS_NBODY)
	$(LD) $(OBJS_NBODY) $(LDFLAGS) -o $@

clean:
	rm -Rf nbody-code *.o *~  gmon.out
