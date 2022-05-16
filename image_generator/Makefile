IDIR = include
CC = mpic++
CFLAGS = -std=c++17 -O1 -fopenmp -g -I $(IDIR)

UDIR = utils
SDIR = src

_DEPS = definitions.h config_reader.h image_gen.h pdb_reader.h quaternions.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_UTILS = config_reader.cpp  pdb_reader.cpp quaternions.cpp
UTILS = $(patsubst %,$(UDIR)/%,$(_UTILS))

_SRC = main.cpp image_gen.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))

$(UDIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

../gen_img.out: $(SRC) $(UTILS)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(UDIR)/*.o *~ core $(INCDIR)/*~ 
