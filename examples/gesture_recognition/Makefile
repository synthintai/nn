include ../../common.mk

.PHONY: all clean

all: train test

# Recursing into the top-level Makefile (rather than duplicating nn.o's
# build rule here) keeps nn.c compiled in exactly one place regardless of
# how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

gesture_data.o: gesture_data.c gesture_data.h
	$(CC) $(CFLAGS) -c $<

train: train.c gesture_data.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) train.c gesture_data.o ../../libnn.a -o $@

test: test.c gesture_data.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) test.c gesture_data.o ../../libnn.a -o $@

clean:
	$(RM) gesture_data.o train test model.*
