include ../../common.mk

.PHONY: all clean

all: train_rnn train_gru train_lstm test

# Recursing into the top-level Makefile (rather than duplicating nn.o's
# build rule here) keeps nn.c compiled in exactly one place regardless of
# how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

fall_data.o: fall_data.c fall_data.h
	$(CC) $(CFLAGS) -c $<

train_rnn: train_rnn.c fall_data.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) train_rnn.c fall_data.o ../../libnn.a -o $@

train_gru: train_gru.c fall_data.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) train_gru.c fall_data.o ../../libnn.a -o $@

train_lstm: train_lstm.c fall_data.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) train_lstm.c fall_data.o ../../libnn.a -o $@

test: test.c fall_data.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) test.c fall_data.o ../../libnn.a -o $@

clean:
	$(RM) fall_data.o train_rnn train_gru train_lstm test model.*
