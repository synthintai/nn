include ../../common.mk

CSV_OUTPUTS := test.csv train.csv validation.csv

.PHONY: all clean distclean

all: train test predict $(CSV_OUTPUTS)

# Recursing into the top-level Makefile (rather than duplicating nn.o's
# build rule here) keeps nn.c compiled in exactly one place regardless of
# how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

data_prep.o: data_prep.c data_prep.h
	$(CC) $(CFLAGS) -c $<

train: train.c data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) train.c data_prep.o ../../libnn.a -o $@

test: test.c data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) test.c data_prep.o ../../libnn.a -o $@

# predict.c doesn't use data_prep.[ch] (it reads a single hardcoded sample,
# not train/test CSVs), so it links against ../../libnn.a alone.
predict: predict.c ../../libnn.a
	$(CC) $(CFLAGS) -I../.. $(LDFLAGS) predict.c ../../libnn.a -o $@

samples.csv:
	curl -s -S -L -f --compressed http://synthint.ai/training_data/$@.gz -z $@ -o $@.gz && gunzip $@.gz

$(CSV_OUTPUTS)&: samples.csv split.py
	python split.py

clean:
	$(RM) data_prep.o train test predict model.* $(CSV_OUTPUTS)

distclean: clean
	$(RM) samples.csv
