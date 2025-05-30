CFLAGS=-Wall -Ofast -march=native -flto -fPIC
LDFLAGS=-lm -s
CSV_OUTPUTS := test.csv train.csv validation.csv
STAMP       := .split.stamp

.PHONY: all clean

all:	train test test_quantized predict quantize prune summary libnn.a libnn.so $(CSV_OUTPUTS)

libnn.a: nn.o data_prep.o
	$(RM) $@
	$(AR) rv $@ $^

libnn.so: nn.o data_prep.o
	$(RM) $@
	$(AR) rcs $@ $^
	$(CC) -shared -Wl,-soname,libnn.so -o $@ $^

data_prep.o: data_prep.c data_prep.h
	$(CC) $(CFLAGS) -c $<

nn.o: nn.c nn.h
	$(CC) $(CFLAGS) -c $<

train: train.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

test: test.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@


test_quantized: test_quantized.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

predict: predict.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

prune: prune.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

summary: summary.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

quantize: quantize.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(CSV_OUTPUTS)&: samples.csv split.py
	python split.py

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) data_prep.o nn.o libnn.a libnn.so train test test_quantized predict quantize prune summary model*.txt tags $(CSV_OUTPUTS)
