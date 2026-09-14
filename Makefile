CFLAGS=-Wall -Ofast -march=native -flto -fPIC
# -lmvec: softmax's forward pass and the cross-entropy loss (nn.c) call
# expf()/logf() in simple loops that -Ofast -march=native auto-vectorizes,
# emitting calls to glibc's vector-ABI variants (_ZGVbN4v_expf etc.) which
# live in libmvec, not libm -- without it, linking any binary that pulls in
# these code paths fails with "undefined reference to `_ZGV...'".
LDFLAGS=-lm -lmvec -s
CSV_OUTPUTS := test.csv train.csv validation.csv
STAMP       := .split.stamp

.PHONY: all clean

all:	export import train_recognition train_gesture train_fall test_recognition test_gesture test_fall predict quantize dequantize prune summary libnn.a libnn.so $(CSV_OUTPUTS)

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

# Synthetic gesture data shared by train_gesture/test_gesture; not part of
# libnn.a (unlike data_prep.o) since it's specific to those two demo
# programs rather than something an embedded target linking libnn.a would want.
gesture_data.o: gesture_data.c gesture_data.h
	$(CC) $(CFLAGS) -c $<

# Synthetic fall-detection data shared by train_fall.c; same rationale as
# gesture_data.o above (not part of libnn.a).
fall_data.o: fall_data.c fall_data.h
	$(CC) $(CFLAGS) -c $<

export: export.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

import: import.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

train_recognition: train_recognition.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

train_gesture: train_gesture.c gesture_data.o libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

train_fall: train_fall.c fall_data.o libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

test_recognition: test_recognition.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

test_gesture: test_gesture.c gesture_data.o libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

test_fall: test_fall.c fall_data.o libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

predict: predict.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

prune: prune.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

summary: summary.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

quantize: quantize.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

dequantize: dequantize.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

samples.csv:
	curl -s -S -L -f --compressed http://synthint.ai/training_data/$@.gz -z $@ -o $@.gz && gunzip $@.gz

$(CSV_OUTPUTS)&: samples.csv split.py
	python split.py

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) data_prep.o nn.o gesture_data.o fall_data.o libnn.a libnn.so export import train_recognition train_gesture train_fall test_recognition test_gesture test_fall predict quantize dequantize prune summary model.* gesture_model.* fall_model.* tags $(CSV_OUTPUTS)

distclean: clean
	$(RM) samples.csv
