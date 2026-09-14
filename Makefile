include common.mk

.PHONY: all clean check tags

# The embeddable library (nn.[ch]) and the general-purpose model tools that
# operate on any saved model regardless of which example produced it.
# Each examples/*/ directory has its own, separate Makefile (see
# examples/*/Makefile) -- run `make` there to build that example; it isn't
# built by this one.
all: prune quantize dequantize summary export import libnn.a libnn.so

libnn.a: nn.o
	$(RM) $@
	$(AR) rv $@ $^

libnn.so: nn.o
	$(RM) $@
	$(AR) rcs $@ $^
	$(CC) -shared -Wl,-soname,libnn.so -o $@ $^

nn.o: nn.c nn.h
	$(CC) $(CFLAGS) -c $<

prune: prune.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

summary: summary.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

quantize: quantize.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

dequantize: dequantize.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

export: export.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

import: import.c libnn.a
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) nn.o libnn.a libnn.so prune quantize dequantize summary export import tags
