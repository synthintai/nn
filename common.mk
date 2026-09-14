# Shared compiler/linker flags for nn.c and every program that links
# against it. Included by the top-level Makefile and by each
# examples/*/Makefile, so they can't silently drift out of sync with each
# other the way four independent copies of these lines could.
CFLAGS=-Wall -Ofast -march=native -flto -fPIC
# -lmvec: softmax's forward pass and the cross-entropy loss (nn.c) call
# expf()/logf() in simple loops that -Ofast -march=native auto-vectorizes,
# emitting calls to glibc's vector-ABI variants (_ZGVbN4v_expf etc.) which
# live in libmvec, not libm -- without it, linking any binary that pulls in
# these code paths fails with "undefined reference to `_ZGV...'".
LDFLAGS=-lm -lmvec -s
