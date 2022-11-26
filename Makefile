all:	train test predict README.pdf libnn.so

libnn.so: nn.o data_prep.o
	$(RM) $@
	$(AR) rcs $@ nn.o data_prep.o
	$(CC) -shared -Wl,-o libnn.so *.o

data_prep.o: data_prep.c data_prep.h
	$(CC) -Wall data_prep.c -c -march=native -flto -Ofast -fPIC

nn.o: nn.c nn.h
	$(CC) -Wall nn.c -c -march=native -flto -Ofast -fPIC

train: train.c nn.o data_prep.o
	$(CC) -Wall train.c data_prep.o nn.o -o train -lm -march=native -Ofast

test: test.c nn.o data_prep.o
	$(CC) -Wall test.c data_prep.o nn.o -o test -lm -march=native -Ofast

predict: predict.c nn.o
	$(CC) -Wall predict.c nn.o -o predict -lm -march=native -Ofast

README.pdf: README.md
	pandoc README.md -o README.pdf

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) data_prep.o nn.o libnn.so train test predict model.txt tags nn.png README.pdf
	$(RM) -r __pycache__
