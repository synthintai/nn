include ../../common.mk

CSV_OUTPUTS := train.csv validation.csv test.csv test_oxford.csv test_oxford_labels.txt
DATASET_DIR := dataset
# Microsoft's Kaggle Cats and Dogs dataset
# (https://www.microsoft.com/en-us/download/details.aspx?id=54765; CDLA-
# Permissive 2.0): ~790MB zip of 12,500 cat and 12,500 dog photos.
KAGGLE_URL := https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
KAGGLE_ARCHIVE := kagglecatsanddogs_5340.zip
# Oxford-IIIT Pet dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/; CC
# BY-SA 4.0, image copyright with the original owners): ~790MB of photos
# and ~19MB of split lists and per-pixel trimaps. This server is slow
# (~250 KB/s per connection when measured), so this is the long download.
OXFORD_URL := https://thor.robots.ox.ac.uk/pets
OXFORD_IMAGES_ARCHIVE := images.tar.gz
OXFORD_ANNOTATIONS_ARCHIVE := annotations.tar.gz

.PHONY: all clean distclean

all: train test predict $(CSV_OUTPUTS)

# Recursing into the top-level Makefile (rather than duplicating nn.o's/
# data_prep.o's build rules here) keeps each compiled in exactly one place
# regardless of how many examples/*/ directories get built.
../../libnn.a:
	$(MAKE) -C ../.. libnn.a

../../data_prep.o:
	$(MAKE) -C ../.. data_prep.o

image_prep.o: image_prep.c image_prep.h
	$(CC) $(CFLAGS) -c $<

train: train.c image_prep.h ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

test: test.c image_prep.o ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. test.c image_prep.o ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

# predict.c and prepare_data.c decode real images (../../stb_image.h,
# vendored at the repository root and shared with object_detection) and run
# them through image_prep.[ch] instead of reading data_prep.[ch]'s CSV rows.
predict: predict.c image_prep.o ../../stb_image.h ../../libnn.a
	$(CC) $(CFLAGS) -I../.. predict.c image_prep.o ../../libnn.a $(LDFLAGS) -o $@

prepare_data: prepare_data.c image_prep.o ../../stb_image.h
	$(CC) $(CFLAGS) -I../.. prepare_data.c image_prep.o $(LDFLAGS) -o $@

# `-z $@` makes curl a conditional fetch (skips re-downloading an
# already-current archive).
$(KAGGLE_ARCHIVE):
	curl -s -S -L -f $(KAGGLE_URL) -z $@ -o $@

$(OXFORD_IMAGES_ARCHIVE) $(OXFORD_ANNOTATIONS_ARCHIVE):
	curl -s -S -L -f $(OXFORD_URL)/$@ -z $@ -o $@

$(DATASET_DIR)/PetImages: $(KAGGLE_ARCHIVE)
	mkdir -p $(DATASET_DIR)
	unzip -q -o $< 'PetImages/*' -d $(DATASET_DIR)
	touch $@

$(DATASET_DIR)/images: $(OXFORD_IMAGES_ARCHIVE)
	mkdir -p $(DATASET_DIR)
	tar -xzf $< -C $(DATASET_DIR)
	touch $@

$(DATASET_DIR)/annotations: $(OXFORD_ANNOTATIONS_ARCHIVE)
	mkdir -p $(DATASET_DIR)
	tar -xzf $< -C $(DATASET_DIR)
	touch $@

# Unlike the other examples, this doesn't go through ../../split.py:
# prepare_data keeps Oxford's published train/test split and splits the
# Kaggle photos by file number -- see prepare_data.c's top comment.
$(CSV_OUTPUTS)&: prepare_data $(DATASET_DIR)/PetImages $(DATASET_DIR)/images $(DATASET_DIR)/annotations
	./prepare_data $(DATASET_DIR)

clean:
	$(RM) train test predict prepare_data image_prep.o model.* $(CSV_OUTPUTS)

distclean: clean
	$(RM) $(KAGGLE_ARCHIVE) $(OXFORD_IMAGES_ARCHIVE) $(OXFORD_ANNOTATIONS_ARCHIVE)
	$(RM) -r $(DATASET_DIR)
