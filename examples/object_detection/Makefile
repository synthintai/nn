include ../../common.mk

CSV_OUTPUTS := train.csv validation.csv test.csv validation_boxes.txt test_boxes.txt
DATASET_DIR := dataset
# BCCD (Blood Cell Count and Detection) dataset
# (https://github.com/Shenggan/BCCD_Dataset; MIT license): ~8MB zip of 364
# annotated blood smear photos. Pinned to a specific commit, so the
# photos, boxes, and split this example's README measured against can't
# change underneath it.
BCCD_COMMIT := d272fb14cdff6e473fafeeeba32aba5f560e9e43
BCCD_URL := https://github.com/Shenggan/BCCD_Dataset/archive/$(BCCD_COMMIT).zip
BCCD_ARCHIVE := BCCD_Dataset-$(BCCD_COMMIT).zip

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

heatmap.o: heatmap.c heatmap.h image_prep.h
	$(CC) $(CFLAGS) -c $<

train: train.c image_prep.h ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. train.c ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

test: test.c image_prep.o heatmap.o ../../data_prep.o ../../libnn.a
	$(CC) $(CFLAGS) -I../.. test.c image_prep.o heatmap.o ../../data_prep.o ../../libnn.a $(LDFLAGS) -o $@

# predict.c and prepare_data.c decode real images (../../stb_image.h,
# vendored at the repository root and shared with image_classification) and
# run them through image_prep.[ch] instead of reading data_prep.[ch]'s CSV
# rows.
predict: predict.c image_prep.o heatmap.o ../../stb_image.h ../../libnn.a
	$(CC) $(CFLAGS) -I../.. predict.c image_prep.o heatmap.o ../../libnn.a $(LDFLAGS) -o $@

prepare_data: prepare_data.c image_prep.o ../../stb_image.h
	$(CC) $(CFLAGS) -I../.. prepare_data.c image_prep.o $(LDFLAGS) -o $@

# `-z $@` makes curl a conditional fetch (skips re-downloading an
# already-current archive).
$(BCCD_ARCHIVE):
	curl -s -S -L -f $(BCCD_URL) -z $@ -o $@

$(DATASET_DIR)/BCCD: $(BCCD_ARCHIVE)
	mkdir -p $(DATASET_DIR)
	unzip -q -o $< 'BCCD_Dataset-$(BCCD_COMMIT)/BCCD/*' -d $(DATASET_DIR)
	$(RM) -r $@
	mv $(DATASET_DIR)/BCCD_Dataset-$(BCCD_COMMIT)/BCCD $@
	$(RM) -r $(DATASET_DIR)/BCCD_Dataset-$(BCCD_COMMIT)
	touch $@

# Unlike most of the other examples, this doesn't go through ../../split.py:
# prepare_data keeps BCCD's published train/val/test split -- see
# prepare_data.c's top comment.
$(CSV_OUTPUTS)&: prepare_data $(DATASET_DIR)/BCCD
	./prepare_data $(DATASET_DIR)/BCCD

clean:
	$(RM) train test predict prepare_data image_prep.o heatmap.o model.* $(CSV_OUTPUTS)

distclean: clean
	$(RM) $(BCCD_ARCHIVE)
	$(RM) -r $(DATASET_DIR)
