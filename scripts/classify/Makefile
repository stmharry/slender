IMAGE_DIR=/mnt/data/food-img
WORKING_DIR_ROOT=/mnt/data/food-save/classify

train:
	./train.py --image_dir $(IMAGE_DIR) --working_dir_root $(WORKING_DIR_ROOT) --gpu_frac 0.5

test:
	./test.py --image_dir $(IMAGE_DIR) --working_dir_root $(WORKING_DIR_ROOT) --gpu_frac 0.15
