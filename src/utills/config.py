from os import path


class CFG:
    PROJECT_PATH = path.realpath(path.curdir)
    TRAIN_DATA_DIR = f"{PROJECT_PATH}/Data/256x256/"
    TEST_DATA_DIR = f"{PROJECT_PATH}/Data/"
    MODEL_SAVE_DIR = f"{PROJECT_PATH}/models/"
    IMAGE_DIR = f"{PROJECT_PATH}/Data/Image_tiff_files/"

    TILE_SIZE = 256
    REDUCE_RATE = 4
    SEED = 42
    BATCH_SIZE = 16
    NUM_EPOCHS = 20

    WINDOW = 1024
    MIN_OVERLAP = 32
    NEW_SIZE = 256
