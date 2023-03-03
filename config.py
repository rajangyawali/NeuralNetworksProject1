import torch
import os

TEST_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

BASE_OUTPUT = "output"
MAX_EPOCHS = 20
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
NUM_WORKERS = 2
