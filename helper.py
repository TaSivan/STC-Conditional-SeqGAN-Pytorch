import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "repository")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

PAD = 0
SOS = 1
EOS = 2
UNK = 3