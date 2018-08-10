import torch

# Pretrain Generator using MLE
from trainer.gen_train_epoch import *

# Pretrain Discriminator
from trainer.dis_train_epoch import *


# Test response with a given state
from evaluator.responsor import responsor

print(responsor.response("没有高考，你拼得过官二代吗？"))
print(responsor.response("没有高考，你拼得过官二代吗？", state="我"))
print(responsor.response("没有高考，你拼得过官二代吗？", state="这个"))