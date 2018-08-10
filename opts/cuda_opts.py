import torch

""" Enable GPU training """

USE_CUDA = torch.cuda.is_available()
print('Use_CUDA={}'.format(USE_CUDA))
if USE_CUDA:
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # print(device)
    # You can change device by `torch.cuda.set_device(device_id)`
    print('current_device={}'.format(torch.cuda.current_device()))


USE_PARALLEL = False
# USE_PARALLEL = True if torch.cuda.device_count() > 1 else False
print('Use_Parallel={}'.format(USE_PARALLEL))
if USE_PARALLEL:
    print("use", torch.cuda.device_count(), "GPUs")