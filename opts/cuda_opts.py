import torch

""" Enable GPU training """

USE_CUDA = torch.cuda.is_available()
print('Use_CUDA={}'.format(USE_CUDA))

if USE_CUDA:
    # You can change device by `torch.cuda.set_device(device_id)`
    print('current_device={}'.format(torch.cuda.current_device()))