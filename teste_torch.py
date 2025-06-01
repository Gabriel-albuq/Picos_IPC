import torch
print(torch.cuda.is_available())        # True se a GPU está funcionando
print(torch.version.cuda)               # '12.1'
print(torch.cuda.get_device_name(0))    # Nome da sua GPU``