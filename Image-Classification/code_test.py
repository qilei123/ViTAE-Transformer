import torch

model = torch.load("pretrained/ViTAEv2-B.pth.tar")

state_dict = model.pop('state_dict_ema')

torch.save(state_dict, "pretrained/ViTAEv2-B.pt")

pass