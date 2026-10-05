import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


num_steps = 10

a = torch.tensor([0.1,0.6])
print(a)
spikes_rate = spikegen.rate(a, num_steps=num_steps)  # shape: [num_steps, batch_size, 784]
# spikes_latency = spikegen.latency(a, num_steps=num_steps, normalize=True)

print(spikes_rate)
# print(spikes_latency)
# # data = spike_data.unsqueeze(1)
# # print(data.shape)
