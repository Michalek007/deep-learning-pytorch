import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt


training_data = datasets.LFWPairs(
    root="data",
    split="train",
    download=True
)
plt.imshow(training_data[0][0])
plt.show()
plt.imshow(training_data[0][1])
plt.show()


# training_data = datasets.LFWPairs(
#     root="data",
#     split="train",
#     download=True,
#     transform=ToTensor()
# )
#
#
# train_dataloader = DataLoader(training_data, batch_size=64)
#
#
# for i, item in enumerate(train_dataloader):
#     if i == 1:
#         break
#     print(i)
#     print(len(item))
#     print(item[0].shape)
#     print(item[1].shape)
#     print(item[2].shape)
#
#
# def display_image():
#     # train_features = next(iter(train_dataloader))
#
#     # print(f"Feature batch shape: {train_features.size()}")
#     # img = train_features[0].squeeze()
#     plt.imshow(torch.reshape(training_data[0][0], (250, 250, 3)))
#     # plt.imshow(torch.reshape(training_data[0][1], (250, 250, 3)))
#     plt.show()
#
#
# display_image()
