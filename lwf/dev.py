import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lwf_dataset import LWFDataset


training_data = LWFDataset(
    root="data",
    split="train",
    download=True
)

# training_data = datasets.LFWPairs(
#     root="data",
#     split="train",
#     download=True
# )
#
# print(training_data.pair_names)

# same = 0
# diff = 0
# same_set = set()
# diff_set = set()
# for label1, label2 in training_data.pair_names:
#     if label1 == label2:
#         same += 1
#         same_set.add(label1)
#         same_set.add(label2)
#     else:
#         diff += 1
#         diff_set.add(label1)
#         diff_set.add(label2)
#
# print(same)
# print(diff)
# print(len(same_set))
# print(len(diff_set))




# same_idx = []
# diff_idx = []
# for i, (label1, label2) in enumerate(training_data.pair_names):
#     if label1 == label2:
#         same_idx.append(i)
#     else:
#         diff_idx.append(i)
#
#
# triplets = []
# for i in range(len(same_idx)):
#     triplets.append((same_idx[i], (diff_idx[i], i%2)))
# print(triplets)
#
# triplets_labels = []
# for same_idx, (diff_idx, idx) in triplets:
#     triplets_labels.append((training_data.pair_names[same_idx], training_data.pair_names[diff_idx][idx]))
#     if triplets_labels[-1][1] in triplets_labels[-1][0]:
#         print(triplets_labels[-1])
#
# print(triplets_labels)



plt.imshow(training_data[0][0])
plt.show()
plt.imshow(training_data[0][1])
plt.show()
plt.imshow(training_data[0][2])
plt.show()
# print(training_data[0][2])

plt.imshow(training_data[1][0])
plt.show()
plt.imshow(training_data[1][1])
plt.show()
plt.imshow(training_data[1][2])
plt.show()
# print(training_data[1][2])



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


# triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
# anchor = torch.randn(100, 128, requires_grad=True)
# positive = torch.randn(100, 128, requires_grad=True)
# negative = torch.randn(100, 128, requires_grad=True)
# output = triplet_loss(anchor, positive, negative)
# print(output)
# output.backward()


# model.train()
# for epoch in tqdm(range(epochs), desc="Epochs"):
#     running_loss = []
#     for step, (anchor_img, positive_img, ne
#     gative_img, anchor_label) in enumerate(
#             tqdm(train_loader, desc="Training", leave=False)):
#         anchor_img = anchor_img.to(device)
#         positive_img = positive_img.to(device)
#         negative_img = negative_img.to(device)
#
#         optimizer.zero_grad()
#         anchor_out = model(anchor_img)
#         positive_out = model(positive_img)
#         negative_out = model(negative_img)
#
#         loss = criterion(anchor_out, positive_out, negative_out)
#         loss.backward()
#         optimizer.step()
#
#         running_loss.append(loss.cpu().detach().numpy())
#     print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))
