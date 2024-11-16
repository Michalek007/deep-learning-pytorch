import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from typing import Optional, Callable
from facenet_pytorch import MTCNN
from pathlib import Path
import PIL
import torch
from torchvision.transforms import ToPILImage


class LWFDataset(datasets.LFWPairs):
    def __init__(self, root: str, split: str = "10fold", image_set: str = "funneled", transform: Optional[Callable] = None,
                        target_transform: Optional[Callable] = None, download: bool = False,):
        super().__init__(root, split, image_set, transform, target_transform, download)
        same_idx = []
        diff_idx = []
        for i, (label1, label2) in enumerate(self.pair_names):
            if label1 == label2:
                same_idx.append(i)
            else:
                diff_idx.append(i)

        self.triplets = []
        for i in range(len(same_idx)):
            self.triplets.append((same_idx[i], (diff_idx[i], i % 2)))

        to_removed = []
        self.triplets_labels = []
        for same_idx, (diff_idx, idx) in  self.triplets:
            if self.pair_names[diff_idx][idx] in self.pair_names[same_idx]:
                to_removed.append((same_idx, (diff_idx, idx)))
                continue
            self.triplets_labels.append((self.pair_names[same_idx][0], self.pair_names[same_idx][1], self.pair_names[diff_idx][idx]))

        for value in to_removed:
            self.triplets.remove(value)

        self.mtcnn = MTCNN(keep_all=True)
        # lwf_mtcnn_path = Path('data/lwf-mtcnn')
        # if not lwf_mtcnn_path.is_dir():
        #     lwf_mtcnn_path.mkdir()
        #     transform = ToPILImage()
        #     img1, img2, img3 = map(transform, self.__getitem__(0))
        #     img1.save(self.triplets_labels[0][0] + '.jpg')
        #     img2.save(self.triplets_labels[0][1] + '.jpg')
        #     img3.save(self.triplets_labels[0][2] + '.jpg')

        # print(self.triplets_labels)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (anchor img, positive img, negative img)
        """
        if self.split == 'test' or self.split == '10fold':
            img1, img2 = self.data[index]
            img1, img2 = self._loader(img1), self._loader(img2)
            target = self.targets[index]
            # if self.transform is not None:
            #     img1, img2 = self.transform(img1), self.transform(img2)
            img1, img2 = self.mtcnn(img1)[0], self.mtcnn(img2)[0]
            return img1, img2, target

        same_idx, (diff_idx, idx) = self.triplets[index]

        img1, img2 = self.data[same_idx]
        img3 = self.data[diff_idx][idx]

        img1, img2, img3 = self._loader(img1), self._loader(img2), self._loader(img3)

        # if self.transform is not None:
        #     img1, img2, img3 = self.transform(img1), self.transform(img2), self.transform(img3)

        # boxes, _ = self.mtcnn.detect(img1)
        # if boxes is not None:
        #     img1 = self.mtcnn(img1)[0]
        # boxes, _ = self.mtcnn.detect(img2)
        # if boxes is not None:
        #     img2 = self.mtcnn(img2)[0]
        # boxes, _ = self.mtcnn.detect(img3)
        # if boxes is not None:
        #     img3 = self.mtcnn(img3)[0]
        img1 = self.mtcnn(img1)[0]
        img2 = self.mtcnn(img2)[0]
        img3 = self.mtcnn(img3)[0]
        # print(img1.size())
        # print(img2.size())
        # print(img3.size())
        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
