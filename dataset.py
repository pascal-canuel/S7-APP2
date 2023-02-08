import json
import os

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MAX_SHAPE_COUNT = 3
SHAPE_TO_CLASS = {
    'circle': 0,
    'triangle': 1,
    'cross': 2
}


class ConveyorSimulator(Dataset):
    def __init__(self, data_path, transform=None, num_classes=4):
        self._transform = transform
        self._data_path = data_path
        self._json_path = os.path.join(data_path, 'metaData.json')
        self._num_classes = num_classes
        with open(self._json_path) as f:
            self._metadata = json.load(f)

    def __len__(self):
        return len(self._metadata.keys())

    def __getitem__(self, index):
        img_id = list(self._metadata.keys())[index]

        image = Image.open(os.path.join(self._data_path, 'images', img_id))

        boxes = np.zeros((MAX_SHAPE_COUNT, 5))
        class_labels = np.zeros((3))
        for i in range(MAX_SHAPE_COUNT):
            if i >= len(self._metadata[img_id]['size']):
                continue
            size = self._metadata[img_id]['size'][i]
            pos_x, pos_y = self._metadata[img_id]['position'][i]
            class_label = SHAPE_TO_CLASS[self._metadata[img_id]['shape'][i]]
            boxes[i] = [1, pos_x, pos_y, size, class_label]
            class_labels[class_label] = 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_labels = torch.as_tensor(class_labels, dtype=torch.float32)
        masks = Image.open(os.path.join(self._data_path, 'masks', img_id))

        image = np.array(image)
        segmentation_target = np.array(masks)[:, :, 2]
        segmentation_target[segmentation_target == 0] = 4
        segmentation_target -= 1

        if self._transform is not None:
            image = self._transform(image)
        segmentation_target = torch.from_numpy(segmentation_target).long()

        return image[0:1, :, :], segmentation_target, boxes, class_labels


if __name__ == '__main__':
    dir_path = os.path.dirname(__file__)
    training_path = os.path.join(dir_path, 'data', 'training')

    t = transforms.Compose([transforms.ToTensor()])

    train_set = ConveyorSimulator(training_path, t)

    train_loader = DataLoader(train_set, batch_size=500, shuffle=True, num_workers=6)

    circle = 0
    triangle = 0
    cross = 0

    for image, masks, boxes, labels in train_loader:
        for i in range(labels.shape[0]):
            if labels[i][0] == 1:
                circle += 1
            if labels[i][1] == 1:
                triangle += 1
            if labels[i][2] == 1:
                cross += 1
    print('Circle : {}, Triangle : {}, Cross : {}'.format(circle, triangle, cross))
