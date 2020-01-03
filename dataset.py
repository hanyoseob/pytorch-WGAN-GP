import os
import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_type='float32', nch=1, transform=[]):
        self.data_dir = data_dir
        self.transform = transform
        self.nch = nch
        self.data_type = data_type

        lst_data = os.listdir(data_dir)

        self.names = lst_data

    def __getitem__(self, index):
        data = plt.imread(os.path.join(self.data_dir, self.names[index]))[:, :, :self.nch]

        if data.dtype == np.uint8:
            data = data / 255.0

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.names)


class ToTensor(object):
    def __call__(self, data):
        data = data.transpose((2, 0, 1)).astype(np.float32)
        return torch.from_numpy(data)


class Normalize(object):
    def __call__(self, data):
        data = 2 * data - 1
        return data


class RandomFlip(object):
    def __call__(self, data):
        if np.random.rand() > 0.5:
            data = np.fliplr(data)

        return data


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        if isinstance(self.output_size, int):
          if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
          else:
            new_h, new_w = self.output_size, self.output_size * w / h
        else:
          new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        data = transform.resize(data, (new_h, new_w))
        return data


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        new_h, new_w = self.output_size

        top = int(abs(h - new_h) / 2)
        left = int(abs(w - new_w) / 2)

        data = data[top: top + new_h, left: left + new_w]

        return data


class RandomCrop(object):

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data = data[top: top + new_h, left: left + new_w]
        return data


class ToNumpy(object):
    def __call__(self, data):

        if data.ndim == 3:
            data = data.to('cpu').detach().numpy().transpose((1, 2, 0))
        elif data.ndim == 4:
            data = data.to('cpu').detach().numpy().transpose((0, 2, 3, 1))

        return data


class Denomalize(object):
    def __call__(self, data):

        return (data + 1) / 2
