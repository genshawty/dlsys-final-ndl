import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import gzip

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        if self.padding == 0:
            return img
        h, w, _ = img.shape

        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        
        pad_img = np.pad(img, self.padding, mode="constant")[:, :, self.padding:-self.padding]

        return pad_img[self.padding + shift_x:self.padding + shift_x + h, self.padding + shift_y:self.padding + shift_y + w]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            arange = np.arange(len(self.dataset))
            np.random.shuffle(arange)
            self.ordering = np.array_split(arange, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        return self

    def make_tensor(self, item):
        return Tensor(item)

    def __next__(self):
        if self.i >= len(self.ordering):
            raise StopIteration()

        batch = tuple(
            map(
                self.make_tensor, 
                self.dataset[
                    self.ordering[self.i]
                    ]
                )
            )

        self.i += 1
        return batch

    def __len__(self):
        return len(self.ordering)


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms

        self.images, self.labels = self.parse_mnist()
        

    def __getitem__(self, index) -> object:
        img = self.images[index]
        label = self.labels[index]
        if len(img.shape) > 1:
            for idx in range(img.shape[0]):
                im = img[idx].reshape(28, 28, 1)
                if self.transforms is not None:
                    for t in self.transforms:
                        im = t(im)
                if isinstance(im, Tensor):
                    im = im.numpy()
                img[idx] = im.reshape(-1)

            return img, label
        else:
            img = img.reshape(28, 28, 1)
            if self.transforms is not None:
                for t in self.transforms:
                    img = t(img)
            if isinstance(img, Tensor):
                img = img.numpy()
            return img.reshape(-1), label

    def __len__(self) -> int:
        return self.images.shape[0]

    def parse_mnist(self):
        image_filename = self.image_filename
        label_filename = self.label_filename

        f = gzip.open(image_filename,'r')
        labels = gzip.open(label_filename, 'r')

        image_size = 28
        f.read(16)
        labels.read(8)

        buf = f.read()
        
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(len(buf) // (image_size * image_size), image_size*image_size)
        
        data = np.divide(data, 255)
        
        data_labels = np.frombuffer(labels.read(), dtype=np.uint8)
        return data, data_labels

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
