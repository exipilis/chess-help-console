import os
import random

import numpy as np
from imageio import imread, imsave
from keras.utils import Sequence
from scipy.misc import imresize


class ChessSequence(Sequence):
    def __init__(self,
                 image_shape: tuple,
                 image_filenames: list,
                 batch_size: int = 64,
                 shuffle: bool = True):

        self.batch_size = batch_size
        self.image_filenames = image_filenames

        self.image_shape = image_shape

        self.shuffle = shuffle
        self.batches_fetched = 0
        self.debug = False

        self.len_images = len(self.image_filenames)
        self.pieces = ['', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        self.classes_count = len(self.pieces)
        self.piece_positions = [
            ['P', 'B', 'Q', 'p', 'b', 'q', '', ''],
            ['P', 'B', 'Q', 'p', 'b', 'q', '', ''],
            ['P', 'B', 'Q', 'p', 'b', 'q', '', ''],
            ['P', 'B', 'Q', 'p', 'b', 'q', '', ''],
            ['N', 'R', 'K', 'n', 'r', 'k', '', ''],
            ['N', 'R', 'K', 'n', 'r', 'k', '', ''],
            ['N', 'R', 'K', 'n', 'r', 'k', '', ''],
            ['N', 'R', 'K', 'n', 'r', 'k', '', ''],
            ['N', 'R', 'K', 'n', 'r', 'k', '', '']
        ]

    def __len__(self):
        return np.math.ceil(self.len_images / self.batch_size)

    def __getitem__(self, index):
        self.batches_fetched += 1

        if not (self.batches_fetched % len(self)) and self.shuffle:
            random.shuffle(self.image_filenames)

        x_batch = np.zeros((self.batch_size,) + self.image_shape)
        y_batch = np.zeros((self.batch_size, self.classes_count))

        for i in range(self.batch_size):
            # random piece
            u = random.randint(0, 7)
            v = random.randint(0, 7)
            piece = self.piece_positions[u][v]
            class_id = self.pieces.index(piece)

            # random board
            image_fn = random.choice(self.image_filenames)
            img = imread(image_fn, as_gray=True)
            h, w = img.shape
            sh, sw = h // 8, w // 8

            # cut corresponding square
            cut = img[sh * u:sh * (u + 1), sw * v:sw * (v + 1)]

            # final resize
            cut = imresize(cut, self.image_shape)
            if self.debug:
                imsave('/tmp/%s_%s.png' % (piece, i), cut)

            x_batch[i] = cut.reshape(self.image_shape)

            # 1-hot encoding
            y_batch[i, class_id] = 1

        x_batch = x_batch / 127.5 - 1

        return x_batch, y_batch


if __name__ == '__main__':
    data_dir = 'data/pngs/'
    boards = [data_dir + s for s in os.listdir(data_dir) if s.endswith('.png')]
    seq = ChessSequence((32, 32, 1), boards)

    seq.__getitem__(0)
