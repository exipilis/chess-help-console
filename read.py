import cv2
from keras.models import load_model
from scipy.misc import imresize

from utils import detect_board_hough, detect_one, whos_turn
import numpy as np

img = cv2.imread('test/180908_165259.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

model = load_model('nn/weights/chess.h5')
image_shape = model.input_shape[1:]

pieces = ['', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

board_coords = detect_board_hough(gray)

if board_coords is not None:
    xmin, ymin, xmax, ymax = board_coords

    board = gray[ymin:ymax + 1, xmin:xmax + 1]
    # cv2.imwrite('cut.png', board)

    h, w = board.shape[0:2]
    sh, sw = h // 8, w // 8

    one_detected = detect_one(gray, board_coords)
    print(one_detected)

    wt = whos_turn(gray, board_coords)
    whites_turn = one_detected and not wt
    print(whites_turn)

    x_batch = np.zeros((64,) + image_shape)

    for u in range(8):
        for v in range(8):
            cut = board[sh * u:sh * (u + 1), sw * v:sw * (v + 1)]
            cut = imresize(cut, image_shape)
            x_batch[u + 8 * v] = cut.reshape(image_shape)

    x_batch = x_batch / 127.5 - 1

    y_batch = model.predict(x_batch)

    am = np.argmax(y_batch, axis=-1)

    position = np.empty((8, 8), dtype=np.str)
    for u in range(8):
        for v in range(8):
            position[u, v] = pieces[am[u + 8 * v]]

    fen = ''
    e = 0
    for u in range(8):
        for v in range(8):
            p = pieces[am[u + v * 8]]
            if p == '':
                e += 1
            else:
                if e > 0:
                    fen += str(e)
                    e = 0
                fen += p
        if e > 0:
            fen += str(e)
            e = 0
        if u < 7:
            fen += '/'
    fen += ' w - - 0 1'
    print(position)
    print(fen)

print(board_coords)
