import cv2.cv2 as cv2
import numpy as np


def detect_board_opencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[0:1400, :]
    # gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
    # gray = cv2.equalizeHist(gray)
    cv2.imwrite('gray.png', gray)
    # edges = cv2.Canny(img, 100, 200)
    # cv2.imwrite('edges.png', edges)

    # Find the chess board corners
    inner_corners = 7
    ret, corners = cv2.findChessboardCorners(gray, (inner_corners, inner_corners), cv2.CALIB_CB_ADAPTIVE_THRESH)

    if ret:
        min_x = np.min(corners[:, 0, 0])
        min_y = np.min(corners[:, 0, 1])
        max_x = np.max(corners[:, 0, 0])
        max_y = np.max(corners[:, 0, 1])

        dx = (max_x - min_x) / (inner_corners - 1.)
        dy = (max_y - min_y) / (inner_corners - 1.)

        board = (int(min_x - dx), int(min_y - dy), int(max_x + dx), int(max_y + dy))
        if True:
            draw = cv2.drawChessboardCorners(gray, (inner_corners, inner_corners), corners, ret)
            cv2.imwrite('draw.png', draw)

        return board
    else:
        return None


def detect_board_color(img):
    # TODO take colors from settings
    color_b1 = [174, 137, 104]
    color_b2 = [170, 162, 86]
    color_w1 = [207, 209, 134]
    color_w2 = [236, 217, 185]

    b, g, r = np.moveaxis(img, -1, 0)

    board = 127 * np.ones(img.shape, dtype=np.uint8)
    mask = (r == 174) & (g == 137) & (b == 104)
    board[mask] = 0

    cv2.imwrite('b.png', board)

    a = np.logical_or(img == color_b1, img == color_b2)

    print('a')
