import cv2.cv2 as cv2
import numpy as np


def detect_board_opencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # we suppose board is located in the left two thirds of the screen and doesn't touch bottom and top of the screen
    print(gray.shape)
    h, w = gray.shape
    # gray = gray[int(0.12 * h): int(0.85 * h), int(0.03 * w): int(0.47 * w)]

    print(gray.shape)
    # gray = gray[300:1400, 300:2200]
    # gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
    # gray = cv2.equalizeHist(gray)
    cv2.imwrite('gray.png', gray)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('edges.png', edges)

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
    mask = (np.abs(r - 174) < 20) & (np.abs(g - 137) < 20) & (np.abs(b - 104) < 20)
    board[mask] = 0

    cv2.imwrite('b.png', board)

    a = np.logical_or(img == color_b1, img == color_b2)

    print('a')


def detect_board_hough(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edges = cv2.Canny(gray, 210, 255, apertureSize=3)

    # look for long lines, at least 60% height
    eps = 1e-7
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(0.6 * h))
    vls = []
    hls = []
    for rho, theta in lines[:, 0]:
        if abs(theta) < eps:
            vls.append(int(rho))
        if abs(np.pi / 2 - theta) < eps:
            hls.append(int(rho))

    # sort lines
    vls.sort()
    hls.sort()

    # difference between lines
    vlsd = np.diff(vls)
    hlsd = np.diff(hls)

    # chess squares are square, w == h
    common_diffs = np.intersect1d(vlsd, hlsd)

    for cd in common_diffs:
        # find square size which is at least 8 times present in horizontal and vertical arrays
        h_where = np.where(hlsd == cd)
        v_where = np.where(vlsd == cd)
        if np.size(h_where) >= 8 and np.size(v_where) >= 8:
            xmin = vls[v_where[0][0]] + 1
            xmax = vls[v_where[0][-1]] + cd + 1
            ymin = hls[h_where[0][0]] + 1
            ymax = hls[h_where[0][-1]] + cd + 1
            cv2.imwrite('cut.png', gray[ymin:ymax, xmin:xmax])
            print('a')

    return None
