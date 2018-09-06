import cv2

from utils import detect_board_opencv, detect_board_color, detect_board_hough

img = cv2.imread('s.png')

# board = detect_board_opencv(img)
# board = detect_board_color(img)
board = detect_board_hough(img)

if board is not None:
    xmin, ymin, xmax, ymax = board
    cv2.imwrite('cut.png', img[ymin:ymax+1, xmin:xmax+1])

print(board)
