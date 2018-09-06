import cv2

from utils import detect_board_opencv, detect_board_color

img = cv2.imread('s.png')

board = detect_board_opencv(img)
# board = detect_board_color(img)
print(board)
