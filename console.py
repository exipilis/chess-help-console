import datetime
import time

import chess.uci
import cv2
import numpy as np
import pyscreenshot
from chess.engine import EngineTerminatedException
from keras.models import load_model

from utils import detect_board_hough, read_board, make_fen, detect_one, whos_turn

handler = chess.uci.InfoHandler()
# handler.multipv(3)
engine = chess.uci.popen_engine('stockfish')  # give correct address of your engine here
engine.info_handlers.append(handler)
pieces = ['', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

model = load_model('nn/weights/chess.h5')
model_shape = model.input_shape[1:]


def main():
    while True:
        # take screenshot
        screenshot = pyscreenshot.grab()
        screenshot = np.array(screenshot)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # detect board position
        board_coords = detect_board_hough(gray)
        if board_coords is None:
            print('board not detected')
            cv2.imwrite('/tmp/' + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '.png', gray)
            continue

        # recognize pieces with nn
        xmin, ymin, xmax, ymax = board_coords
        board = gray[ymin:ymax + 1, xmin:xmax + 1]

        # detect board orientation
        one_detected = detect_one(gray, board_coords)
        # print('one %s' % one_detected)

        # detect who's turn
        wt = whos_turn(gray, board_coords)
        whites_turn = one_detected != wt
        # print('whites turn %s' % whites_turn)

        piece_predictions = read_board(board, model)
        if not one_detected:
            piece_predictions = np.flip(piece_predictions, 0)

        fen = make_fen(piece_predictions, pieces, whites_turn)

        # position = np.empty((8, 8), dtype=np.str)
        # for u in range(8):
        #     for v in range(8):
        #         position[u, v] = pieces[piece_predictions[u + 8 * v]]
        # print(position)
        # print(fen)

        # engine
        time.sleep(0.01)
        board = chess.Board(fen)
        engine.position(board)

        time.sleep(0.1)

        evaltime = 500
        try:
            evaluation = engine.go(movetime=evaltime)

            # print best move, evaluation and mainline:
            print('best move: ', board.san(evaluation[0]))
            if handler.info['score'][1].cp is not None:
                print('evaluation value: ', handler.info["score"][1].cp / 100.0)
            print('Corresponding line: ', board.variation_san(handler.info["pv"][1]))
            print()
            # print(handler.info)
            # handler.multipv(5)
        except (EngineTerminatedException, ValueError) as e:
            print(e)
        continue


if __name__ == '__main__':
    main()
