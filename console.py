import chess.uci
import numpy as np
import pyscreenshot

from utils import detect_board_opencv

handler = chess.uci.InfoHandler()
handler.multipv(3)
engine = chess.uci.popen_engine('stockfish')  # give correct address of your engine here
engine.info_handlers.append(handler)
i = 0

if __name__ == '__main__':

    while True:
        # take screenshot
        screenshot = pyscreenshot.grab()
        screenshot = np.array(screenshot)

        # detect board position
        board = detect_board_opencv(screenshot)
        if board is None:
            print('board not detected')
            continue

        print(board)
        continue

        # detect position

        # give your position to the engine:
        # fen = randfen.generate_board()
        # print(fen)

        i = 1 - i
        if i == 0:
            board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        else:
            board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")
        # board = chess.Board(fen)
        engine.position(board)

        # Set your evaluation time, in ms:
        evaltime = 1000  # so 5 seconds
        try:
            evaluation = engine.go(movetime=evaltime)

            # print best move, evaluation and mainline:
            print('best move: ', board.san(evaluation[0]))
            if handler.info['score'][1].cp is not None:
                print('evaluation value: ', handler.info["score"][1].cp / 100.0)
            print('Corresponding line: ', board.variation_san(handler.info["pv"][1]))
            # print(handler.info)
            # handler.multipv(5)
        except chess.engine.EngineTerminatedException:
            print('exception')
