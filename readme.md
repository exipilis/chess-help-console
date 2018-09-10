### Description

This is fun weekend project intended to test the idea of 
engine help without hacking into website communication protocol.

High level idea:
  * take screenshot of the browser with Lichess board
  * detect board, pieces, board orientation and who's turn to move using 
  computer vision and neural network techniques
  * send current position to Stockfish and get best move estimate

### Disclaimer

This is not intended for serious cheating and 
I don't encourage you to use it for playing internet chess.
Moreover, I never use and stand against any kind of cheating
during chess games.

### Usage

Open Lichess web page for approximate 2/3 of screen width.

Choose black background settings and usual pieces. The board size
should be large enough to take almost all height. 
The clocks should be visible (it is used to detect who's turn).

Open console window to the right of browser window and run
`python console.py`

Console takes screenshot, detects position and runs Stockfish for 1 second.
The result and 

### Requirements

  * Python 3.6
  * Stockfish engine

### Installation

Install `virtualenv` with `python3.6`.

```bash
pip install -r requirements.txt
```

### Generate training data

To detect each piece we need to generate boards full of pieces 
for each board background and piece style.

Inside `nn/data` folder run simple server
```bash
php -S localhost:9000
```
and navigate Google Chrome to `http://localhost:9000`.
The page will render board for one piece set and all board styles.
Images are saved to `pngs/` folder by `save.php` script.

Changing `let i = 0` in `index.html` and refreshing page will yield board images for other piece set 
(doing it in a loop causes web browser hanging).

Generated pngs are stored [here](https://drive.google.com/open?id=1Xhd4tvIjbg8T7l0BRe0qKiakf_roZGkX).

These pngs are used to train simple convolution neural network which detects
individual piece.