### Requirements

  * Python 3.6

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