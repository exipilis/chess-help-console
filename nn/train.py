from keras.callbacks import ModelCheckpoint

from sequence import ChessSequence
from model import piece_model
import os

data_dir = 'data/pngs/'
boards = [data_dir + s for s in os.listdir(data_dir) if s.endswith('.png')]

model = piece_model()
image_shape = model.input_shape[1:]

weights_filename = 'weights/chess.h5'

try:
    model.load_weights(weights_filename)
    print('weights loaded from ' + weights_filename)
except (OSError, ValueError):
    print('weights random')

train_gen = ChessSequence(image_shape, boards, 16)
valid_gen = ChessSequence(image_shape, boards, 16)

steps = 100

model.fit_generator(
    generator=train_gen,
    steps_per_epoch=steps,
    epochs=30,
    callbacks=[ModelCheckpoint(weights_filename, save_best_only=True)],
    validation_data=valid_gen,
    validation_steps=steps / 10,
    max_queue_size=train_gen.batch_size * 2,
    workers=os.cpu_count() // 2,
    use_multiprocessing=True
)
