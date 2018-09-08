from keras.models import Model
from keras.layers import Conv2D, Dense, Input, MaxPool2D, Flatten
from keras.utils import plot_model


def piece_model() -> Model:
    img = Input((32, 32, 1))
    x = Conv2D(16, 3, activation='relu')(img)
    x = Conv2D(16, 3, activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(16, 3, activation='relu')(x)
    x = Conv2D(16, 3, activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(16, 3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(13, activation='softmax')(x)

    m = Model(inputs=img, outputs=x)
    m.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    return m


if __name__ == '__main__':
    model = piece_model()
    model.summary()
    plot_model(model)
