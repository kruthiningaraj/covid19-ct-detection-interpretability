
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from preprocess import create_generators

def build_model(input_shape=(224,224,3)):
    base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False
    model = models.Sequential([
        base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

if __name__ == "__main__":
    train_gen, val_gen = create_generators('data/train', 'data/val')
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save('models/covid_ct_vgg16.h5')
