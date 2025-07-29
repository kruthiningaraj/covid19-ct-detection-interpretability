
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def segment_lung(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.bitwise_and(image, image, mask=thresh)

def create_generators(train_dir, val_dir, img_size=(224,224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
    val_gen = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
    return train_gen, val_gen
