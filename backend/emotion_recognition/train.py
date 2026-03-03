import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_emotion_model_v2

# Configuration
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 7

# Path to backend/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')

def compute_class_weights(generator):
    """Compute class weights to handle FER-2013 class imbalance using numpy."""
    class_counts = np.bincount(generator.classes)
    total = class_counts.sum()
    n_classes = len(class_counts)
    weights = total / (n_classes * class_counts.astype(float))
    return dict(enumerate(weights))

def train_model():
    # Data Augmentation and Normalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load Data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Compute class weights for imbalanced FER-2013
    class_weights = compute_class_weights(train_generator)
    print(f"Class weights: {class_weights}")

    # Create V2 Model
    model = create_emotion_model_v2(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        'emotion_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )

    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )

    print("Training Complete. Model saved as emotion_cnn_model.h5")

if __name__ == '__main__':
    # Check if data exists
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"ERROR: Data directory not found at {DATA_DIR}")
        print("Please download the FER-2013 dataset and extract it so that 'train' and 'test' folders are in 'backend/data/'")
        exit(1)

    train_model()
