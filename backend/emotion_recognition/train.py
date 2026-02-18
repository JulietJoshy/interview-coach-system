import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import create_emotion_model

# Configuration
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7

# Path to backend/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')

def train_model():
    # Data Augmentation and Normalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
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

    # Create Model
    model = create_emotion_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)
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
        patience=10,
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
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    print("Training Complete. Model saved as emotion_cnn_model.h5")

if __name__ == '__main__':
    # Check if data exists
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print(f"ERROR: Data directory not found at {DATA_DIR}")
        print("Please download the FER-2013 dataset and extract it so that 'train' and 'test' folders are in 'backend/data/'")
        exit(1)
        
    train_model()
