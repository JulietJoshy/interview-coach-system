from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, Add, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy

def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Creates a Mini-Xception model for facial expression recognition.
    This architecture is lighter than VGG16 but very effective for FER-2013 (targeting ~66-70%+).
    """
    regularization = l2(0.01)
    
    # Input
    img_input = Input(input_shape)
    
    # Block 1 - Entry Flow
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Module 1 (Residual)
    processed = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)
    
    x = SeparableConv2D(16, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = Add()([x, processed])
    
    # Module 2 (Residual)
    processed = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)
    
    x = SeparableConv2D(32, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = Add()([x, processed])
    
    # Module 3 (Residual)
    processed = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)
    
    x = SeparableConv2D(64, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = Add()([x, processed])
    
    # Module 4 (Residual)
    processed = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)
    
    x = SeparableConv2D(128, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = Add()([x, processed])
    
    # Global Average Pooling & Output
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    
    model = Model(img_input, output)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_emotion_model_v2(input_shape=(48, 48, 1), num_classes=7):
    """
    V2 Mini-Xception model with wider filters, dropout regularization,
    dense classification head, and label smoothing. Targets ~70-73%+ on FER-2013.
    """
    regularization = l2(0.001)

    # Input
    img_input = Input(input_shape)

    # Block 1 - Entry Flow
    x = Conv2D(32, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Module 1 (Residual) - 64 filters
    processed = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)

    x = SeparableConv2D(64, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Add()([x, processed])
    x = Dropout(0.25)(x)

    # Module 2 (Residual) - 128 filters
    processed = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)

    x = SeparableConv2D(128, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Add()([x, processed])
    x = Dropout(0.25)(x)

    # Module 3 (Residual) - 256 filters
    processed = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)

    x = SeparableConv2D(256, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Add()([x, processed])
    x = Dropout(0.25)(x)

    # Module 4 (Residual) - 256 filters
    processed = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    processed = BatchNormalization()(processed)

    x = SeparableConv2D(256, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Add()([x, processed])
    x = Dropout(0.25)(x)

    # Dense classification head (replaces Conv2D→GAP→softmax)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, output)

    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    return model
