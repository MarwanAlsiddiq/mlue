import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(3):
        x = residual_block(x, 64)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# Load the image data
btc_image_data = np.load("data/processed/btc_image_data.npy")
gala_image_data = np.load("data/processed/gala_image_data.npy")
image_data = np.concatenate((btc_image_data, gala_image_data), axis=0)

# Example labels (replace with your actual labels)
labels = np.array([...])  # Ensure labels match the number of samples

# Split the data
X_train, X_val, y_train, y_val = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Define input shape and number of classes
input_shape = (15, 15, 1)  # Adjust based on your data
num_classes = 3  # Example: Buy, Sell, Hold

# Load an existing model or build a new one
try:
    model = load_model("models/stone_jr.h5")
    print("Loaded existing model: stone_jr.h5")
except Exception as e:
    print(f"Error loading model: {e}. Building a new model.")
    model = build_resnet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (if needed)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the model
model.save("models/resnet_model.h5")