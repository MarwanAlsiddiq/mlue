import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# PARAMETERS
img_height, img_width = 150, 150
batch_size = 32
epochs = 10
num_classes = 3  # set to 3 for up, down, neutral

# DATA PATHS
train_data_dir = r'X:/stone/data/processed'
val_data_dir = r'X:/data/val'

# DATA GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# LOAD & CUSTOMIZE INCEPTIONV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# FREEZE BASE MODEL LAYERS
for layer in base_model.layers:
    layer.trainable = False

# COMPILE & TRAIN
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# SAVE MODEL
os.makedirs('../models', exist_ok=True)
model.save('../models/inceptionv3_custom.h5')
print("Model saved at ../models/inceptionv3_custom.h5")