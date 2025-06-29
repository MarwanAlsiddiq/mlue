import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from src.data.process_data import DataProcessor
from src.models.inception_model import InceptionTradingModel

def train_model(data_path, model_path):
    # Initialize data processor
    processor = DataProcessor(data_path)
    
    # Load and prepare data
    data = processor.load_data()
    train_images, test_images = processor.prepare_dataset(data)
    
    # Initialize model
    model_builder = InceptionTradingModel()
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    # Create checkpoint callback
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_path, 'inception_trading_model.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        train_images,
        epochs=50,
        validation_data=test_images,
        callbacks=[checkpoint]
    )
    
    return history
