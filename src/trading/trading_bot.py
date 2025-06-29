import os
import numpy as np
import pandas as pd
import torch
from src.data.process_data import DataProcessor
from src.models.inception_model import InceptionTradingModel

class TradingBot:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.model = self.load_model()
        self.data_processor = DataProcessor(data_path)
        
    def load_model(self):
        """Load the trained Inception model"""
        model_builder = InceptionTradingModel()
        model = model_builder.build_model()
        model.load_state_dict(torch.load(os.path.join(self.model_path, 'inception_trading_model.pth')))
        model.eval()
        return model
        
    def predict(self, current_data):
        """Make a prediction using the trained model"""
        # Process the current data
        processed_data = self.data_processor.create_image_data(current_data)
        
        # Convert to torch tensor and add batch dimension
        input_tensor = torch.from_numpy(processed_data).float()
        input_tensor = input_tensor.unsqueeze(0)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, prediction = torch.max(outputs, 1)
            
        return prediction.item()
        
    def execute_trade(self, prediction):
        """Execute trading based on model prediction"""
        # Implement your trading logic here
        # 0: Hold, 1: Buy, 2: Sell (example)
        if prediction == 1:
            print("Executing buy order...")
        elif prediction == 2:
            print("Executing sell order...")
        
    def run(self):
        """Main trading loop"""
        while True:
            # Get current market data
            current_data = self.data_processor.load_data()
            
            # Make prediction
            prediction = self.predict(current_data)
            
            # Execute trade based on prediction
            self.execute_trade(prediction)
            
            # Add delay between trades
            time.sleep(60)  # 1 minute delay
