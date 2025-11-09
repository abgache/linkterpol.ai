import torch
from torch.nn import Module, Linear, ReLU, Sigmoid, Sequential

class neural_net():
    def __init__(self, logger, dnn_config, epochs=25):
        self.logger = logger
        self.input_size = dnn_config.get("input_size", 196608)  # Default to 256*256*3
        self.num_epochs = dnn_config.get("num_epochs", 100)
        self.batch_size = dnn_config.get("batch_size", 32)
        self.path = dnn_config.get("model_path", "model/model.pth")
        self.learning_rate = dnn_config.get("learning_rate", 0.001)
    
    def check_model_path(self):
        return os.path.isfile(self.model_path) and os.path.getsize(self.model_path) > 0

    def build_model(self):
        """
        Architecture:
        - Input Layer: 256*256*3 = 196608 neurons (for 256x256 RGB images)
        - Hidden Layer 1: Convolutional Layer with 2048 neurons + ReLU
        - Hidden Layer 2: Convolutional Layer with 1024 neurons + ReLU
        - Hidden Layer 3: Dense Layer with 1024 neurons + ReLU
        - Output Layer: Dense Layer with 1 neurons + Sigmoid

        Output: Probability distribution over 1 class : 0=Linkedin, 1=Interpol
        """
        if self.model is not None:
            self.logger.log("Model building skipped: Model already exists.", v=True, Wh=True, mention=False)
            return
        self.model = Sequential(
            Flatten(),                        # 256*256*3 = 196608
            Linear(196608, 2048),             # Hidden Layer 1
            ReLU(),
            Linear(2048, 1024),               # Hidden Layer 2
            ReLU(),
            Linear(1024, 1024),               # Hidden Layer 3
            ReLU(),
            Linear(1024, 1),                  # Output Layer
            Sigmoid()
        )
        self.logger.log("Neural network model built successfully.", v=True, Wh=True, mention=False)
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.path)
        self.logger.log(f"Model saved to {self.path}.", v=True, Wh=True, mention=False)
    def load_model(self):
        if not self.check_model_path():
            self.logger.log(f"Model loading failed: No model found at {self.path}.", v=False, Wh=True, mention=True)
            raise FileNotFoundError(f"Model loading failed: No model found at {self.path}.")
        self.build_model()
        self.model.load_state_dict(torch.load(self.path))
        self.model.eval()
        self.logger.log(f"Model loaded from {self.path}.", v=True, Wh=True, mention=False)
    
    def train_model(self, device, x_data, y_data):
        pass # To be implemented
    
    def predict(self, x_input):
        return [[1]] # To be implemented

    