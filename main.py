import json
import sys
import os
import torch
from scripts.time_log import time_log_module as tlm
from scripts.logger import logger
from data.data import data
from scripts.model import neural_net

# Environment variables
with open("config.json", "r") as f:
    config = json.load(f)
webhook_url = config.get("webhook_url", "") # leave empty to disable webhook logging
model_path = config.get("model_path", "model/model.pth")
version = config.get("version", "None")
examples = config.get("examples", 100)
dnn_config = config.get("dnn", {})
del config

train = "--train" in sys.argv or "-t" in sys.argv
predict = "--predict" in sys.argv
force_cpu = "--cpu" in sys.argv
force_cuda = "--cuda" in sys.argv
overwrite_data = "--overwrite-data" in sys.argv or "-od" in sys.argv

def get_arg_value(text, arg): # Ty Chatgpt
    parts = text.split()
    if arg in parts:
        i = parts.index(arg)
        if i + 1 < len(parts):
            return parts[i + 1]
    return None

if __name__ == "__main__":
    print(f"{tlm()} Start of program.")
    logger = logger(discord_webhook=webhook_url) # créer le logger

    # Logging system info
    logger.log(f"LinkePol.ai V{version}.", v=True, Wh=True, mention=False)
    logger.log(f"PyTorch version: {torch.__version__}", v=True, Wh=True, mention=False)
    logger.log(f"CUDA status : {str(torch.cuda.is_available())}", v=True, Wh=True, mention=False)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        msg = f"{count} GPU{'s' if count > 1 else ''} detected."
        logger.log(msg, v=True, Wh=True, mention=False)
        for i in range(count):
            logger.log(f" -> Device {i}: {torch.cuda.get_device_name(i)}", v=True, Wh=True, mention=False)
    
    if train:
        # Load data
        data = data(logger, overwrite_data=overwrite_data, examples=examples)
        if data.check_data_path(): # si ya pas de données
            data.build_data()      # on les crée
        else:
            logger.log("Data path check: Data already exists, skipping data building.", v=True, Wh=True, mention=False)
            data.load_data()
        x_data, y_data = data.dnn_ready_data()
        del data # pr la ram

        # Neural Network
        model = neural_net(logger, dnn_config, epochs=25)
        model.build_model()
        
        # Select device
        if force_cuda:
            if not torch.cuda.is_available(): # Si pas de GPU
                logger.log("CUDA forced but no GPU detected. Exiting.", v=False, Wh=True, mention=True)
                raise EnvironmentError(f"{tlm()} CUDA forced but no GPU detected. Exiting.")
            if force_cpu: # Si les deux sont forcés
                logger.log("Both --force-cuda and --force-cpu flags detected. Please choose only one.", v=False, Wh=True, mention=True)
                raise ValueError(f"{tlm()} Both --force-cuda and --force-cpu flags detected. Please choose only one.")
            logger.log("CUDA forced. Using GPU for computations.", v=True, Wh=True, mention=False)
            device = torch.device("cuda")
        elif force_cpu:
            logger.log("CPU forced. Using CPU for computations.", v=True, Wh=True, mention=False)
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.log(f"Using device: {device}.", v=True, Wh=True, mention=False)

        model.train_model(device, x_data, y_data)
        model.save_model()
    
    if predict:
        # Load model
        if model is None:
            model = neural_net(logger, dnn_config, epochs=25)
            if not model.check_model_path():
                logger.log("No trained model found. Please train the model using the --train argument before prediction.", v=False, Wh=True, mention=True)
                raise FileNotFoundError("No trained model found. Please train the model using the --train argument before prediction.")
            model.load_model()
        
        image_path = get_arg_value(" ".join(sys.argv), "--predict")

        if image_path is None or not os.path.isfile(image_path):
            logger.log("Invalid or missing image path for prediction. Please provide a valid path using the --predict <image_path> argument.", v=False, Wh=True, mention=True)
            raise ValueError("Invalid or missing image path for prediction. Please provide a valid path using the --predict <image_path> argument.")
        
        prediction = model.predict(image_path, device=device)
        if int(prediction >= 0.5):
            res = "Interpol"
        else:
            res = "Linkedin"
        logger.log(f"The model thinks this image is from {res}.", v=True, Wh=True, mention=False)
        logger.log(f"Prediction for image {image_path}: {prediction}", v=True, Wh=True, mention=False)

        
        