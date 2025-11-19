from scripts.time_log import time_log_module as tlm
import os
import asyncio
import aiohttp
from PIL import Image
import random, time

class WebScraper():
    def __init__(self, logger, examples=100):
        self.logger = logger
        self.examples = examples
        self.api = "https://linkedin-or-interpol.com/api/random-person"
        self.api_check = "https://linkedin-or-interpol.com/api/check-answer"

    async def create_example(self, session):
        async with session.get(self.api) as response:
            if response.status != 200:
                self.logger.log("WebScraper: Failed to retrieve data from thebrainfox API.", v=False, Wh=True, mention=True)
                raise ConnectionError("WebScraper: Failed to retrieve data from thebrainfox API.")
            person_data = await response.json()

        image_url = f"https://linkedin-or-interpol.com/{person_data['photoUrl']}"

        async with session.post(self.api_check, json={"personId": person_data["id"], "userChoice": "linkedin"}) as check_response:
            if check_response.status != 200:
                self.logger.log("WebScraper: Failed to check answer from thebrainfox API.", v=False, Wh=True, mention=True)
                raise ConnectionError("WebScraper: Failed to check answer from thebrainfox API.")
            result_data = await check_response.json()

        save_dir = "data/linkedin" if result_data.get("correct", "") == True else "data/interpol"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{person_data['id']}.png"

        await self.webp2png(image_url, save_path)

    async def webp2png(self, url, save_path):
        from PIL import Image
        from io import BytesIO

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    img_data = await resp.read()
                    img = Image.open(BytesIO(img_data)).convert("RGBA")
                    img.save(save_path, "PNG")

    async def build_data(self):
        self.logger.log(f"WebScraper: Starting data building with {self.examples} examples.", v=True, Wh=True, mention=False)
        async with aiohttp.ClientSession() as session:
            tasks = [self.create_example(session) for _ in range(self.examples)]
            await asyncio.gather(*tasks)
        self.logger.log(f"WebScraper: Data building completed.", v=True, Wh=True, mention=False)
    

class data():
    def __init__(self, logger, overwrite_data, examples=100):
        self.logger = logger
        self.data_path_status = False
        self.data_loaded_status = False
        self.overwrite_data = overwrite_data
        self.examples = examples
        self.data = [[], []]  # [interpol_data[], linkedin_data[]]
    
    def __add__(self, other): # Kinda useless for now, maybe later for data augmentation
        return NotImplemented
    
    def check_data_path(self):
        interpol_status = len(os.listdir("data/interpol")) > 0
        linkedin_status = len(os.listdir("data/linkedin")) > 0
        self.data_path_status = interpol_status and linkedin_status
        return self.data_path_status
    
    def check_data(self):
        self.data_loaded_status = self.data_path_status and len(self.data[0])>0 and len(self.data[1])>0
        return self.data_loaded_status
    
    def load_data(self):
        if not self.check_data_path():
            self.logger.log("Data loading failed: Data folders are missing or incompatible.", v=False, Wh=True, mention=True)
            raise FileNotFoundError("Data loading failed: Data folders are missing or incompatible.")

        # Load interpol data
        self.data[0] = [os.path.join("data/interpol", f) for f in sorted(os.listdir("data/interpol"))]

        # Load linkedin data
        self.data[1] = [os.path.join("data/linkedin", f) for f in sorted(os.listdir("data/linkedin"))]

        self.check_data()
        if not self.data_loaded_status:
            self.logger.log("Data loading failed: Data could not be loaded properly.", v=False, Wh=True, mention=True)
            raise ValueError("Data loading failed: Data could not be loaded properly.")

        return self.data

    async def build_data(self):
        self.ws = WebScraper(self.logger, self.examples)
        await self.ws.build_data()  # juste await, pas asyncio.run
        # Attendre que les dossiers soient créés
        while not self.check_data_path():
            await asyncio.sleep(1)
        self.load_data()
    
        
    def dnn_ready_data(self): # 256*256 pixels images, 256*256*3=196608 input neurons
        if not self.check_data():
            self.logger.log("DNN ready data preparation failed: Data not loaded.", v=False, Wh=True, mention=True)
            raise ValueError("DNN ready data preparation failed: Data not loaded.")
        
        # 3 steps : Load INTERPOL images, Load LINKEDIN images, Randomly mix them
        # [0] = INTERPOL, [1] = LINKEDIN
        # Interpol images:
        interpol_x = []
        interpol_y = []
        for file_path in self.data[0]:
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize((256, 256))
                img_data = list(img.getdata())
                img_flat = [value for pixel in img_data for value in pixel]  # Flatten the list
                interpol_x.append(img_flat)
                interpol_y.append([0])  # INTERPOL label
            except Exception as e:
                self.logger.log(f"DNN ready data preparation warning: Failed to process {file_path} in Interpol. Error: {e}", v=False, Wh=True, mention=False)
                continue

        # Linkedin images:
        linkedin_x = []
        linkedin_y = []
        for file_path in self.data[1]:
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize((256, 256))
                img_data = list(img.getdata())
                img_flat = [value for pixel in img_data for value in pixel]  # Flatten the list
                linkedin_x.append(img_flat)
                linkedin_y.append([1])  # linkedin label
            except Exception as e:
                self.logger.log(f"DNN ready data preparation warning: Failed to process {file_path} in Linkedin. Error: {e}", v=False, Wh=True, mention=False)
                continue

        # Randomly mix data
        x_data = list(list(interpol_x) + list(linkedin_x))
        y_data = list(list(interpol_y) + list(linkedin_y))
        random.shuffle(x_data)
        random.shuffle(y_data)
        del interpol_x, interpol_y, linkedin_x, linkedin_y  # Free memory

        # Return data
        if not len(x_data) == len(y_data):
            self.logger.log("DNN ready data preparation failed: Data size mismatch.", v=False, Wh=True, mention=True)
            raise ValueError("DNN ready data preparation failed: Data size mismatch.")
        
        self.logger.log(f"- [DEBUG] x_data {str(x_data[0])}.", v=True, Wh=True, mention=False)
        
        return (x_data, y_data)
            