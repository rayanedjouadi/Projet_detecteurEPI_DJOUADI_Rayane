from roboflow import Roboflow
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- PARTIE 1 : TÉLÉCHARGEMENT ---

# On initialise Roboflow avec notre clé API
rf = Roboflow(api_key="vQukIDlkPAM2zlSrwqwI")
project = rf.workspace("testcasque").project("ppe-detection-qlq3d")
version = project.version(1)
dataset = version.download("yolov8")
                

# --- PARTIE 2 : ENTRAÎNEMENT ---
# On charge le modèle de base pour le spécialiser
model = YOLO('yolov8n.pt') 

# On lance l'entraînement
# 'data' pointe vers le fichier .yaml téléchargé par Roboflow
model.train(data=f"{dataset.location}/data.yaml", epochs=10, imgsz=640)