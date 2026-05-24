import cv2
import time
import os
import datetime
import zipfile
import logging
import winsound  # Pour le bip
from ultralytics import YOLO

# --- 1. CONFIGURATION DU LOGGING ---
logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)

# --- 2. INITIALISATION ---
logging.info("--- Début des variables et du modèle ---")
model = YOLO("yolov8n.pt")
port_camera = 1 
nom_archive_vides = "archives_vides.zip"
logging.info("--- Démarrage du programme de surveillance ---")

# On ouvre la caméra UNE SEULE FOIS pour éviter les erreurs de pilote
cap = cv2.VideoCapture(port_camera, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
try:
    while True:
       
        for _ in range(10):
            cap.grab()
            
        ret, frame = cap.read()
        
        if ret:
            horodatage = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            nom_jpg = f"temp_photo_{horodatage}.jpg"
            
            cv2.imwrite(nom_jpg, frame)
            logging.info(f"Capture réussie : {nom_jpg}")
            
            # Analyse IA
            results = model.predict(source=nom_jpg, conf=0.4, verbose=False)
            
            oiseau_present = False
            for r in results:
                for box in r.boxes:
                    if model.names[int(box.cls)] == "bird":
                        oiseau_present = True
                        break
            
            # Gestion des fichiers et Logs
            if oiseau_present:
                # --- LE BIP ---
                winsound.Beep(1000, 500) # Fréquence 1000Hz, Durée 500ms
                
                nom_final = f"OISEAU__{horodatage}.jpg"
                os.rename(nom_jpg, nom_final)
                msg = f"SUCCÈS : Oiseau détecté ! Image gardée sous {nom_final}"
                print(msg)
                logging.info(msg)
            else:
                try:
                    with zipfile.ZipFile(nom_archive_vides, 'a') as mon_zip:
                        mon_zip.write(nom_jpg)
                    os.remove(nom_jpg)
                    msg = "INFO : Rien détecté sur l'image. Archivée."
                    print(msg)
                    logging.info(msg)
                except Exception as e:
                    logging.error(f"Erreur ZIP : {e}")

        else:
            logging.error(f"ERREUR : Flux caméra perdu sur le port {port_camera}")
            # On tente de ré-ouvrir si perdu
            cap.release()
            cap = cv2.VideoCapture(port_camera, cv2.CAP_DSHOW)
        
        print("En attente de la prochaine analyse (15s)...")
        time.sleep(1)

finally:
    cap.release()
    logging.info("--- Fin de la session de capture ---")
