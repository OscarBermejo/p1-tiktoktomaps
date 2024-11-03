# setup_models.py
import easyocr
reader = easyocr.Reader(['en'], model_storage_directory='./models')
print("Models downloaded successfully!")