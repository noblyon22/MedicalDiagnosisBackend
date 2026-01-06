# Google Drive Model Integration 
import os 
import requests 
import logging 
 
logger = logging.getLogger(__name__) 
 
def download_from_gdrive(file_id, destination): 
    """Download file from Google Drive using file ID""" 
    url = f"https://drive.google.com/uc?id={file_id}&export=download" 
    try: 
        logger.info(f"Downloading from Google Drive: {file_id}") 
        response = requests.get(url, stream=True, timeout=300) 
        if response.status_code == 200: 
            os.makedirs(os.path.dirname(destination), exist_ok=True) 
            with open(destination, 'wb') as f: 
                for chunk in response.iter_content(chunk_size=8192): 
                    if chunk: 
                        f.write(chunk) 
            logger.info(f"✓ Downloaded successfully: {os.path.basename(destination)}") 
            return True 
        else: 
            logger.error(f"Download failed with status: {response.status_code}") 
            return False 
    except Exception as e: 
        logger.error(f"Download error: {e}") 
        return False 
 
def download_models(): 
    """Download ML models from Google Drive if not present locally""" 
    models_dir = os.path.join(os.path.dirname(__file__), 'models') 
    os.makedirs(models_dir, exist_ok=True) 
 
    models = { 
        'medical_tb_detector.h5': '1835T27ZjBp0V9w-WqVRv4phGetfg3Mro', 
        'best_precision_model_phase1.h5': '1snTZHWvMm9zhRQkr_sYGjZxZo_C5CZkf' 
    } 
 
    for model_name, file_id in models.items(): 
        model_path = os.path.join(models_dir, model_name) 
        if not os.path.exists(model_path): 
            logger.info(f"Downloading {model_name} from Google Drive...") 
            success = download_from_gdrive(file_id, model_path) 
            if success: 
                logger.info(f"✓ {model_name} ready for use") 
            else: 
                logger.error(f"❌ Failed to download {model_name}") 
        else: 
            logger.info(f"✓ {model_name} already exists") 
    return models_dir 
