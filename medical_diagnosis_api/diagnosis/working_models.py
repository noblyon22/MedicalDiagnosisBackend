# Working Hugging Face Model Downloader
import os
import requests
import logging

logger = logging.getLogger(__name__)

def download_model_from_huggingface(url, destination, filename):
    """Download model from Hugging Face with proper redirect handling"""
    try:
        logger.info(f"Downloading {filename} from Hugging Face...")
        
        # Proper headers for Hugging Face
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Handle redirects properly
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, stream=True, timeout=300, allow_redirects=True)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            # Verify download
            if os.path.exists(destination) and os.path.getsize(destination) > 10000000:  # At least 10MB
                file_size = os.path.getsize(destination) / (1024 * 1024)
                logger.info(f"SUCCESS: {filename} downloaded ({file_size:.1f}MB)")
                return True
            else:
                logger.error(f"Downloaded file is too small")
                return False
                
        else:
            logger.error(f"Download failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False

def download_models():
    """Download ML models from Hugging Face repository"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Direct download URLs for your Hugging Face models
    models = {
        'medical_tb_detector.h5': 'https://huggingface.co/Noblhyon/medical-diagnosis-models/resolve/main/medical_tb_detector.h5',
        'best_precision_model_phase1.h5': 'https://huggingface.co/Noblhyon/medical-diagnosis-models/resolve/main/best_precision_model_phase1.h5'
    }
    
    success_count = 0
    for model_name, url in models.items():
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path):
            success = download_model_from_huggingface(url, model_path, model_name)
            if success:
                success_count += 1
        else:
            logger.info(f"Model already exists: {model_name}")
            success_count += 1
    
    logger.info(f"Model download summary: {success_count}/{len(models)} models available")
    return models_dir

# Legacy compatibility
def download_from_gdrive(file_id, destination):
    """Legacy compatibility function"""
    filename = os.path.basename(destination)
    
    if 'tb_detector' in filename.lower():
        url = 'https://huggingface.co/Noblhyon/medical-diagnosis-models/resolve/main/medical_tb_detector.h5'
    elif 'precision_model' in filename.lower():
        url = 'https://huggingface.co/Noblhyon/medical-diagnosis-models/resolve/main/best_precision_model_phase1.h5'
    else:
        return False
    
    return download_model_from_huggingface(url, destination, filename)
