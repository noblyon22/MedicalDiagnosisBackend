# Google Drive Model Integration 
import os 
import requests 
import logging 
import re
 
logger = logging.getLogger(__name__) 
 
def download_from_gdrive(file_id, destination): 
    """Download file from Google Drive using file ID with proper large file handling""" 
    session = requests.Session()
    
    try: 
        logger.info(f"Downloading from Google Drive: {file_id}") 
        
        # First request to get the download page
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        response = session.get(url, stream=True, timeout=60)
        
        # Check if we need to handle virus scan warning (for large files)
        if "virus scan warning" in response.text.lower() or "download_warning" in response.text:
            logger.info("Large file detected, handling virus scan warning...")
            
            # Extract the confirm token from the response
            confirm_token = None
            
            # Method 1: Look for confirm token in cookies
            for cookie in session.cookies:
                if cookie.name.startswith('download_warning'):
                    confirm_token = cookie.value
                    break
            
            # Method 2: Extract from HTML form if not in cookies
            if not confirm_token:
                confirm_match = re.search(r'name="confirm" value="([^"]+)"', response.text)
                if confirm_match:
                    confirm_token = confirm_match.group(1)
            
            # Method 3: Look for UUID pattern in response
            if not confirm_token:
                uuid_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', response.text)
                if uuid_match:
                    confirm_token = uuid_match.group(1)
            
            if confirm_token:
                logger.info(f"Found confirm token: {confirm_token[:10]}...")
                # Make the confirmed download request
                confirmed_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                response = session.get(confirmed_url, stream=True, timeout=300)
            else:
                logger.warning("Could not find confirm token, trying direct download...")
        
        # Check if we got a valid response
        if response.status_code == 200:
            # Verify we're getting actual file content, not an error page
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type and response.headers.get('content-length', '0') == '0':
                logger.error("Received HTML page instead of file - download may have failed")
                return False
            
            # Create directory and download file
            os.makedirs(os.path.dirname(destination), exist_ok=True) 
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f: 
                for chunk in response.iter_content(chunk_size=32768):  # Larger chunks for better performance
                    if chunk: 
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded / (1024*1024):.1f}MB)")
            
            # Verify file was downloaded successfully
            if os.path.exists(destination) and os.path.getsize(destination) > 1000:  # At least 1KB
                file_size_mb = os.path.getsize(destination) / (1024 * 1024)
                logger.info(f"✓ Downloaded successfully: {os.path.basename(destination)} ({file_size_mb:.1f}MB)") 
                return True
            else:
                logger.error(f"Download failed - file is too small or doesn't exist")
                if os.path.exists(destination):
                    os.remove(destination)
                return False
                
        else: 
            logger.error(f"Download failed with status: {response.status_code}")
            logger.error(f"Response headers: {dict(response.headers)}")
            return False 
            
    except Exception as e: 
        logger.error(f"Download error: {e}") 
        return False
    finally:
        session.close() 
 
def download_models(): 
    """Download ML models from Google Drive if not present locally""" 
    models_dir = os.path.join(os.path.dirname(__file__), 'models') 
    os.makedirs(models_dir, exist_ok=True) 
 
    models = { 
        'medical_tb_detector.h5': '1835T27ZjBp0V9w-WqVRv4phGetfg3Mro', 
        'best_precision_model_phase1.h5': '1snTZHWvMm9zhRQkr_sYGjZxZo_C5CZkf' 
    } 
 
    success_count = 0
    for model_name, file_id in models.items(): 
        model_path = os.path.join(models_dir, model_name) 
        if not os.path.exists(model_path): 
            logger.info(f"Downloading {model_name} from Google Drive...") 
            success = download_from_gdrive(file_id, model_path) 
            if success: 
                logger.info(f"✓ {model_name} ready for use") 
                success_count += 1
            else: 
                logger.error(f"❌ Failed to download {model_name}") 
        else: 
            logger.info(f"✓ {model_name} already exists") 
            success_count += 1
    
    logger.info(f"Model download summary: {success_count}/{len(models)} models available")
    return models_dir 
