import os
import numpy as np
import logging
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models from Hugging Face
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'diagnosis', 'models')

tb_model = None
leukemia_model = None

def download_models_from_huggingface():
    """Download models from Hugging Face if not present locally"""
    try:
        from .working_models import download_models
        download_models()
        return True
    except Exception as e:
        logger.error(f"Failed to download models from Hugging Face: {str(e)}")
        return False

try:
    from tensorflow.keras.models import load_model
    
    # Try to download models from Hugging Face first
    download_models_from_huggingface()
    
    # Try to load TB model
    tb_model_path = os.path.join(MODELS_DIR, 'medical_tb_detector.h5')
    if os.path.exists(tb_model_path):
        tb_model = load_model(tb_model_path)
        logger.info("TB model loaded successfully from Hugging Face")
    else:
        logger.warning("TB model not available - downloading from Hugging Face...")
        
    # Try to load Leukemia model
    leukemia_model_path = os.path.join(MODELS_DIR, 'best_precision_model_phase1.h5')
    if os.path.exists(leukemia_model_path):
        leukemia_model = load_model(leukemia_model_path)
        logger.info("Leukemia model loaded successfully from Hugging Face")
    else:
        logger.warning("Leukemia model not available - downloading from Hugging Face...")
        
except Exception as e:
    logger.error(f"Could not load ML models: {e}")
    logger.info("Models will be downloaded from Hugging Face on first use")

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CONFIDENCE_THRESHOLD_HIGH = 0.9
CONFIDENCE_THRESHOLD_LOW = 0.7

def validate_image_upload(image_file):
    """Validate uploaded image file"""
    errors = []
    
    # Check file size
    if image_file.size > MAX_FILE_SIZE:
        errors.append("File size too large. Maximum 10MB allowed.")
    
    # Check file type - accept any image format
    if not image_file.content_type.startswith('image/'):
        errors.append("Invalid file type. Only image files are allowed.")
    
    # Try to open and validate image
    try:
        img = Image.open(io.BytesIO(image_file.read()))
        image_file.seek(0)  # Reset file pointer
        
        # Check image dimensions
        if img.size[0] < 50 or img.size[1] < 50:
            errors.append("Image too small. Minimum 50x50 pixels required.")
            
        if img.size[0] > 5000 or img.size[1] > 5000:
            errors.append("Image too large. Maximum 5000x5000 pixels allowed.")
            
    except Exception as e:
        errors.append("Invalid or corrupted image file.")
        logger.error(f"Image validation error: {str(e)}")
    
    return errors

def interpret_confidence(confidence, disease_type):
    """Convert raw confidence to interpretable risk assessment"""
    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        return {
            "risk_level": "High",
            "interpretation": f"Strong indication of {disease_type}. Immediate medical consultation recommended.",
            "reliability": "High confidence"
        }
    elif confidence >= CONFIDENCE_THRESHOLD_LOW:
        return {
            "risk_level": "Moderate", 
            "interpretation": f"Possible indication of {disease_type}. Medical evaluation recommended.",
            "reliability": "Moderate confidence"
        }
    else:
        return {
            "risk_level": "Low",
            "interpretation": f"Low probability of {disease_type}. Continue preventive measures.",
            "reliability": "Low confidence - consider retesting with higher quality image"
        }

# TB Recommendations
def get_tb_recommendations(is_positive):
    if is_positive:
        return {
            "first_aid": [
                "Seek immediate medical attention - TB requires professional treatment",
                "Isolate yourself to prevent spreading the infection",
                "Wear a mask when around others",
                "Cover your mouth and nose when coughing or sneezing",
                "Ensure good ventilation in living spaces",
                "Take prescribed medications exactly as directed by healthcare provider",
                "Get plenty of rest and maintain good nutrition"
            ],
            "urgent_actions": [
                "Contact your healthcare provider immediately",
                "Start anti-TB treatment as prescribed",
                "Inform close contacts so they can get tested",
                "Follow up regularly with your doctor"
            ]
        }
    else:
        return {
            "prevention": [
                "Maintain a strong immune system with proper nutrition",
                "Get adequate sleep and exercise regularly",
                "Avoid close contact with people who have active TB",
                "Ensure good ventilation in living and working spaces",
                "Consider TB vaccination (BCG) if recommended by your doctor",
                "Practice good hygiene - wash hands frequently",
                "Avoid smoking and excessive alcohol consumption",
                "Get regular health check-ups"
            ],
            "lifestyle_tips": [
                "Eat a balanced diet rich in vitamins and minerals",
                "Manage stress effectively",
                "Maintain good indoor air quality",
                "Stay up to date with routine medical screenings"
            ]
        }

# Leukemia Recommendations
def get_leukemia_recommendations(is_positive):
    if is_positive:
        return {
            "first_aid": [
                "Seek immediate medical attention - leukemia requires urgent professional care",
                "Avoid exposure to infections - your immune system may be compromised",
                "Monitor for signs of bleeding or bruising",
                "Stay hydrated and maintain nutrition",
                "Avoid contact sports or activities that could cause injury",
                "Take temperature regularly and report fever immediately",
                "Follow all medical appointments and treatment plans"
            ],
            "urgent_actions": [
                "Contact an oncologist or hematologist immediately",
                "Go to the emergency room if experiencing severe symptoms",
                "Prepare for possible hospitalization",
                "Inform family members about potential genetic factors"
            ]
        }
    else:
        return {
            "prevention": [
                "Maintain a healthy lifestyle with regular exercise",
                "Eat a diet rich in fruits, vegetables, and antioxidants",
                "Avoid exposure to harmful chemicals and radiation",
                "Don't smoke and limit alcohol consumption",
                "Maintain a healthy weight",
                "Get regular blood tests as recommended by your doctor",
                "Manage stress through relaxation techniques",
                "Stay up to date with vaccinations"
            ],
            "lifestyle_tips": [
                "Include foods rich in folate, vitamin C, and antioxidants",
                "Exercise regularly to boost immune function",
                "Practice good hygiene to prevent infections",
                "Get adequate sleep (7-9 hours per night)",
                "Consider genetic counseling if there's family history"
            ]
        }

@api_view(['POST'])
def tb_detection(request):
    """
    TB Detection API with robust validation and decision logic
    """
    # Log API usage (without storing personal data)
    logger.info(f"TB detection request from IP: {request.META.get('REMOTE_ADDR', 'unknown')}")
    
    if tb_model is None:
        logger.error("TB model not loaded")
        return Response(
            {
                "error": "TB detection service temporarily unavailable",
                "message": "ML models are being downloaded from Hugging Face. Please try again in a few minutes.",
                "status": "models_downloading",
                "instructions": "Models are being downloaded from Hugging Face repository"
            }, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    if 'image' not in request.FILES:
        return Response(
            {
                "error": "No image file provided",
                "message": "Please upload a chest X-ray image",
                "accepted_formats": ["Any image format"],
                "max_size": "10MB"
            }, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        img_file = request.FILES['image']
        
        # Validate image upload
        validation_errors = validate_image_upload(img_file)
        if validation_errors:
            return Response(
                {
                    "error": "Invalid image upload",
                    "details": validation_errors,
                    "message": "Please upload a valid chest X-ray image"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Process the uploaded image
        img = Image.open(io.BytesIO(img_file.read()))
        
        # Convert to grayscale for TB detection (chest X-rays are typically grayscale)
        if img.mode != 'L':
            img = img.convert('L')  # Convert to grayscale
        
        # Preprocess image for model
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        
        # For grayscale, we need to expand to 3 channels if model expects RGB input
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
            
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        prediction = tb_model.predict(img_array)
        confidence = float(prediction[0][0])
        
        # Apply decision logic
        risk_assessment = interpret_confidence(confidence, "Tuberculosis")
        is_positive = confidence > CONFIDENCE_THRESHOLD_LOW
        
        # Get recommendations
        recommendations = get_tb_recommendations(is_positive)
        
        # Build response
        response_data = {
            "result": {
                "detection": "TB Detected" if is_positive else "No TB Detected",
                "confidence_score": round(confidence, 3),
                "risk_assessment": risk_assessment
            },
            "recommendations": recommendations,
            "next_steps": [
                "Consult with a qualified healthcare provider",
                "This is a screening tool, not a diagnostic device",
                "Professional medical evaluation is required for confirmation"
            ],
            "disclaimer": "This AI screening tool is for educational and preliminary assessment purposes only. It does not replace professional medical diagnosis, treatment, or advice. Always consult qualified healthcare providers for medical decisions.",
            "timestamp": request.META.get('HTTP_DATE', 'Not provided')
        }
        
        # Log successful prediction (without storing image data)
        logger.info(f"TB prediction completed - Risk: {risk_assessment['risk_level']}, Confidence: {confidence:.3f}")
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        # Log error without exposing internal details
        logger.error(f"TB detection error: {str(e)}")
        return Response(
            {
                "error": "Processing failed",
                "message": "Unable to process the image. Please ensure it's a clear chest X-ray and try again.",
                "support": "If the problem persists, please contact technical support"
            }, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def leukemia_detection(request):
    """
    Leukemia Detection API with robust validation and decision logic
    Expects color blood smear microscopy images
    """
    # Log API usage
    logger.info(f"Leukemia detection request from IP: {request.META.get('REMOTE_ADDR', 'unknown')}")
    
    if leukemia_model is None:
        logger.error("Leukemia model not loaded")
        return Response(
            {
                "error": "Leukemia detection service temporarily unavailable",
                "message": "ML models are being downloaded from Hugging Face. Please try again in a few minutes.",
                "status": "models_downloading",
                "instructions": "Models are being downloaded from Hugging Face repository"
            }, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    if 'image' not in request.FILES:
        return Response(
            {
                "error": "No image file provided",
                "message": "Please upload a blood smear microscopy image",
                "accepted_formats": ["Any image format"],
                "max_size": "10MB"
            }, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Process the uploaded image
        img_file = request.FILES['image']
        
        # Validate image upload
        validation_errors = validate_image_upload(img_file)
        if validation_errors:
            return Response(
                {
                    "error": "Invalid image upload",
                    "details": validation_errors,
                    "message": "Please upload a valid blood smear microscopy image"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        img = Image.open(io.BytesIO(img_file.read()))
        
        # Convert to RGB for leukemia detection (blood smear images are color)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess image for model
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        prediction = leukemia_model.predict(img_array)
        confidence = float(prediction[0][0])
        
        # Apply decision logic
        risk_assessment = interpret_confidence(confidence, "Leukemia")
        is_positive = confidence > CONFIDENCE_THRESHOLD_LOW
        
        # Get recommendations
        recommendations = get_leukemia_recommendations(is_positive)
        
        # Build response
        response_data = {
            "result": {
                "detection": "Leukemia Detected" if is_positive else "No Leukemia Detected",
                "confidence_score": round(confidence, 3),
                "risk_assessment": risk_assessment
            },
            "recommendations": recommendations,
            "next_steps": [
                "Consult with a qualified hematologist or oncologist",
                "This is a screening tool for blood smear analysis, not a diagnostic device",
                "Professional medical evaluation and additional tests are required for confirmation"
            ],
            "disclaimer": "This AI screening tool is for educational and preliminary assessment purposes only. It does not replace professional medical diagnosis, treatment, or advice. Always consult qualified healthcare providers for medical decisions.",
            "timestamp": request.META.get('HTTP_DATE', 'Not provided')
        }
        
        # Log successful prediction
        logger.info(f"Leukemia prediction completed - Risk: {risk_assessment['risk_level']}, Confidence: {confidence:.3f}")
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        # Log error without exposing internal details
        logger.error(f"Leukemia detection error: {str(e)}")
        return Response(
            {
                "error": "Processing failed",
                "message": "Unable to process the blood smear image. Please ensure it's a clear microscopy image and try again.",
                "support": "If the problem persists, please contact technical support"
            }, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def health_check(request):
    return Response({
        "status": "healthy",
        "tb_model_loaded": tb_model is not None,
        "leukemia_model_loaded": leukemia_model is not None,
        "models_status": {
            "tb_model": "loaded" if tb_model is not None else "downloading from Hugging Face - medical_tb_detector.h5",
            "leukemia_model": "loaded" if leukemia_model is not None else "downloading from Hugging Face - best_precision_model_phase1.h5"
        },
        "model_source": "Hugging Face Repository: https://huggingface.co/Noblhyon/medical-diagnosis-models"
    })

def index_view(request):
    """Serve the main frontend interface"""
    from django.shortcuts import render
    return render(request, 'diagnosis/index.html')
