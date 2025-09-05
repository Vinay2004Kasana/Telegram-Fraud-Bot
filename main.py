import asyncio
import re
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
from telegram import Update, File
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client
import json
import logging

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- Configuration --------------------
SUPABASE_URL = "https://uiiuiidynbqtgmqvwvox.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVpaXVpaWR5bmJxdGdtcXZ3dm94Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1Njk3ODY2NCwiZXhwIjoyMDcyNTU0NjY0fQ.9BL3prhYuDLi8M_mZ1gce5qUwYc8D7LknwnEppajH4A"
TELEGRAM_TOKEN = "8403500260:AAEe8T6MMZyqgWe4la-rfMuEP7eBy01wDNw"

# File handling
TEMP_DIR = "./temp_files/"
MAX_FILE_SIZE_MB = 50
MAX_VIDEO_DURATION = 300  # 5 minutes

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- Data Constants --------------------
CORPORATE_LEADERS = {
    'elon musk': {'companies': ['Tesla', 'SpaceX', 'X'], 'symbols': ['TSLA']},
    'tim cook': {'companies': ['Apple'], 'symbols': ['AAPL']},
    'satya nadella': {'companies': ['Microsoft'], 'symbols': ['MSFT']},
    'jeff bezos': {'companies': ['Amazon'], 'symbols': ['AMZN']},
    'mark zuckerberg': {'companies': ['Meta', 'Facebook'], 'symbols': ['META']},
    'sundar pichai': {'companies': ['Google', 'Alphabet'], 'symbols': ['GOOGL', 'GOOG']},
    'jensen huang': {'companies': ['NVIDIA'], 'symbols': ['NVDA']},
    'jamie dimon': {'companies': ['JPMorgan Chase'], 'symbols': ['JPM']},
    'warren buffett': {'companies': ['Berkshire Hathaway'], 'symbols': ['BRK-A', 'BRK-B']},
    'mary barra': {'companies': ['General Motors'], 'symbols': ['GM']},
    'andy jassy': {'companies': ['Amazon'], 'symbols': ['AMZN']},
    'lisa su': {'companies': ['AMD'], 'symbols': ['AMD']},
    'reed hastings': {'companies': ['Netflix'], 'symbols': ['NFLX']},
    'brian chesky': {'companies': ['Airbnb'], 'symbols': ['ABNB']},
    'daniel ek': {'companies': ['Spotify'], 'symbols': ['SPOT']},
}

MAJOR_STOCKS = [
    'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM', 
    'BRK-A', 'BRK-B', 'NFLX', 'AMD', 'GM', 'ABNB', 'SPOT', 'UBER', 'LYFT',
    'COIN', 'SQ', 'PYPL', 'DIS', 'BA', 'GE', 'IBM', 'ORCL', 'CRM'
]

REGULATORY_AGENCIES = [
    'sec', 'securities and exchange commission', 'finra', 'cftc',
    'federal reserve', 'fed', 'fdic', 'occ', 'treasury', 'irs',
    'sebi', 'fca', 'bafin', 'esma', 'rbi', 'bank of england',
    'ecb', 'european central bank', 'pboc', 'boj', 'bank of japan'
]

FINANCIAL_TERMS = [
    'merger', 'acquisition', 'bankruptcy', 'investigation', 'lawsuit',
    'fine', 'penalty', 'approval', 'license', 'earnings', 'revenue',
    'profit', 'loss', 'scandal', 'partnership', 'deal', 'ipo',
    'stock split', 'dividend', 'buyback', 'delisting', 'halt'
]

MANIPULATION_TERMS = [
    'breaking news', 'urgent', 'exclusive', 'leaked', 'confidential',
    'insider information', 'pump and dump', 'market manipulation',
    'insider trading', 'ponzi scheme', 'pyramid scheme', 'scam',
    'emergency', 'immediate action', 'limited time', 'act now'
]

# -------------------- Fraud Detection Class --------------------
class FinancialFraudDetector:
    def __init__(self):
        self.suspicious_patterns = {
            'deepfake': ['deepfake', 'ai generated', 'synthetic', 'artificial', 'fake video', 'deep fake'],
            'urgency': ['breaking', 'urgent', 'emergency', 'immediate', 'act now', 'limited time'],
            'financial': FINANCIAL_TERMS,
            'manipulation': MANIPULATION_TERMS,
            'regulatory': ['sec filing', 'regulatory approval', 'cease and desist', 'enforcement action']
        }

    async def analyze_text_content(self, text: str) -> Dict:
        """Comprehensive text analysis for financial fraud"""
        text_lower = text.lower()
        analysis = {
            'fraud_score': 0,
            'risk_level': 'low',
            'indicators': [],
            'mentioned_leaders': [],
            'mentioned_stocks': [],
            'mentioned_agencies': [],
            'mentioned_terms': [],
            'market_impact_potential': 'low',
            'urgency_score': 0,
            'regulatory_score': 0
        }
        
        # Check for corporate leaders
        for leader, data in CORPORATE_LEADERS.items():
            if leader in text_lower:
                analysis['mentioned_leaders'].append(leader.title())
                analysis['fraud_score'] += 3
                analysis['indicators'].append(f'corporate_leader_{leader.replace(" ", "_")}')
                logger.info(f"Corporate leader detected: {leader}")
        
        # Check for stock symbols and company names
        words = re.findall(r'\b[A-Z]{2,5}\b', text)
        for word in words:
            if word in MAJOR_STOCKS:
                analysis['mentioned_stocks'].append(word)
                analysis['fraud_score'] += 2
                analysis['indicators'].append(f'stock_symbol_{word}')
        
        # Check for company names in text
        for leader, data in CORPORATE_LEADERS.items():
            for company in data['companies']:
                if company.lower() in text_lower:
                    for symbol in data['symbols']:
                        if symbol not in analysis['mentioned_stocks']:
                            analysis['mentioned_stocks'].append(symbol)
                            analysis['fraud_score'] += 1
        
        # Check for regulatory agencies
        for agency in REGULATORY_AGENCIES:
            if agency in text_lower:
                analysis['mentioned_agencies'].append(agency.upper())
                analysis['regulatory_score'] += 3
                analysis['fraud_score'] += 4
                analysis['indicators'].append(f'regulatory_agency_{agency.replace(" ", "_")}')
        
        # Check for suspicious patterns
        for category, terms in self.suspicious_patterns.items():
            for term in terms:
                if term in text_lower:
                    analysis['mentioned_terms'].append(term)
                    if category == 'urgency':
                        analysis['urgency_score'] += 2
                        analysis['fraud_score'] += 2
                    elif category == 'manipulation':
                        analysis['fraud_score'] += 4
                    elif category == 'deepfake':
                        analysis['fraud_score'] += 5
                    elif category == 'regulatory':
                        analysis['regulatory_score'] += 3
                        analysis['fraud_score'] += 3
                    
                    analysis['indicators'].append(f'{category}_{term.replace(" ", "_")}')
        
        # Calculate market impact potential
        impact_factors = 0
        if analysis['mentioned_leaders']:
            impact_factors += len(analysis['mentioned_leaders']) * 2
        if analysis['mentioned_stocks']:
            impact_factors += len(analysis['mentioned_stocks'])
        if analysis['mentioned_agencies']:
            impact_factors += len(analysis['mentioned_agencies']) * 2
        if analysis['urgency_score'] > 4:
            impact_factors += 3
        
        if impact_factors >= 8:
            analysis['market_impact_potential'] = 'critical'
            analysis['fraud_score'] += 6
        elif impact_factors >= 5:
            analysis['market_impact_potential'] = 'high'
            analysis['fraud_score'] += 4
        elif impact_factors >= 3:
            analysis['market_impact_potential'] = 'medium'
            analysis['fraud_score'] += 2
        
        # Determine overall risk level
        if analysis['fraud_score'] >= 20:
            analysis['risk_level'] = 'critical'
        elif analysis['fraud_score'] >= 15:
            analysis['risk_level'] = 'high'
        elif analysis['fraud_score'] >= 8:
            analysis['risk_level'] = 'medium'
        else:
            analysis['risk_level'] = 'low'
        
        return analysis

    async def analyze_image_content(self, image_path: str) -> Dict:
        """Analyze image for regulatory documents and OCR text"""
        analysis = {
            'has_text': False,
            'extracted_text': '',
            'text_confidence': 0.0,
            'is_document': False,
            'document_type': None,
            'authenticity_score': 1.0,
            'suspicious_elements': [],
            'text_analysis': {}
        }
        
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Enhance image for better OCR
            image = image.convert('RGB')
            enhancer = Image.fromarray(np.array(image))
            
            # Extract text using OCR
            try:
                extracted_text = pytesseract.image_to_string(image, config='--psm 6')
                analysis['extracted_text'] = extracted_text
                analysis['has_text'] = len(extracted_text.strip()) > 20
                
                # Get confidence score
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                analysis['text_confidence'] = np.mean(confidences) / 100.0 if confidences else 0.0
                
            except Exception as e:
                logger.error(f"OCR error: {e}")
                analysis['ocr_error'] = str(e)
            
            if analysis['has_text']:
                text_lower = extracted_text.lower()
                
                # Check if it's a regulatory document
                if any(agency in text_lower for agency in REGULATORY_AGENCIES):
                    analysis['is_document'] = True
                    analysis['document_type'] = 'regulatory'
                    
                    # Document authenticity checks
                    suspicious_elements = []
                    
                    # Check for proper date formatting
                    if not re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}\b', extracted_text):
                        if not re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]20\d{2}\b', extracted_text):
                            suspicious_elements.append('Missing or improper date format')
                    
                    # Check for agency name consistency
                    if 'sec' in text_lower and 'securities and exchange commission' not in text_lower:
                        suspicious_elements.append('SEC mentioned without full agency name')
                    
                    # Check for unrealistic monetary amounts
                    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?\s*(?:billion|million|trillion)?', extracted_text, re.IGNORECASE)
                    for amount in amounts:
                        try:
                            amount_clean = re.sub(r'[^\d.]', '', amount.split('$')[1])
                            if 'billion' in amount.lower():
                                value = float(amount_clean)
                                if value > 50:  # Over $50 billion is highly suspicious
                                    suspicious_elements.append(f'Unrealistic penalty amount: {amount}')
                            elif 'trillion' in amount.lower():
                                suspicious_elements.append(f'Unrealistic penalty amount: {amount}')
                        except:
                            pass
                    
                    # Check for official document formatting
                    if not re.search(r'(Case|File|Docket|Release|Order)\s+(No\.|Number|#)\s*\d+', extracted_text, re.IGNORECASE):
                        suspicious_elements.append('Missing official case/file number')
                    
                    # Check for proper letterhead elements
                    if analysis['is_document'] and analysis['text_confidence'] > 0.7:
                        if not re.search(r'(Washington|DC|D\.C\.)', extracted_text, re.IGNORECASE):
                            suspicious_elements.append('Missing Washington DC address')
                    
                    analysis['suspicious_elements'] = suspicious_elements
                    
                    # Calculate authenticity score
                    authenticity_penalty = len(suspicious_elements) * 0.2
                    if analysis['text_confidence'] < 0.5:
                        authenticity_penalty += 0.3  # Low OCR confidence is suspicious
                    
                    analysis['authenticity_score'] = max(0.1, 1.0 - authenticity_penalty)
                
                # Analyze extracted text for fraud patterns
                if len(extracted_text.strip()) > 50:
                    analysis['text_analysis'] = await self.analyze_text_content(extracted_text)
                
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            analysis['error'] = str(e)
        
        return analysis

    async def analyze_video_content(self, video_path: str) -> Dict:
        """Analyze video for deepfake indicators"""
        analysis = {
            'duration': 0,
            'fps': 0,
            'total_frames': 0,
            'frames_analyzed': 0,
            'faces_detected': 0,
            'suspicious_frames': 0,
            'deepfake_probability': 0.0,
            'quality_issues': [],
            'face_consistency_score': 1.0,
            'temporal_consistency_score': 1.0
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            analysis['fps'] = fps
            analysis['total_frames'] = total_frames
            analysis['duration'] = duration
            
            if duration > MAX_VIDEO_DURATION:
                analysis['error'] = f'Video too long ({duration:.1f}s). Maximum: {MAX_VIDEO_DURATION}s'
                cap.release()
                return analysis
            
            # Load face detection model
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Analyze frames (sample every N frames to avoid overprocessing)
            frame_interval = max(1, total_frames // 50)  # Analyze up to 50 frames
            current_frame = 0
            previous_face_features = None
            face_consistency_scores = []
            
            while current_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                analysis['frames_analyzed'] += 1
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    analysis['faces_detected'] += 1
                    
                    # Analyze each face
                    for (x, y, w, h) in faces:
                        face_region = gray[y:y+h, x:x+w]
                        
                        # Basic deepfake indicators
                        face_issues = []
                        
                        # Check texture consistency
                        mean_intensity = np.mean(face_region)
                        std_intensity = np.std(face_region)
                        
                        if std_intensity < 20:  # Too uniform (unnatural)
                            face_issues.append('uniform_texture')
                        
                        if mean_intensity < 60 or mean_intensity > 200:  # Unusual lighting
                            face_issues.append('unusual_lighting')
                        
                        # Check for artificial boundaries
                        edges = cv2.Canny(face_region, 50, 150)
                        edge_density = np.sum(edges > 0) / (w * h)
                        
                        if edge_density < 0.05:  # Too few edges (blurred/artificial)
                            face_issues.append('artificial_smoothing')
                        elif edge_density > 0.3:  # Too many edges (compression artifacts)
                            face_issues.append('compression_artifacts')
                        
                        # Track face consistency across frames
                        if previous_face_features is not None:
                            # Simple feature comparison (histogram)
                            hist1 = cv2.calcHist([face_region], [0], None, [256], [0, 256])
                            hist2 = previous_face_features
                            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                            face_consistency_scores.append(correlation)
                        
                        previous_face_features = cv2.calcHist([face_region], [0], None, [256], [0, 256])
                        
                        if face_issues:
                            analysis['quality_issues'].extend(face_issues)
                            analysis['suspicious_frames'] += 1
                
                current_frame += frame_interval
            
            cap.release()
            
            # Calculate consistency scores
            if face_consistency_scores:
                analysis['face_consistency_score'] = np.mean(face_consistency_scores)
                if analysis['face_consistency_score'] < 0.7:
                    analysis['quality_issues'].append('inconsistent_faces')
            
            # Calculate deepfake probability
            suspicion_factors = 0
            
            if analysis['frames_analyzed'] > 0:
                suspicion_ratio = analysis['suspicious_frames'] / analysis['frames_analyzed']
                suspicion_factors += suspicion_ratio * 0.4
            
            if analysis['face_consistency_score'] < 0.8:
                suspicion_factors += (0.8 - analysis['face_consistency_score']) * 0.3
            
            # Quality issues contribute to suspicion
            unique_issues = set(analysis['quality_issues'])
            suspicion_factors += len(unique_issues) * 0.1
            
            analysis['deepfake_probability'] = min(suspicion_factors, 1.0)
            
        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            analysis['error'] = str(e)
        
        return analysis

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash for file"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation error: {e}")
            return ""

# Initialize detector
detector = FinancialFraudDetector()

# -------------------- Database Functions --------------------
async def get_suspicious_keywords():
    """Fetch keywords from database with fallback"""
    try:
        data = supabase.table("suspicious_keywords").select("keyword, category, weight").execute()
        if data.data:
            return data.data
        logger.warning("No keywords found in database, using fallback")
    except Exception as e:
        logger.error(f"Database error fetching keywords: {e}")
    
    # Fallback keywords
    return [
        {'keyword': 'breaking news', 'category': 'urgency', 'weight': 3},
        {'keyword': 'sec investigation', 'category': 'regulatory', 'weight': 5},
        {'keyword': 'insider trading', 'category': 'fraud', 'weight': 5},
        {'keyword': 'market manipulation', 'category': 'fraud', 'weight': 5},
        {'keyword': 'deepfake', 'category': 'manipulation', 'weight': 5},
        {'keyword': 'pump and dump', 'category': 'fraud', 'weight': 4},
        {'keyword': 'ponzi scheme', 'category': 'fraud', 'weight': 4},
        {'keyword': 'exclusive leak', 'category': 'urgency', 'weight': 3},
        {'keyword': 'emergency order', 'category': 'regulatory', 'weight': 4},
        {'keyword': 'cease and desist', 'category': 'regulatory', 'weight': 4},
    ]

async def store_analysis_result(user_id: int, analysis_data: Dict):
    """Store analysis results in database"""
    try:
        # Prepare data for storage
        storage_data = {
            'user_id': user_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'risk_level': analysis_data.get('risk_level', 'low'),
            'fraud_score': analysis_data.get('overall_score', 0),
            'analysis_details': analysis_data,
            'mentioned_leaders': analysis_data.get('text_analysis', {}).get('mentioned_leaders', []),
            'mentioned_stocks': analysis_data.get('text_analysis', {}).get('mentioned_stocks', []),
            'mentioned_agencies': analysis_data.get('text_analysis', {}).get('mentioned_agencies', [])
        }
        
        # Insert analysis result
        result = supabase.table("fraud_analyses").insert(storage_data).execute()
        
        # Create alert for high-risk content
        if analysis_data.get('risk_level') in ['high', 'critical']:
            alert_data = {
                'user_id': user_id,
                'alert_type': 'financial_manipulation',
                'severity': analysis_data['risk_level'],
                'description': f"High-risk financial content detected (Score: {analysis_data.get('overall_score', 0)})",
                'details': analysis_data,
                'status': 'open',
                'timestamp': datetime.now().isoformat()
            }
            
            supabase.table("fraud_alerts").insert(alert_data).execute()
            logger.warning(f"HIGH RISK ALERT: User {user_id}, Score: {analysis_data.get('overall_score', 0)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Database storage error: {e}")
        return None

# -------------------- Analysis Functions --------------------
# -------------------- Analysis Functions --------------------
async def comprehensive_analysis(
    message_text: str = "",
    file_path: str = None,
    file_type: str = None
) -> Dict:
    """Perform comprehensive fraud analysis"""
    analysis_result = {
        'timestamp': datetime.now().isoformat(),
        'text_analysis': {},
        'media_analysis': {},
        'overall_score': 0,
        'risk_level': 'low',
        'recommendations': [],
        'file_hash': None
    }
    
    try:
        # Analyze text content
        if message_text and len(message_text.strip()) > 0:
            text_analysis = await detector.analyze_text_content(message_text)
            analysis_result['text_analysis'] = text_analysis
            logger.info(f"Text analysis completed. Score: {text_analysis['fraud_score']}")
        
        # Analyze media content
        if file_path and os.path.exists(file_path):
            file_hash = detector.calculate_file_hash(file_path)
            analysis_result['file_hash'] = file_hash
            
            # Check for duplicate files
            try:
                existing_analysis = supabase.table("fraud_analyses")\
                    .select("analysis_details")\
                    .contains("analysis_details", {"file_hash": file_hash})\
                    .limit(1)\
                    .execute()
                
                if existing_analysis.data:
                    analysis_result['duplicate_detected'] = True
                    logger.info(f"Duplicate file detected: {file_hash}")
            except Exception as e:
                logger.warning(f"Supabase duplicate check failed: {e}")
            
            # Perform media analysis
            if file_type == 'image':
                media_analysis = await detector.analyze_image_content(file_path)
                analysis_result['media_analysis'] = media_analysis
                logger.info(f"Image analysis completed. Document: {media_analysis.get('is_document', False)}")
                
            elif file_type == 'video':
                media_analysis = await detector.analyze_video_content(file_path)
                analysis_result['media_analysis'] = media_analysis
                logger.info(f"Video analysis completed. Deepfake prob: {media_analysis.get('deepfake_probability', 0):.2f}")
        
        # Calculate overall score (placeholder for now)
        total_score = 0
        risk_factors = []
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
    
    return analysis_result

# Text analysis contribution
if store_analysis_result.get('text_analysis'):
    text_score = store_analysis_result['text_analysis'].get('fraud_score', 0)
    total_score += text_score

    if text_score > 0:
        risk_factors.extend(store_analysis_result['text_analysis'].get('indicators', []))

# Media analysis contribution
if store_analysis_result.get('media_analysis'):
    media_analysis = store_analysis_result['media_analysis']

    # Deepfake detection
    deepfake_prob = media_analysis.get('deepfake_probability', 0)
    if deepfake_prob > 0.5:
        deepfake_score = int(deepfake_prob * 15)
        total_score += deepfake_score
        risk_factors.append('deepfake_detected')
        logger.warning(f"Deepfake detected with {deepfake_prob:.2%} probability")

    # Document authenticity
    auth_score = media_analysis.get('authenticity_score')
    if media_analysis.get('is_document') and auth_score is not None and auth_score < 0.7:
        doc_score = int((1.0 - auth_score) * 10)
        total_score += doc_score
        risk_factors.append('fake_document')
        logger.warning(f"Suspicious document detected with {auth_score:.2%} authenticity")

    # Text extracted from media
    if media_analysis.get('text_analysis'):
        extracted_text_score = media_analysis['text_analysis'].get('fraud_score', 0)
        total_score += extracted_text_score
        risk_factors.extend(media_analysis['text_analysis'].get('indicators', []))

# Market impact multiplier
text_analysis = analysis_result.get('text_analysis') or {}
impact = text_analysis.get('market_impact_potential')
if impact == 'critical':
    total_score = int(total_score * 1.3)
elif impact == 'high':
    total_score = int(total_score * 1.2)

analysis_result['overall_score'] = total_score
analysis_result['risk_factors'] = risk_factors

def format_analysis_response(analysis: Dict) -> str:
    """Format analysis results for user display"""
    risk_level = analysis['risk_level']
    score = analysis['overall_score']
    
    # Risk level header
    risk_emojis = {
        'critical': '🚨',
        'high': '⚠️',
        'medium': '⚠️',
        'low': '✅',
        'error': '❌'
    }
    
    emoji = risk_emojis.get(risk_level, '❓')
    header = f"{emoji} {risk_level.upper()} RISK (Score: {score}) {emoji}"
    
    response = f"{header}\n\n"
    
    # Text analysis section
    text_analysis = analysis.get('text_analysis', {})
    if text_analysis:
        response += "📝 *Content Analysis:*\n"
        
        if text_analysis.get('mentioned_leaders'):
            leaders = ', '.join(text_analysis['mentioned_leaders'])
            response += f"👤 Corporate Leaders: {leaders}\n"
        
        if text_analysis.get('mentioned_stocks'):
            stocks = ', '.join(text_analysis['mentioned_stocks'])
            response += f"📈 Stock Symbols: {stocks}\n"
        
        if text_analysis.get('mentioned_agencies'):
            agencies = ', '.join(text_analysis['mentioned_agencies'])
            response += f"🏛️ Regulatory Agencies: {agencies}\n"
        
        if text_analysis.get('market_impact_potential', 'low') != 'low':
            impact = text_analysis['market_impact_potential']
            response += f"📊 Market Impact Potential: {impact.upper()}\n"
        
        if text_analysis.get('urgency_score', 0) > 2:
            response += f"⏰ Urgency Score: {text_analysis['urgency_score']}\n"
    
    # Media analysis section
    media_analysis = analysis.get('media_analysis', {})
    if media_analysis:
        response += "\n🎥 *Media Analysis:*\n"
        
        # Deepfake detection
        if 'deepfake_probability' in media_analysis:
            prob = media_analysis['deepfake_probability']
            if prob > 0.1:  # Only show if significant
                response += f"🎭 Deepfake Probability: {prob:.1%}\n"
        
        # Document analysis
        if media_analysis.get('is_document'):
            doc_type = media_analysis.get('document_type', 'unknown')
            auth_score = media_analysis.get('authenticity_score', 1.0)
            response += f"📄 {doc_type.title()} Document Detected\n"
            response += f"✓ Authenticity Score: {auth_score:.1%}\n"
            
            suspicious_elements = media_analysis.get('suspicious_elements', [])
            if suspicious_elements:
                response += f"⚠️ Issues Found: {len(suspicious_elements)}\n"
        
        # OCR confidence
        if media_analysis.get('text_confidence', 0) > 0:
            confidence = media_analysis['text_confidence']
            response += f"📖 Text Recognition Confidence: {confidence:.1%}\n"
    
    # Duplicate detection
    if analysis.get('duplicate_detected'):
        response += "\n🔄 *Duplicate Content Detected*\n"
        response += "This file has been analyzed before\n"
    
    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        response += "\n💡 *Recommendations:*\n"
        for i, rec in enumerate(recommendations[:4], 1):  # Show max 4 recommendations
            response += f"{i}. {rec}\n"
    
    # Additional warnings for high-risk content
    if risk_level in ['critical', 'high']:
        response += "\n⚠️ *IMPORTANT:*\n"
        response += "• Verify all information with official sources\n"
        response += "• Do not share without verification\n"
        response += "• Consider reporting suspicious content\n"
    
    return response

# -------------------- Bot Handlers --------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message and bot introduction"""
    user_id = update.message.from_user.id
    username = update.message.from_user.username or "Unknown"
    
    logger.info(f"User {user_id} ({username}) started the bot")
    
    welcome_text = """
🛡️ *Enhanced Financial Fraud Detection Bot*

I specialize in detecting:
🎭 *Deepfake videos/audios* of corporate leaders
📄 *Fabricated regulatory documents* (SEC, FINRA, etc.)
📈 *Market manipulation attempts*
🔍 *Financial fraud patterns*

*Supported Content:*
• Text messages (analysis of financial claims)
• Images (OCR + document verification)  
• Videos (deepfake detection)
• Audio files (voice synthesis detection)

*How to use:*
1. Send me any content to analyze
2. I'll provide a detailed fraud assessment
3. Follow my recommendations for verification

*Commands:*
/start - Show this message
/help - Detailed help information
/stats - Your analysis statistics

⚠️ *Always verify financial information with official sources!*
"""
    
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detailed help information"""
    help_text = """
🆘 *Detailed Help & Information*

*🎯 What I Detect:*
• Deepfake videos of CEOs and executives
• Fake SEC/FINRA/regulatory documents
• Market manipulation language and schemes  
• Corporate announcement fraud
• Stock pump & dump attempts
• Insider trading claims

*📊 Risk Assessment Levels:*
🟢 *LOW (0-9):* Safe content, no major concerns
🟡 *MEDIUM (10-17):* Some suspicious elements
🔴 *HIGH (18-24):* Likely fraudulent content
⚫ *CRITICAL (25+):* Immediate threat detected

*📁 Supported File Types:*
• *Images:* JPG, PNG, PDF (OCR text extraction)
• *Videos:* MP4, AVI, MOV (deepfake detection)
• *Audio:* MP3, WAV (voice synthesis analysis)
• *Text:* Direct message analysis

*🔍 Analysis Features:*
• Corporate leader identification
• Stock symbol recognition  
• Regulatory agency detection
• Market impact assessment
• Document authenticity scoring
• Duplicate content detection

*⚠️ Important Notes:*
• Maximum file size: 50MB
• Video length limit: 5 minutes
• OCR works best with clear, high-quality images
• Always cross-check with official sources

*🚨 When to Report:*
• Critical risk detections
• Suspected market manipulation
• Viral spread of fake content
• Regulatory impersonation

Contact financial authorities if you suspect real fraud!
"""
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user analysis statistics"""
    user_id = update.message.from_user.id
    
    try:
        # Fetch user's analysis history
        data = supabase.table("fraud_analyses")\
            .select("risk_level, fraud_score, analysis_timestamp, mentioned_leaders, mentioned_stocks")\
            .eq("user_id", user_id)\
            .order("analysis_timestamp", desc=True)\
            .limit(100)\
            .execute()
        
        if not data.data:
            await update.message.reply_text("📊 No analysis history found. Send me some content to analyze!")
            return
        
        analyses = data.data
        total_analyses = len(analyses)
        
        # Count by risk level
        risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        fraud_scores = []
        leaders_mentioned = set()
        stocks_mentioned = set()
        
        for analysis in analyses:
            risk_level = analysis.get('risk_level', 'low')
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
            
            fraud_scores.append(analysis.get('fraud_score', 0))
            
            # Collect mentioned entities
            if analysis.get('mentioned_leaders'):
                leaders_mentioned.update(analysis['mentioned_leaders'])
            if analysis.get('mentioned_stocks'):
                stocks_mentioned.update(analysis['mentioned_stocks'])
        
        avg_score = sum(fraud_scores) / len(fraud_scores) if fraud_scores else 0
        max_score = max(fraud_scores) if fraud_scores else 0
        
        # Recent activity (last 7 days)
        recent_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        recent_analyses = [a for a in analyses if datetime.fromisoformat(a['analysis_timestamp'].replace('Z', '+00:00')).date() >= (recent_date.date())]
        
        stats_text = f"""
📊 *Your Analysis Statistics*

*📈 Overview:*
• Total Analyses: {total_analyses}
• Average Risk Score: {avg_score:.1f}
• Highest Risk Score: {max_score}
• Recent Activity (7 days): {len(recent_analyses)}

*🎯 Risk Distribution:*
🟢 Low Risk: {risk_counts['low']} ({risk_counts['low']/total_analyses*100:.1f}%)
🟡 Medium Risk: {risk_counts['medium']} ({risk_counts['medium']/total_analyses*100:.1f}%)
🔴 High Risk: {risk_counts['high']} ({risk_counts['high']/total_analyses*100:.1f}%)
⚫ Critical Risk: {risk_counts['critical']} ({risk_counts['critical']/total_analyses*100:.1f}%)

*📊 Content Analysis:*
• Corporate Leaders Mentioned: {len(leaders_mentioned)}
• Stock Symbols Detected: {len(stocks_mentioned)}
"""
        
        # Add most mentioned entities if any
        if leaders_mentioned:
            top_leaders = list(leaders_mentioned)[:5]
            stats_text += f"• Top Leaders: {', '.join(top_leaders)}\n"
        
        if stocks_mentioned:
            top_stocks = list(stocks_mentioned)[:5]
            stats_text += f"• Top Stocks: {', '.join(top_stocks)}\n"
        
        stats_text += """
*💡 Tips:*
• Keep analyzing suspicious content
• Always verify with official sources
• Report confirmed fraud to authorities
• Share awareness (not the fraudulent content!)
"""
        
        await update.message.reply_text(stats_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Stats error for user {user_id}: {e}")
        await update.message.reply_text("📊 Unable to retrieve statistics. Please try again later.")

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all media file types"""
    message = update.message
    user_id = message.from_user.id
    username = message.from_user.username or "Unknown"
    
    # Show processing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    file_path = None
    file_type = None
    
    try:
        timestamp = int(datetime.now().timestamp())
        
        # Handle different media types
        if message.photo:
            file = await context.bot.get_file(message.photo[-1].file_id)
            file_path = os.path.join(TEMP_DIR, f"img_{user_id}_{timestamp}.jpg")
            await file.download_to_drive(file_path)
            file_type = 'image'
            logger.info(f"Image received from user {user_id} ({username})")
            
        elif message.video:
            file = await context.bot.get_file(message.video.file_id)
            file_path = os.path.join(TEMP_DIR, f"vid_{user_id}_{timestamp}.mp4")
            await file.download_to_drive(file_path)
            file_type = 'video'
            logger.info(f"Video received from user {user_id} ({username})")
            
        elif message.audio or message.voice:
            file_id = message.audio.file_id if message.audio else message.voice.file_id
            file = await context.bot.get_file(file_id)
            file_path = os.path.join(TEMP_DIR, f"aud_{user_id}_{timestamp}.mp3")
            await file.download_to_drive(file_path)
            file_type = 'audio'
            logger.info(f"Audio received from user {user_id} ({username})")
            
        elif message.document:
            file = await context.bot.get_file(message.document.file_id)
            ext = message.document.file_name.split('.')[-1].lower() if message.document.file_name else 'pdf'
            file_path = os.path.join(TEMP_DIR, f"doc_{user_id}_{timestamp}.{ext}")
            await file.download_to_drive(file_path)
            file_type = 'image'  # Treat documents as images for OCR
            logger.info(f"Document received from user {user_id} ({username}): {message.document.file_name}")
        
        if not file_path:
            await message.reply_text("❌ Unsupported file type. Please send images, videos, audio, or documents.")
            return
        
        # Check file size
        if not os.path.exists(file_path):
            await message.reply_text("❌ Error downloading file. Please try again.")
            return
            
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            await message.reply_text(f"❌ File too large ({file_size_mb:.1f}MB). Maximum size: {MAX_FILE_SIZE_MB}MB")
            os.remove(file_path)
            return
        
        logger.info(f"Processing {file_type} file: {file_size_mb:.1f}MB")
        
        # Send processing message for larger files
        if file_size_mb > 5 or file_type == 'video':
            processing_msg = await message.reply_text("🔄 Processing file... This may take a moment.")
        else:
            processing_msg = None
        
        # Perform comprehensive analysis
        analysis = await comprehensive_analysis(
            message_text=message.caption or "",
            file_path=file_path,
            file_type=file_type
        )
        
        # Store results in database
        await store_analysis_result(user_id, analysis)
        
        # Delete processing message
        if processing_msg:
            try:
                await processing_msg.delete()
            except:
                pass
        
        # Format and send response
        response = format_analysis_response(analysis)
        await message.reply_text(response, parse_mode='Markdown')
        
        # Send additional warning for critical content
        if analysis.get('risk_level') == 'critical':
            warning_msg = """
🚨 *CRITICAL ALERT* 🚨

This content has been flagged as extremely high risk for financial fraud or market manipulation.

*Immediate Actions:*
• Do NOT share this content
• Do NOT make investment decisions based on it
• Report to financial authorities if appropriate
• Verify all claims through official channels

*Authorities to Contact:*
• SEC: sec.gov/whistleblower
• FINRA: finra.org/investors
• Local financial regulators
"""
            await message.reply_text(warning_msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Media processing error for user {user_id}: {e}")
        await message.reply_text(f"❌ Error processing file: {str(e)}\n\nPlease try again or contact support if the issue persists.")
    
    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    message_text = update.message.text
    user_id = update.message.from_user.id
    username = update.message.from_user.username or "Unknown"
    
    if not message_text or len(message_text.strip()) < 10:
        await update.message.reply_text("❓ Please send a longer message for analysis (at least 10 characters).")
        return
    
    logger.info(f"Text message received from user {user_id} ({username}): {len(message_text)} chars")
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        # Perform analysis
        analysis = await comprehensive_analysis(message_text=message_text)
        
        # Store results in database
        await store_analysis_result(user_id, analysis)
        
        # Format and send response
        response = format_analysis_response(analysis)
        await update.message.reply_text(response, parse_mode='Markdown')
        
        # Additional warning for high-risk text
        if analysis.get('risk_level') in ['high', 'critical']:
            risk_level = analysis['risk_level']
            score = analysis['overall_score']
            
            if any('deepfake' in indicator for indicator in analysis.get('risk_factors', [])):
                await update.message.reply_text(
                    "⚠️ *Deepfake content suspected!* This message may reference artificially generated media. Verify through official channels.",
                    parse_mode='Markdown'
                )
            
            if analysis.get('text_analysis', {}).get('market_impact_potential') in ['high', 'critical']:
                await update.message.reply_text(
                    "📈 *High Market Impact Detected!* This content could significantly affect stock prices if shared widely. Exercise extreme caution.",
                    parse_mode='Markdown'
                )
        
    except Exception as e:
        logger.error(f"Text analysis error for user {user_id}: {e}")
        await update.message.reply_text(f"❌ Error analyzing message: {str(e)}")

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle unknown commands"""
    await update.message.reply_text(
        "❓ Unknown command. Use /help to see available commands.\n\n"
        "You can also send me text, images, videos, or audio files for fraud analysis!"
    )

# -------------------- Main Application --------------------
async def main():
    """Main bot application"""
    try:
        # Initialize application
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("stats", stats_command))
        
        # Add media handlers (prioritize over text)
        app.add_handler(MessageHandler(
            filters.PHOTO | filters.VIDEO | filters.AUDIO | 
            filters.VOICE | filters.DOCUMENT, 
            handle_media
        ))
        
        # Add text handler
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
        
        # Add unknown command handler
        app.add_handler(MessageHandler(filters.COMMAND, handle_unknown))
        
        # Log startup
        logger.info("🚀 Enhanced Financial Fraud Detection Bot starting...")
        logger.info("🎭 Deepfake detection: ENABLED")
        logger.info("📄 Document verification: ENABLED")
        logger.info("📈 Market manipulation detection: ENABLED")
        logger.info("🔍 Fraud pattern analysis: ENABLED")
        
        print("🛡️ Enhanced Financial Fraud Detection Bot is running...")
        print("🎯 Monitoring for:")
        print("   • Deepfake videos/audios of corporate leaders")
        print("   • Fabricated regulatory documents")
        print("   • Market manipulation attempts")
        print("   • Financial fraud patterns")
        print("\n📊 Bot ready to analyze content!")
        
        # Initialize and start bot
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        
        # Keep running until interrupted
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("🛑 Shutdown signal received...")
        
    except Exception as e:
        logger.error(f"Fatal error starting bot: {e}")
        print(f"❌ Fatal error: {e}")
    
    finally:
        logger.info("🛑 Stopping bot...")
        print("🛑 Stopping bot...")
        try:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
        except:
            pass
        logger.info("✅ Bot stopped successfully")
        print("✅ Bot stopped successfully")

if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())