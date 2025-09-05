# 🛡️ Financial Fraud Detection Telegram Bot

A sophisticated Telegram bot that detects **deepfake videos/audios** of corporate leaders and **fabricated regulatory documents** that could manipulate stock prices.

## 🎯 Key Features

- **🎭 Deepfake Detection**: AI-powered analysis of videos and audio for synthetic corporate leader content
- **📄 Document Verification**: OCR analysis of regulatory documents with authenticity scoring
- **📈 Market Impact Assessment**: Identifies potential stock price manipulation attempts
- **🔍 Real-time Analysis**: Instant fraud scoring with detailed risk assessments
- **📊 Comprehensive Tracking**: User reputation system and duplicate content detection

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+** installed
- **Tesseract OCR** installed on your system
- **Supabase account** for database
- **Telegram Bot Token** from @BotFather

### 2. Installation

```bash
# Clone or download the project
git clone <your-repo-url>
cd financial-fraud-detection-bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. System Dependencies

#### Windows
```powershell
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use Chocolatey:
choco install tesseract

# Update path in .env file:
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

#### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install tesseract opencv python@3.11
```

#### Ubuntu/Debian
```bash
# Update system
sudo apt update

# Install dependencies
sudo apt install python3 python3-pip tesseract-ocr tesseract-ocr-eng
sudo apt install libopencv-dev python3-opencv
sudo apt install ffmpeg  # For video processing
```

### 4. Database Setup

1. **Create Supabase Project**:
   - Go to [supabase.com](https://supabase.com)
   - Create new project
   - Copy your Project URL and Service Role Key

2. **Run Database Schema**:
   - Open Supabase SQL Editor
   - Copy content from `database_schema.sql`
   - Execute the schema

### 5. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
nano .env
```

**Required configuration:**
```env
TELEGRAM_BOT_TOKEN=your-bot-token-here
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-service-role-key
TESSERACT_PATH=/usr/bin/tesseract  # Adjust for your system
```

### 6. Run the Bot

```bash
python main.py
```

You should see:
```
🛡️ Enhanced Financial Fraud Detection Bot is running...
🎯 Monitoring for:
   • Deepfake videos/audios of corporate leaders
   • Fabricated regulatory documents
   • Market manipulation attempts
   • Financial fraud patterns

📊 Bot ready to analyze content!
```

## 📱 How to Use

### Bot Commands

- `/start` - Welcome message and feature overview
- `/help` - Detailed help and risk level explanations
- `/stats` - Your personal analysis statistics

### Content Analysis

1. **Send any content** to the bot:
   - Text messages
   - Images (documents, screenshots)
   - Videos (MP4, AVI, MOV)
   - Audio files (MP3, WAV)

2. **Get instant analysis**:
   - Risk level (Low/Medium/High/Critical)
   - Fraud score (0-30+)
   - Detected entities (leaders, stocks, agencies)
   - Specific recommendations

### Example Interactions

**Text Analysis:**
```
User: "BREAKING: SEC fines Tesla $50 billion for Elon Musk insider trading"

Bot: 🚨 CRITICAL RISK (Score: 28) 🚨

📝 Content Analysis:
👤 Corporate Leaders: Elon Musk
📈 Stock Symbols: TSLA
🏛️ Regulatory Agencies: SEC
📊 Market Impact Potential: CRITICAL

💡 Recommendations:
1. 🚨 CRITICAL THREAT DETECTED
2. Extremely high probability of market manipulation
3. DO NOT SHARE OR ACT ON THIS INFORMATION
4. Report to relevant financial authorities immediately
```

**Video Analysis:**
```
User: [Sends deepfake video of Tim Cook]

Bot: ⚠️ HIGH RISK (Score: 22) ⚠️

📝 Content Analysis:
👤 Corporate Leaders: Tim Cook
📈 Stock Symbols: AAPL

🎥 Media Analysis:
🎭 Deepfake Probability: 78.5%

💡 Recommendations:
1. ⚠️ HIGH RISK: Likely fraudulent financial content
2. Multiple suspicious indicators detected
3. Cross-reference with official regulatory announcements
4. Do not make investment decisions based on this content
```

## 🔧 Configuration Options

### Analysis Thresholds

```env
# Deepfake detection sensitivity (0.0 to 1.0)
DEEPFAKE_THRESHOLD=0.5

# Fraud score thresholds
FRAUD_SCORE_THRESHOLD=15      # High risk alert
CRITICAL_ALERT_THRESHOLD=20   # Critical risk alert
```

### File Processing Limits

```env
# Maximum file size in MB
MAX_FILE_SIZE_MB=50

# Maximum video length in seconds
MAX_VIDEO_DURATION_SECONDS=300

# Temporary files directory
TEMP_FILES_DIR=./temp_files/
```

### Feature Toggles

```env
# Enable/disable specific features
ENABLE_DEEPFAKE_DETECTION=True
ENABLE_DOCUMENT_VERIFICATION=True
ENABLE_STOCK_MONITORING=True
ENABLE_REPUTATION_SYSTEM=True
```

## 📊 Risk Assessment Levels

| Level | Score Range | Description | Action Required |
|-------|-------------|-------------|-----------------|
| 🟢 **LOW** | 0-9 | Safe content | None |
| 🟡 **MEDIUM** | 10-17 | Some suspicious elements | Verify independently |
| 🔴 **HIGH** | 18-24 | Likely fraudulent | Do not share/act |
| ⚫ **CRITICAL** | 25+ | Immediate threat | Report to authorities |

## 🎭 Detection Capabilities

### Deepfake Detection
- **Video Analysis**: Frame-by-frame consistency checking
- **Face Detection**: Facial recognition and anomaly detection
- **Temporal Analysis**: Motion and lighting consistency
- **Quality Assessment**: Compression artifact detection
- **Leader Recognition**: Specific targeting of corporate executives

### Document Verification  
- **OCR Analysis**: Text extraction from images
- **Format Validation**: Official document structure checking
- **Content Analysis**: Unrealistic claims detection
- **Agency Verification**: Regulatory authority authenticity
- **Cross-referencing**: Official announcement comparison

### Market Impact Analysis
- **Entity Recognition**: Corporate leaders and stock symbols
- **Impact Scoring**: Potential market effect calculation
- **Urgency Detection**: False urgency tactic identification
- **Historical Patterns**: Known fraud scheme recognition

## 🗄️ Database Structure

The bot uses Supabase with these main tables:

- **`fraud_analyses`** - All analysis results and risk scores
- **`fraud_alerts`** - High-priority threat notifications
- **`user_profiles`** - User reputation and behavior tracking
- **`media_fingerprints`** - Duplicate content detection
- **`corporate_leaders`** - Executive database for targeting
- **`monitored_stocks`** - Tracked company symbols
- **`suspicious_keywords`** - Dynamic fraud pattern database

## 🔐 Security Features

- **Rate Limiting**: Prevents abuse and spam
- **Content Validation**: Input sanitization and size limits
- **File Security**: Temporary file cleanup and hash verification
- **User Reputation**: Behavioral analysis and scoring
- **Audit Logging**: Complete activity tracking
- **Data Privacy**: GDPR-compliant data handling

## 🛠️ Troubleshooting

### Common Issues

**Bot not responding:**
```bash
# Check bot token
python -c "import requests; print(requests.get('https://api.telegram.org/bot<TOKEN>/getMe').json())"

# Check logs
tail -f fraud_detection.log
```

**OCR not working:**
```bash
# Test Tesseract installation
tesseract --version

# Install additional language packs
sudo apt install tesseract-ocr-eng tesseract-ocr-fra
```

**Database connection issues:**
```bash
# Test Supabase connection
python -c "from supabase import create_client; client = create_client('URL', 'KEY'); print('Connected!')"
```

**High memory usage:**
```env
# Reduce processing limits
MAX_FRAMES_TO_ANALYZE=20
MAX_CONCURRENT_ANALYSES=2
CLEANUP_TEMP_FILES=True
```

### Performance Optimization

**For high-volume usage:**

1. **Enable caching:**
   ```env
   ENABLE_CACHING=True
   REDIS_URL=redis://localhost:6379
   ```

2. **Optimize video processing:**
   ```env
   MAX_FRAMES_TO_ANALYZE=30
   FRAME_ANALYSIS_INTERVAL=5
   ```

3. **Database optimization:**
   ```env
   DB_POOL_SIZE=20
   ENABLE_DB_LOGGING=False
   ```

## 📈 Monitoring & Analytics

### Built-in Statistics

The bot tracks:
- Total analyses performed
- Risk level distribution
- Most mentioned corporate leaders
- Most referenced stock symbols
- User reputation scores
- Detection accuracy metrics

### Logging

```bash
# View real-time logs
tail -f fraud_detection.log

# Filter by severity
grep "CRITICAL\|ERROR" fraud_detection.log

# Analysis statistics
grep "Analysis completed" fraud_detection.log | wc -l
```

### Database Queries

```sql
-- Get fraud statistics for last 7 days
SELECT * FROM get_fraud_statistics(7);

-- View recent high-risk detections
SELECT user_id, risk_level, fraud_score, mentioned_leaders, mentioned_stocks
FROM fraud_analyses 
WHERE risk_level IN ('high', 'critical')
ORDER BY analysis_timestamp DESC 
LIMIT 20;

-- User reputation overview
SELECT risk_category, COUNT(*) as user_count
FROM user_profiles 
GROUP BY risk_category;
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create temp directory
RUN mkdir -p temp_files

CMD ["python", "main.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  fraud-bot:
    build: .
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
    volumes:
      - ./temp_files:/app/temp_files
      - ./logs:/app/logs
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    restart: unless-stopped
```

### System Service (Linux)

```ini
# /etc/systemd/system/fraud-bot.service
[Unit]
Description=Financial Fraud Detection Bot
After=network.target

[Service]
Type=simple
User=fraud-bot
WorkingDirectory=/opt/fraud-bot
Environment=PATH=/opt/fraud-bot/venv/bin
ExecStart=/opt/fraud-bot/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable fraud-bot
sudo systemctl start fraud-bot

# Check status
sudo systemctl status fraud-bot
```

## 🔮 Future Enhancements

### Planned Features
- **Advanced ML Models**: Integration with specialized deepfake detection APIs
- **Multi-language Support**: Analysis in multiple languages
- **Social Media Integration**: Cross-platform content verification
- **Real-time Alerts**: Webhook notifications for critical threats
- **Admin Dashboard**: Web interface for monitoring and management
- **API Endpoints**: RESTful API for external integrations

### External Integrations
- **News APIs**: Cross-reference with legitimate news sources
- **Social Media APIs**: Verify content across platforms
- **Financial APIs**: Real-time stock price and news correlation
- **Regulatory APIs**: Direct integration with SEC/FINRA feeds

## 📞 Support & Contributing

### Getting Help
1. Check the troubleshooting section
2. Review logs for error messages
3. Verify configuration settings
4. Test individual components

### Reporting Issues
When reporting issues, include:
- Error messages from logs
- Configuration details (without sensitive keys)
- Steps to reproduce the problem
- System environment information

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This bot is designed to detect potential financial fraud and market manipulation. It should be used as a supplementary tool alongside human judgment and official verification processes. Always:

- Verify financial information through official sources
- Report suspected fraud to appropriate authorities
- Use the bot's assessments as guidance, not absolute truth
- Maintain awareness of evolving fraud techniques

## 🏆 Success Metrics

Track these metrics to measure effectiveness:

- **Detection Accuracy**: Correctly identified fraud attempts
- **False Positive Rate**: Legitimate content flagged incorrectly  
- **Response Time**: Average analysis completion speed
- **User Engagement**: Active users and content analyzed
- **Threat Prevention**: Potential market manipulation stopped

---

**🛡️ Stay vigilant against financial fraud and protect market integrity!**

For questions, support, or contributions, please refer to the project documentation or create an issue in the repository.