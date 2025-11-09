from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Import the full phishing detector with spellchecker
try:
    from phishing_detector import NonMLPhishingDetector
    print("✅ Full Phishing Detector with Spellchecker Loaded!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("⚠️ Please install: pip install pyspellchecker language-tool-python dnspython tldextract")
    exit(1)

app = Flask(__name__)
CORS(app)
detector = NonMLPhishingDetector()

@app.route('/')
def home():
    return send_from_directory('.', 'phishing_detector.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/analyze', methods=['POST'])
def analyze_email():
    try:
        data = request.json
        print("📧 Analyzing email...")
        
        result = detector.detect_phishing({
            'sender_email': data.get('sender_email', ''),
            'subject': data.get('subject', ''),
            'body': data.get('body', '')
        })
        
        print(f"🎯 Result - Score: {result['phishing_score']:.2f}, Phishing: {result['is_phishing']}")
        if result.get('misspelled_words'):
            print(f"✏️ Misspelled: {result['misspelled_words']}")
            
        return jsonify(result)
        
    except Exception as e:
        print(f"💥 Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'detector': 'with_spellchecker'})

if __name__ == '__main__':
    print("🚀 Starting Phishing Detector with Spellchecker...")
    print("🌐 http://localhost:5000")
    app.run(debug=True, port=5000)