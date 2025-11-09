# Non-ML algorithms to detect phishing emails

import re
#import pandas as pd 
from spellchecker import SpellChecker
import language_tool_python
import dns.resolver
from urllib.parse import urlparse
import tldextract

class NonMLPhishingDetector:
    def __init__(self):
        self.spell = SpellChecker()
        with language_tool_python.LanguageTool('en-US') as tool:
            self.tool = tool
    
    #Spellcheck and language analysis: 
    def check_spelling_errors(self, text):
        #Check for spelling mistakes (phishing emails often have more errors)"""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        misspelled = self.spell.unknown(words)
        error_ratio = len(misspelled) / len(words) if words else 0
        return error_ratio, list(misspelled)
    
    def check_grammar_errors(self, text):
        #Check grammar mistakes using LanguageTool"""
        matches = self.tool.check(text)
        error_density = len(matches) / len(text.split()) if text else 0
        return error_density, len(matches)

    #Email address check: 
    def validate_sender_email(self, email_address):
        #Validate sender email address"""
        try:
            #Basic email format validation
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email_address):
                return False, "Invalid email format"
            
            domain = email_address.split('@')[1] # Get email domain
            
            #Check MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                if not mx_records:
                    return False, "No MX records found"
            except:
                return False, "Domain does not exist"
            
            return True, "Valid email"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def check_suspicious_domains(self, email_address):
        #Check for suspicious email domains"""
        suspicious_domains = [
            '.ru', '.tk', '.ml', '.ga', '.cf', '.xyz', '.top',
            '.club', '.info', '.biz', '.work', '.gq'
        ]
        
        domain = email_address.lower()
        is_suspicious = any(sd in domain for sd in suspicious_domains)
        return is_suspicious
    
    #URL analysis:
    def analyze_single_url(self, url):
        """Analyze a single URL for phishing indicators"""
        try:
            parsed = urlparse(url)
            domain_info = tldextract.extract(url)
            
            #Check for IP address instead of domain
            has_ip = re.match(r'^\d+\.\d+\.\d+\.\d+$', parsed.netloc) is not None
            
            #Check for suspicious characters
            has_suspicious_chars = any(char in url for char in ['@', '//'])
            
            #Check URL length
            is_long_url = len(url) > 75
            
            #Check for brand names in subdomains
            has_brand_hijacking = any(brand in domain_info.subdomain.lower() 
                                    for brand in ['paypal', 'microsoft', 'apple', 'google', 'amazon'])
            
            return {
                'url': url,
                'domain': f"{domain_info.domain}.{domain_info.suffix}",
                'has_ip_address': has_ip,
                'has_suspicious_chars': has_suspicious_chars,
                'is_long_url': is_long_url,
                'has_brand_hijacking': has_brand_hijacking,
                'is_https': parsed.scheme == 'https'
            }
        except Exception as e:
            return {'url': url, 'error': str(e)}
    
    def check_suspicious_urls(self, body_text):
        #Extract and analyze URLs in email body"""
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body_text)
        
        url_analysis = []
        for url in urls:
            analysis = self.analyze_single_url(url)
            url_analysis.append(analysis)
        
        return url_analysis
    
    #Body analysis: 
    def analyze_email_content(self, subject, body):
        """Analyze email content for phishing indicators"""
        #Urgency keywords
        urgency_keywords = ['urgent', 'immediately', 'asap', 'important', 'action required',
                            'verify', 'confirm', 'update', 'security', 'alert']
        
        #Request keywords
        request_keywords = ['click', 'link', 'password', 'account', 'login', 'credentials',
                            'bank', 'payment', 'invoice', 'suspend', 'limit']
        
        #Threat keywords
        threat_keywords = ['suspend', 'close', 'terminate', 'locked', 'expire']
        
        subject_lower = subject.lower()
        body_lower = body.lower()
        
        analysis = {
            'urgency_score': sum(1 for word in urgency_keywords if word in subject_lower or word in body_lower),
            'request_score': sum(1 for word in request_keywords if word in body_lower),
            'threat_score': sum(1 for word in threat_keywords if word in body_lower),
            'has_urgent_subject': any(word in subject_lower for word in urgency_keywords),
            'has_requests': any(word in body_lower for word in request_keywords),
            'has_threats': any(word in body_lower for word in threat_keywords),
            'subject_exclamation': subject.count('!'),
            'subject_uppercase_ratio': sum(1 for c in subject if c.isupper()) / len(subject) if subject else 0
        }
        return analysis
    
    def detect_phishing(self, email_data):
        """Main function to detect phishing using non-ML methods"""
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        sender_email = email_data.get('sender_email', '')
        
        scores = {}
        
        #Email validation
        is_valid_email, email_msg = self.validate_sender_email(sender_email)
        scores['email_valid'] = 1 if is_valid_email else 0
        scores['suspicious_domain'] = 1 if self.check_suspicious_domains(sender_email) else 0
        
        #Content analysis
        content_analysis = self.analyze_email_content(subject, body)
        scores.update(content_analysis)
        
        #Spelling and grammar
        spelling_errors, misspelled_words = self.check_spelling_errors(body)
        grammar_errors, grammar_count = self.check_grammar_errors(body)
        
        scores['spelling_error_ratio'] = spelling_errors
        scores['grammar_error_density'] = grammar_errors
        scores['has_spelling_errors'] = 1 if spelling_errors > 0.1 else 0  # >10% errors
        
        #URL analysis
        url_analysis = self.check_suspicious_urls(body)
        scores['num_urls'] = len(url_analysis)
        
        if url_analysis:
            suspicious_urls = [url for url in url_analysis 
                                if url.get('has_ip_address') or 
                                url.get('has_suspicious_chars') or
                                url.get('has_brand_hijacking')]
            scores['suspicious_url_ratio'] = len(suspicious_urls) / len(url_analysis)
        else:
            scores['suspicious_url_ratio'] = 0
        
        #Calculate overall phishing score (0-1)
        phishing_score = self.calculate_overall_score(scores)
        
        return {
            'phishing_score': phishing_score,
            'is_phishing': phishing_score > 0.6,
            'detailed_scores': scores,
            'url_analysis': url_analysis,
            'misspelled_words': misspelled_words
        }
    
    def calculate_overall_score(self, scores):
        """Calculate overall phishing probability"""
        weights = {
            'email_valid': 0.1,
            'suspicious_domain': 0.15,
            'urgency_score': 0.1,
            'request_score': 0.15,
            'threat_score': 0.1,
            'has_spelling_errors': 0.1,
            'spelling_error_ratio': 0.1,
            'suspicious_url_ratio': 0.2
        }
        
        total_score = 0
        for key, weight in weights.items():
            total_score += scores.get(key, 0) * weight
        
        return min(total_score, 1.0)

if __name__ == "__main__":
    detector = NonMLPhishingDetector()
    
    #Test
    email_data = {
        'subject': "Urgent: Verify Your Account Now!",
        'body': """Dear User,

        We have detected suspicious activity on your account. Please click the link below to verify your information immediately to avoid suspension.

        http://secure-login.example.com/verify?user=12345

        Thank you,
        Support Team""",
        'sender_email': 'security@abc-update.com'
    }
    
    result = detector.detect_phishing(email_data)
    print("Result:", result) 