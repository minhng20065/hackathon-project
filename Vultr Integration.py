import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError
import boto3
import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

def upload_to_vultr(file_path, object_name=None):
    """Upload a file to Vultr Object Storage"""
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    # Get credentials from environment
    access_key = os.getenv('FAL4ABU4TGFNGDXLL7H4')
    secret_key = os.getenv('JwMd4DGG1qcxIDtyFnC1PWDLhYtWDcxZDcTb5jDc')
    bucket = os.getenv('dataset-bucket')
    endpoint = os.getenv('https://ewr1.vultrobjects.com/')
    
    # Verify credentials exist
    if not all([access_key, secret_key, bucket, endpoint]):
        print("❌ Missing Vultr credentials in .env file")
        print("Current .env values:")
        print(f"  VULTR_ACCESS_KEY: {access_key}")
        print(f"  VULTR_SECRET_KEY: {'*' * len(secret_key) if secret_key else 'None'}")
        print(f"  VULTR_BUCKET: {bucket}")
        print(f"  VULTR_ENDPOINT: {endpoint}")
        return None
    
    try:
        # Initialize client
        s3 = boto3.client('s3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4')
        )
        
        # Upload the file
        s3.upload_file(file_path, bucket, object_name)
        print(f"✅ Successfully uploaded {file_path} as {object_name}")
        
        # Generate URL
        url = f"https://{bucket}.vultrobjects.com/{object_name}"
        print(f"📁 Access at: {url}")
        return url
        
    except ClientError as e:
        print(f"❌ Upload failed: {e}")
        return None

def upload_datasets():
    """Find and upload all CSV datasets"""
    
    # Find all CSV files automatically
    datasets = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                datasets.append(full_path)
    
    if not datasets:
        print("❌ No CSV files found!")
        return
    
    print(f"Found {len(datasets)} CSV files:")
    for dataset in datasets:
        print(f"  - {dataset}")
    
    # Upload each file
    for file_path in datasets:
        print(f"\n📤 Uploading {os.path.basename(file_path)}...")
        upload_to_vultr(file_path)

# Run the upload
upload_datasets()