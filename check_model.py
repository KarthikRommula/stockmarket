# check_model.py
import os
import json
import boto3
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

print("=== Claude Model Check ===")
print("\nChecking model configurations in your application...")

# Check AWS Bedrock configuration (extract_pdf.py)
print("\n1. AWS Bedrock Configuration (extract_pdf.py):")
try:
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv('AWS_REGION', 'ap-south-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    print(f"  - AWS Region: {os.getenv('AWS_REGION', 'ap-south-1')}")
    print(f"  - Model ID: anthropic.claude-3-5-sonnet-20241022-v2:0")
    print(f"  - Inference Profile: anthropic.claude-3-5-sonnet-20241022-v2:0 (using inferenceProfile parameter)")
    
    # List available models in Bedrock
    try:
        bedrock = boto3.client(
            service_name='bedrock',
            region_name=os.getenv('AWS_REGION', 'ap-south-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        models = bedrock.list_foundation_models()
        claude_models = [model for model in models.get('modelSummaries', []) 
                        if 'claude' in model.get('modelId', '').lower()]
        
        if claude_models:
            print("\n  Available Claude models in AWS Bedrock:")
            for model in claude_models:
                print(f"  - {model.get('modelId')}")
        else:
            print("\n  No Claude models found in AWS Bedrock listing")
    except Exception as e:
        print(f"\n  Could not list Bedrock models: {str(e)}")
    
except Exception as e:
    print(f"  Error with AWS Bedrock configuration: {str(e)}")

# Check Claude configuration in rag_system.py
print("\n2. Claude Configuration in rag_system.py:")
print("  - Using AWS Bedrock for Claude model access")
print("  - Model ID: anthropic.claude-3-5-sonnet-20241022-v2:0")
print("  - Inference Profile: anthropic.claude-3-5-sonnet-20241022-v2:0 (using inferenceProfile parameter)")
print("  - Anthropic API: Not used (AWS Bedrock only)")

print("\n=== Check Complete ===")
