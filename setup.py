#!/usr/bin/env python3
"""
Setup script for Mathematical PDF Search Application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_environment():
    """Setup environment configuration."""
    env_file = Path(".env")
    env_template = Path("env_template")
    
    if not env_file.exists() and env_template.exists():
        print("📝 Creating .env file from template...")
        with open(env_template, 'r') as template:
            content = template.read()
        
        with open(env_file, 'w') as env:
            env.write(content)
        print("✅ .env file created. Please edit it with your API keys and settings.")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No env_template found. You'll need to create .env manually.")

def setup_pdf_directory():
    """Create PDF directory if it doesn't exist."""
    pdf_dir = Path("pdfs")
    if not pdf_dir.exists():
        pdf_dir.mkdir()
        print("✅ Created 'pdfs' directory")
        print("📁 Add your PDF research papers to the 'pdfs' directory")
    else:
        print("✅ 'pdfs' directory already exists")

def check_optional_dependencies():
    """Check for optional dependencies and provide guidance."""
    print("\n🔍 Checking optional dependencies...")
    
    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        print("⚠️  PyTorch not found - install for GPU acceleration")
    
    try:
        import transformers
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers required for embeddings")
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers available")
    except ImportError:
        print("❌ Sentence Transformers required for embeddings")

def main():
    """Main setup function."""
    print("🔧 Setting up Mathematical PDF Search Application\n")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Create PDF directory
    setup_pdf_directory()
    
    # Check optional dependencies
    check_optional_dependencies()
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys and configuration")
    print("2. Add PDF files to the 'pdfs' directory")
    print("3. Run the application: streamlit run MainApp.py")
    print("\n💡 For help, see README.md")

if __name__ == "__main__":
    main() 