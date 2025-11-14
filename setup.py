"""
Quick setup script for Emotion Detection Project
Run this to set up the complete project structure
"""

import os
import sys

def create_project_structure():
    """Create all necessary directories"""
    print("="*60)
    print("EMOTION DETECTION PROJECT - SETUP")
    print("="*60)
    
    directories = ['data', 'models', 'utils']
    
    print("\n1. Creating project directories...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✓ Created '{directory}/' directory")
        else:
            print(f"   ✓ '{directory}/' already exists")
    
    print("\n2. Checking for required files...")
    
    # Check for data files
    if os.path.exists('data/emotion_test.csv'):
        print("   ✓ emotion_test.csv found")
    else:
        print("   ✗ emotion_test.csv NOT found - Please add to data/ folder")
    
    if os.path.exists('data/emotion_validation.csv'):
        print("   ✓ emotion_validation.csv found")
    else:
        print("   ✗ emotion_validation.csv NOT found - Please add to data/ folder")
    
    # Check for code files
    if os.path.exists('utils/preprocessing.py'):
        print("   ✓ preprocessing.py found")
    else:
        print("   ✗ preprocessing.py NOT found")
    
    if os.path.exists('train_model.py'):
        print("   ✓ train_model.py found")
    else:
        print("   ✗ train_model.py NOT found")
    
    if os.path.exists('app.py'):
        print("   ✓ app.py found")
    else:
        print("   ✗ app.py NOT found")
    
    print("\n3. Next steps:")
    print("="*60)
    
    if not os.path.exists('data/emotion_test.csv') or not os.path.exists('data/emotion_validation.csv'):
        print("\n📁 Place your dataset files in the data/ folder:")
        print("   - emotion_test.csv")
        print("   - emotion_validation.csv")
    
    print("\n📦 Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n🎓 Train the model:")
    print("   python train_model.py")
    
    print("\n🚀 Run the Streamlit app:")
    print("   streamlit run app.py")
    
    print("\n" + "="*60)
    print("Setup complete! Follow the steps above to get started.")
    print("="*60)

if __name__ == "__main__":
    create_project_structure()
