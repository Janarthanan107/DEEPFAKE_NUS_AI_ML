#!/bin/bash
# Kaggle API Setup Script

echo "======================================================================"
echo "🔑 Kaggle API Setup"
echo "======================================================================"
echo ""
echo "📋 Step 1: Get Your Kaggle API Token"
echo "----------------------------------------------------------------------"
echo "1. Open your browser and go to: https://www.kaggle.com/settings"
echo "2. Scroll down to the 'API' section"
echo "3. Click 'Create New Token' button"
echo "4. A file called 'kaggle.json' will be downloaded"
echo ""
echo "Press ENTER when you have downloaded kaggle.json..."
read

echo ""
echo "📋 Step 2: Install the API Token"
echo "----------------------------------------------------------------------"

# Create kaggle directory
mkdir -p ~/.kaggle

# Check if kaggle.json is in Downloads
if [ -f ~/Downloads/kaggle.json ]; then
    echo "✅ Found kaggle.json in Downloads!"
    mv ~/Downloads/kaggle.json ~/.kaggle/
    echo "✅ Moved to ~/.kaggle/"
else
    echo "❌ kaggle.json not found in Downloads directory"
    echo ""
    echo "Please manually move it:"
    echo "  mv /path/to/kaggle.json ~/.kaggle/"
    echo ""
    echo "Enter the full path to your kaggle.json file:"
    read KAGGLE_PATH
    
    if [ -f "$KAGGLE_PATH" ]; then
        cp "$KAGGLE_PATH" ~/.kaggle/
        echo "✅ Copied to ~/.kaggle/"
    else
        echo "❌ File not found: $KAGGLE_PATH"
        exit 1
    fi
fi

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
echo "✅ Set correct permissions (600)"

echo ""
echo "======================================================================"
echo "✅ Setup Complete!"
echo "======================================================================"
echo ""
echo "🧪 Testing API connection..."
kaggle datasets list --max-size 1000000 | head -5

echo ""
echo "======================================================================"
echo "🎉 You're ready to download datasets!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Run: python3 download_datasets.py"
echo "  2. Or manually: kaggle datasets download -d <dataset-id> --unzip"
echo ""
