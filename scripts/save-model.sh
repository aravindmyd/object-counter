#!/bin/bash
set -e  # Exit on error

# Define paths
MODEL_URL="https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz"
DOWNLOAD_DIR="tmp"
EXTRACT_DIR="tmp/rfcn_resnet101_coco_2018_01_28"
MODEL_DIR="tmp/model/resnet/1"  # TensorFlow Serving expects versioned folders

# Create required directories
mkdir -p "$DOWNLOAD_DIR" "$MODEL_DIR"

# Download the model
wget "$MODEL_URL" -O "$DOWNLOAD_DIR/model.tar.gz"

# Extract model
tar -xzvf "$DOWNLOAD_DIR/model.tar.gz" -C "$DOWNLOAD_DIR"

# Clean up tar file
rm "$DOWNLOAD_DIR/model.tar.gz"

# Ensure correct permissions
chmod -R 777 "$EXTRACT_DIR"

# Move the saved model to the correct directory
if [ -d "$EXTRACT_DIR/saved_model" ]; then
    cp -R "$EXTRACT_DIR/saved_model/"* "$MODEL_DIR/"
    echo "Copied saved model files to $MODEL_DIR"
else
    echo "Error: saved_model directory not found in expected location!"
    exit 1
fi

# Create model config file if it doesn't exist
if [ ! -f "tmp/model/model_config.config" ]; then
    cat > "tmp/model/model_config.config" << EOL
model_config_list {
  config {
    name: "resnet"
    base_path: "/models/resnet"
    model_platform: "tensorflow"
  }
}
EOL
    echo "Created model_config.config file"
fi

# Clean up extracted folder
rm -rf "$EXTRACT_DIR"

echo "Model setup complete."

