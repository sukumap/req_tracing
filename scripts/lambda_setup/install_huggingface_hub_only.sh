#!/bin/bash
# Script to install huggingface_hub on Lambda server

# Define variables
LAMBDA_USER="ubuntu"
LAMBDA_IP="150.136.115.159"
SSH_KEY="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/Lambda-ssh.pem"

echo "Installing huggingface_hub on Lambda server..."

# SSH into Lambda and install huggingface_hub
ssh -i "$SSH_KEY" "$LAMBDA_USER@$LAMBDA_IP" << 'EOF'
  echo "Activating conda environment..."
  source /home/ubuntu/req-tracer-file-system/miniconda/etc/profile.d/conda.sh
  conda activate req_trace_env
  
  echo "Installing huggingface_hub..."
  # First uninstall if it exists with issues
  pip uninstall -y huggingface_hub
  
  # Then install the specific version
  pip install huggingface_hub==0.30.2
  
  # Verify installation
  if pip list | grep -q "huggingface_hub"; then
    version=$(pip list | grep "huggingface_hub" | awk '{print $2}')
    echo "huggingface_hub $version installed successfully"
  else
    echo "Failed to install huggingface_hub"
    exit 1
  fi
EOF

echo "huggingface_hub installation completed."
