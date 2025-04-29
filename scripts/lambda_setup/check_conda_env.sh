#!/bin/bash
# Script to check conda environments on Lambda server

# Define variables
LAMBDA_USER="ubuntu"
LAMBDA_IP="150.136.115.159"
SSH_KEY="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/Lambda-ssh.pem"

echo "Checking conda environments on Lambda server..."

# SSH into Lambda and check conda environments
ssh -i "$SSH_KEY" "$LAMBDA_USER@$LAMBDA_IP" << 'EOF'
  echo "=== Checking Conda Installation ==="
  # Check if conda exists
  if [ -f "/home/ubuntu/req-tracer-file-system/miniconda/bin/conda" ]; then
    echo "Conda found at /home/ubuntu/req-tracer-file-system/miniconda/"
    
    # Initialize conda
    source /home/ubuntu/req-tracer-file-system/miniconda/etc/profile.d/conda.sh
    
    # List all conda environments
    echo -e "\n=== Available Conda Environments ==="
    conda env list
    
    # Check if req_trace_env exists
    if conda env list | grep -q "req_trace_env"; then
      echo -e "\n=== req_trace_env Environment Found ==="
      # Activate and check packages
      conda activate req_trace_env
      echo "Python version: $(python --version)"
      echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
      
      # Check for critical packages
      echo -e "\n=== Critical Packages ==="
      for pkg in transformers peft huggingface_hub datasets accelerate bitsandbytes; do
        if pip list | grep -q "$pkg"; then
          version=$(pip list | grep "$pkg" | awk '{print $2}')
          echo "$pkg: $version (INSTALLED)"
        else
          echo "$pkg: MISSING"
        fi
      done
    else
      echo "req_trace_env environment not found"
    fi
  else
    echo "Conda not found at /home/ubuntu/req-tracer-file-system/miniconda/"
    
    # Check for alternative locations
    echo -e "\n=== Checking Alternative Locations ==="
    ls -la /home/ubuntu/req-tracer-file-system/
    
    # Check if activate_env.sh exists
    if [ -f "/home/ubuntu/req-tracer-file-system/activate_env.sh" ]; then
      echo -e "\n=== Contents of activate_env.sh ==="
      cat /home/ubuntu/req-tracer-file-system/activate_env.sh
    fi
  fi
EOF

echo "Conda environment check completed."
