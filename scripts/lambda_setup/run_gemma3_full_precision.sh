#!/bin/bash
# Script to run Gemma3 training on Lambda with FULL PRECISION (no quantization)
# This version uses more GPU memory but may achieve better results

# Parse command-line arguments with defaults
EPOCHS=${1:-4}  # Default: 4 epochs if not specified
BATCH_SIZE=${2:-4}  # Default: batch size 4 if not specified
MODEL_SIZE=${3:-"4b"}  # Default: 4b model if not specified (options: 1b, 4b)

# Define variables
LAMBDA_IP="150.136.115.159"
SSH_KEY="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/Lambda-ssh.pem"
LAMBDA_USER="ubuntu"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/lambda_training_gemma3_full_precision_${TIMESTAMP}.log"
SCRIPT_PATH="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/efficient_rl_simplified_withoutquantization.py"

# Determine the full model ID based on the model size
if [ "$MODEL_SIZE" == "1b" ]; then
  MODEL_ID="google/gemma-3-1b-it"
else
  MODEL_ID="google/gemma-3-4b-it"  # Default to 4b model
fi

# Display training configuration
echo "=== Full Precision Training Configuration ==="
echo "Epochs:     $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Model:      $MODEL_ID (${MODEL_SIZE} parameters)"
echo "================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# First, transfer the full precision script to Lambda
echo "Transferring full precision training script to Lambda..."
scp -i $SSH_KEY $SCRIPT_PATH $LAMBDA_USER@$LAMBDA_IP:/home/ubuntu/req-tracer-file-system/req_tracing/
echo "Transfer complete!"

# Connect to Lambda and run training with logging
ssh -i $SSH_KEY $LAMBDA_USER@$LAMBDA_IP << REMOTE_EOF
echo "Connected to Lambda instance"

# Create log directory if it doesn't exist
mkdir -p /home/ubuntu/req-tracer-file-system/training_logs

# Activate the environment
source /home/ubuntu/req-tracer-file-system/activate_env.sh

# Navigate to the project directory
cd /home/ubuntu/req-tracer-file-system/req_tracing

# Run the training with the specified parameters and capture output to both terminal and log file
echo "Starting FULL PRECISION training for gemma3 with batch size $BATCH_SIZE and epochs $EPOCHS..."
echo "Logging output to /home/ubuntu/req-tracer-file-system/training_logs/${TIMESTAMP}.log"

# First, update the model ID in the Python script
sed -i "s|\"id\": \"google/gemma-3-1b-it\"|\"id\": \"$MODEL_ID\"|g" efficient_rl_simplified_withoutquantization.py

# Use tee to capture output to both terminal and file
python efficient_rl_simplified_withoutquantization.py --model gemma3 --epochs $EPOCHS --batch-size $BATCH_SIZE 2>&1 | tee /home/ubuntu/req-tracer-file-system/training_logs/${TIMESTAMP}.log

# Check exit status
if [ \${PIPESTATUS[0]} -eq 0 ]; then
  echo "Training completed successfully!" | tee -a /home/ubuntu/req-tracer-file-system/training_logs/${TIMESTAMP}.log
else
  echo "Training encountered errors. Check the log file for details." | tee -a /home/ubuntu/req-tracer-file-system/training_logs/${TIMESTAMP}.log
fi

echo "Log file saved at: /home/ubuntu/req-tracer-file-system/training_logs/${TIMESTAMP}.log"
REMOTE_EOF

# Copy the log file back to local machine
echo "Copying log file from Lambda to local machine..."
scp -i $SSH_KEY $LAMBDA_USER@$LAMBDA_IP:/home/ubuntu/req-tracer-file-system/training_logs/${TIMESTAMP}.log ./${LOG_FILE}

echo "Log file saved locally as: ${LOG_FILE}"
