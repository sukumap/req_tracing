#!/bin/bash
# Script to run simplified Gemma fine-tuning on Lambda server

# Default parameters
MODEL_SIZE=${1:-"4b"}  # Default to 4b if not specified
MAX_EXAMPLES=${2:-50}  # Default to 50 examples for debugging
EPOCHS=${3:-3}         # Default to 3 epochs if not specified
BATCH_SIZE=${4:-2}     # Default to batch size 2 if not specified

# SSH key location (absolute path)
SSH_KEY="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/Lambda-ssh.pem"

# Lambda server details
LAMBDA_USER="ubuntu"
LAMBDA_IP="150.136.115.159"

# Local and remote paths
LOCAL_SCRIPT_DIR="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing"
REMOTE_BASE_DIR="/home/ubuntu/req-tracer-file-system"
REMOTE_SCRIPT_DIR="${REMOTE_BASE_DIR}/req_tracing"
REMOTE_LOG_DIR="${REMOTE_BASE_DIR}/training_logs"
REMOTE_MODEL_DIR="${REMOTE_BASE_DIR}/models"
LOCAL_LOG_DIR="${LOCAL_SCRIPT_DIR}/logs"

# Generate timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_LOG_FILE="${REMOTE_LOG_DIR}/finetune_simple_${MODEL_SIZE}_${TIMESTAMP}.log"
LOCAL_LOG_FILE="${LOCAL_LOG_DIR}/lambda_finetune_simple_${MODEL_SIZE}_${TIMESTAMP}.log"

echo "Preparing to run simplified Gemma fine-tuning on Lambda..."
echo "Model size: ${MODEL_SIZE}"
echo "Max examples: ${MAX_EXAMPLES}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key not found at $SSH_KEY"
    exit 1
fi

# Create remote directories if they don't exist
echo "Creating remote directories..."
ssh -i "$SSH_KEY" ${LAMBDA_USER}@${LAMBDA_IP} "mkdir -p ${REMOTE_SCRIPT_DIR} ${REMOTE_LOG_DIR} ${REMOTE_MODEL_DIR}/gemma_lora_weights"

# Copy the fine-tuning script to Lambda
echo "Copying fine-tuning script to Lambda..."
scp -i "$SSH_KEY" ${LOCAL_SCRIPT_DIR}/gemma_finetune_simple.py ${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_SCRIPT_DIR}/

# Copy the training data to Lambda if it exists
if [ -f "${LOCAL_SCRIPT_DIR}/training_data.jsonl" ]; then
    echo "Copying training data to Lambda..."
    scp -i "$SSH_KEY" ${LOCAL_SCRIPT_DIR}/training_data.jsonl ${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_SCRIPT_DIR}/
else
    echo "Warning: training_data.jsonl not found at ${LOCAL_SCRIPT_DIR}"
    echo "Please ensure the training data is available on Lambda or the fine-tuning will fail."
fi

# Run the fine-tuning script on Lambda
echo "Starting fine-tuning on Lambda..."
ssh -i "$SSH_KEY" ${LAMBDA_USER}@${LAMBDA_IP} << REMOTE_EOF
echo "Connected to Lambda instance"

# Create log directory if it doesn't exist
mkdir -p ${REMOTE_LOG_DIR}

# Activate the environment
source /home/ubuntu/req-tracer-file-system/activate_env.sh

# Navigate to the project directory
cd ${REMOTE_SCRIPT_DIR}

# Run the fine-tuning with the specified parameters and capture output to both terminal and log file
echo "Starting simplified fine-tuning for Gemma ${MODEL_SIZE} with batch size ${BATCH_SIZE} and epochs ${EPOCHS}..."
echo "Logging output to ${REMOTE_LOG_FILE}"

# Make the script executable
chmod +x gemma_finetune_simple.py

# Use tee to capture output to both terminal and file
python gemma_finetune_simple.py \
    --model-size ${MODEL_SIZE} \
    --max-examples ${MAX_EXAMPLES} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --save-dir ${REMOTE_MODEL_DIR}/gemma_lora_weights \
    2>&1 | tee ${REMOTE_LOG_FILE}

# Check exit status
if [ \${PIPESTATUS[0]} -eq 0 ]; then
  echo "Fine-tuning completed successfully!" | tee -a ${REMOTE_LOG_FILE}
else
  echo "Fine-tuning encountered errors. Check the log file for details." | tee -a ${REMOTE_LOG_FILE}
fi

echo "Log file saved at: ${REMOTE_LOG_FILE}"
REMOTE_EOF

# Check if the fine-tuning was successful
if [ $? -eq 0 ]; then
    echo "Fine-tuning completed successfully!"
else
    echo "Fine-tuning failed with an error."
fi

# Copy the log file from Lambda to local machine
echo "Copying log file from Lambda to local machine..."
scp -i "$SSH_KEY" ${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_LOG_FILE} ${LOCAL_LOG_FILE}

echo "Log file saved locally as: ${LOCAL_LOG_FILE}"

# Copy the trained model from Lambda to local machine
echo "Copying trained model from Lambda to local machine..."
mkdir -p ${LOCAL_SCRIPT_DIR}/models/gemma_lora_weights
scp -i "$SSH_KEY" -r ${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_MODEL_DIR}/gemma_lora_weights/epoch_* ${LOCAL_SCRIPT_DIR}/models/gemma_lora_weights/ || echo "No model files to copy"

echo "Fine-tuning process complete!"
