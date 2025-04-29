#!/bin/bash
# Script to run Gemma fine-tuned model inference on Lambda server

# SSH key location (absolute path)
SSH_KEY="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/Lambda-ssh.pem"

# Lambda server details
LAMBDA_USER="ubuntu"
LAMBDA_IP="150.136.115.159"

# Local and remote paths
LOCAL_SCRIPT_DIR="/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing"
REMOTE_BASE_DIR="/home/ubuntu/req-tracer-file-system"
REMOTE_SCRIPT_DIR="${REMOTE_BASE_DIR}/req_tracing"
REMOTE_LOG_DIR="${REMOTE_BASE_DIR}/inference_logs"

# Ensure the remote log directory exists
ssh -i "${SSH_KEY}" ${LAMBDA_USER}@${LAMBDA_IP} "mkdir -p ${REMOTE_LOG_DIR}"

# Sync the latest code to Lambda
echo "Syncing code to Lambda..."
rsync -avz -e "ssh -i ${SSH_KEY}" \
  --exclude="__pycache__" \
  --exclude=".git" \
  --exclude="*.pyc" \
  --exclude="venv" \
  --exclude="models/gemma_lora_weights/epoch_*/adapter_model.safetensors" \
  "${LOCAL_SCRIPT_DIR}/" \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_SCRIPT_DIR}/"

# Run the inference command on Lambda
echo "Running fine-tuned Gemma inference on Lambda..."
ssh -i "${SSH_KEY}" ${LAMBDA_USER}@${LAMBDA_IP} "cd ${REMOTE_SCRIPT_DIR} && \
  conda activate req_trace_env && \
  python src/main.py \
    --use-gemma-finetuned \
    --codebase tests/data/cpp_samples \
    --requirements tests/data/sample_requirements.xlsx \
    --output tests/data/output_gemma_finetuned_strict.xlsx \
    --mapping-method llm \
    --threshold 0.90 \
    --log-level DEBUG \
    2>&1 | tee ${REMOTE_LOG_DIR}/gemma_inference_$(date +%Y%m%d_%H%M%S).log"

# Copy back the results
echo "Copying results back from Lambda..."
rsync -avz -e "ssh -i ${SSH_KEY}" \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_SCRIPT_DIR}/tests/data/output_gemma_finetuned_strict.xlsx" \
  "${LOCAL_SCRIPT_DIR}/tests/data/"

echo "Inference complete. Results saved to tests/data/output_gemma_finetuned_strict.xlsx"
