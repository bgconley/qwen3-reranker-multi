#!/bin/bash
# =============================================================================
# Qwen3-Reranker Instance Setup Script
# =============================================================================
# This script runs on a fresh Lambda Cloud instance to:
# 1. Install Tailscale and join the tailnet
# 2. Clone the repository and install dependencies
# 3. Configure and start the reranker as a systemd service
#
# Environment variables (set by Terraform):
#   TAILSCALE_AUTH_KEY  - Tailscale auth key for unattended setup
#   TAILSCALE_HOSTNAME  - Hostname for the Tailscale node
#   TAILSCALE_TAGS      - Comma-separated Tailscale tags (optional)
#   GIT_REPO_URL        - Git repository URL to clone
#   GIT_BRANCH          - Git branch to checkout
#   RERANKER_PROFILE    - Reranker profile to use
#   RERANKER_PORT       - Port for the reranker service
#   RERANKER_BACKEND    - Backend to use (pytorch, vllm, auto)
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Configuration
# =============================================================================

TAILSCALE_AUTH_KEY="${TAILSCALE_AUTH_KEY:-}"
TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-qwen3-reranker}"
TAILSCALE_TAGS="${TAILSCALE_TAGS:-}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/yourusername/qwen3-reranker-multi.git}"
GIT_BRANCH="${GIT_BRANCH:-master}"
RERANKER_PROFILE="${RERANKER_PROFILE:-qwen3_4b_cuda}"
RERANKER_PORT="${RERANKER_PORT:-9003}"
RERANKER_BACKEND="${RERANKER_BACKEND:-pytorch}"
FILESYSTEM_NAME="${FILESYSTEM_NAME:-}"

INSTALL_DIR="/home/ubuntu/qwen3-reranker-multi"
VENV_DIR="${INSTALL_DIR}/.venv"

# Persistent storage paths (if filesystem attached)
if [ -n "$FILESYSTEM_NAME" ]; then
    FILESYSTEM_MOUNT="/home/ubuntu/${FILESYSTEM_NAME}"
    HF_CACHE_DIR="${FILESYSTEM_MOUNT}/huggingface"
else
    HF_CACHE_DIR=""
fi

# =============================================================================
# Validation
# =============================================================================

if [ -z "$TAILSCALE_AUTH_KEY" ]; then
    log_error "TAILSCALE_AUTH_KEY is required but not set"
    exit 1
fi

log_info "Starting Qwen3-Reranker setup..."
log_info "  Tailscale hostname: ${TAILSCALE_HOSTNAME}"
log_info "  Git repo: ${GIT_REPO_URL}"
log_info "  Git branch: ${GIT_BRANCH}"
log_info "  Reranker profile: ${RERANKER_PROFILE}"
log_info "  Reranker port: ${RERANKER_PORT}"
log_info "  Reranker backend: ${RERANKER_BACKEND}"
if [ -n "$FILESYSTEM_NAME" ]; then
    log_info "  Filesystem: ${FILESYSTEM_NAME} (mounted at ${FILESYSTEM_MOUNT})"
    log_info "  HF cache: ${HF_CACHE_DIR}"
else
    log_info "  Filesystem: none (using ephemeral storage)"
fi

# =============================================================================
# Step 1: System Updates
# =============================================================================

log_info "Step 1/6: Updating system packages..."

sudo apt-get update -qq
sudo apt-get install -y -qq curl git

log_info "System packages updated."

# =============================================================================
# Step 2: Install Tailscale
# =============================================================================

log_info "Step 2/6: Installing Tailscale..."

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Build the tailscale up command
TAILSCALE_CMD="sudo tailscale up --auth-key=${TAILSCALE_AUTH_KEY} --hostname=${TAILSCALE_HOSTNAME}"

# Add tags if specified
if [ -n "$TAILSCALE_TAGS" ]; then
    # Convert comma-separated tags to --advertise-tags format
    TAGS_ARG="--advertise-tags=${TAILSCALE_TAGS}"
    TAILSCALE_CMD="${TAILSCALE_CMD} ${TAGS_ARG}"
fi

# Add SSH flag to enable Tailscale SSH
TAILSCALE_CMD="${TAILSCALE_CMD} --ssh"

# Join the tailnet
log_info "Joining Tailscale network..."
eval $TAILSCALE_CMD

# Wait for Tailscale to connect
sleep 5

# Get Tailscale IP for logging
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "pending")
log_info "Tailscale connected. IP: ${TAILSCALE_IP}"

# =============================================================================
# Step 3: Clone Repository
# =============================================================================

log_info "Step 3/6: Cloning repository..."

if [ -d "$INSTALL_DIR" ]; then
    log_warn "Directory ${INSTALL_DIR} already exists, pulling latest changes..."
    cd "$INSTALL_DIR"
    git fetch origin
    git checkout "$GIT_BRANCH"
    git pull origin "$GIT_BRANCH"
else
    git clone --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

log_info "Repository cloned to ${INSTALL_DIR}"

# =============================================================================
# Step 4: Setup Python Environment
# =============================================================================

log_info "Step 4/6: Setting up Python environment..."

# Lambda Stack should have Python 3.10+ pre-installed
PYTHON_CMD="python3"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install the package with CUDA dependencies
log_info "Installing dependencies (this may take a few minutes)..."
pip install -e ".[cuda]"

log_info "Python environment ready."

# =============================================================================
# Step 5: Configure Persistent Storage (if filesystem attached)
# =============================================================================

if [ -n "$FILESYSTEM_NAME" ]; then
    log_info "Step 5/7: Configuring persistent storage..."

    # Wait for filesystem to be mounted (Lambda mounts automatically)
    MOUNT_WAIT=0
    while [ ! -d "$FILESYSTEM_MOUNT" ] && [ $MOUNT_WAIT -lt 30 ]; do
        log_info "Waiting for filesystem to mount..."
        sleep 2
        MOUNT_WAIT=$((MOUNT_WAIT + 2))
    done

    if [ -d "$FILESYSTEM_MOUNT" ]; then
        log_info "Filesystem mounted at ${FILESYSTEM_MOUNT}"

        # Create HuggingFace cache directory
        mkdir -p "${HF_CACHE_DIR}"

        log_info "HuggingFace cache directory created at ${HF_CACHE_DIR}"
    else
        log_warn "Filesystem not mounted after 30s, using ephemeral storage"
        HF_CACHE_DIR=""
    fi
else
    log_info "Step 5/7: No filesystem attached, skipping persistent storage setup"
fi

# =============================================================================
# Step 6: Configure Systemd Service
# =============================================================================

log_info "Step 6/7: Configuring systemd service..."

# Build HF cache environment line if filesystem is available
if [ -n "$HF_CACHE_DIR" ]; then
    HF_ENV_LINE="Environment=\"HF_HOME=${HF_CACHE_DIR}\""
else
    HF_ENV_LINE=""
fi

# Create systemd service file
sudo tee /etc/systemd/system/qwen3-reranker.service > /dev/null << EOF
[Unit]
Description=Qwen3-Reranker Service
After=network.target tailscaled.service
Wants=tailscaled.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${VENV_DIR}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
Environment="QWEN_RERANK_PROFILE=${RERANKER_PROFILE}"
Environment="QWEN_RERANK_BACKEND=${RERANKER_BACKEND}"
Environment="QWEN_RERANK_PORT=${RERANKER_PORT}"
Environment="QWEN_RERANK_HOST=0.0.0.0"
Environment="QWEN_RERANK_LOG_LEVEL=INFO"
Environment="QWEN_RERANK_LOG_FORMAT=json"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="TRANSFORMERS_TRUST_REMOTE_CODE=1"
${HF_ENV_LINE}
ExecStart=${VENV_DIR}/bin/python -m qwen3_reranker.api.app
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65535
LimitNPROC=65535

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable qwen3-reranker

log_info "Systemd service configured."

# =============================================================================
# Step 7: Start Service
# =============================================================================

log_info "Step 7/7: Starting reranker service..."

sudo systemctl start qwen3-reranker

# Wait for service to start
sleep 10

# Check service status
if sudo systemctl is-active --quiet qwen3-reranker; then
    log_info "Reranker service started successfully!"
else
    log_error "Service failed to start. Checking logs..."
    sudo journalctl -u qwen3-reranker -n 50 --no-pager
    exit 1
fi

# =============================================================================
# Final Summary
# =============================================================================

# Get the final Tailscale IP
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "unknown")

echo ""
echo "============================================================"
echo "  Qwen3-Reranker Setup Complete!"
echo "============================================================"
echo ""
echo "  Tailscale Hostname: ${TAILSCALE_HOSTNAME}"
echo "  Tailscale IP:       ${TAILSCALE_IP}"
echo "  Service Port:       ${RERANKER_PORT}"
if [ -n "$HF_CACHE_DIR" ]; then
echo "  HF Cache:           ${HF_CACHE_DIR} (persistent)"
else
echo "  HF Cache:           ~/.cache/huggingface (ephemeral)"
fi
echo ""
echo "  Access URLs (via Tailscale):"
echo "    Health:  http://${TAILSCALE_HOSTNAME}:${RERANKER_PORT}/health"
echo "    Healthz: http://${TAILSCALE_HOSTNAME}:${RERANKER_PORT}/healthz"
echo "    Rerank:  http://${TAILSCALE_HOSTNAME}:${RERANKER_PORT}/v1/rerank"
echo ""
echo "  Service Management:"
echo "    Status:  sudo systemctl status qwen3-reranker"
echo "    Logs:    sudo journalctl -u qwen3-reranker -f"
echo "    Restart: sudo systemctl restart qwen3-reranker"
echo ""
echo "  Configure wekadocs-matrix:"
echo "    export RERANKER_BASE_URL=http://${TAILSCALE_HOSTNAME}:${RERANKER_PORT}"
echo ""
echo "============================================================"

log_info "Setup complete! The service may take a few minutes to load the model."
