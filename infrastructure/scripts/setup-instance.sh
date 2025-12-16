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

# Log everything to a file for debugging (Terraform suppresses output due to sensitive values)
LOGFILE="/tmp/setup-instance.log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "=== Setup started at $(date) ==="

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

# Better failure diagnostics without leaking secrets
on_error() {
    local exit_code=$?
    log_error "Setup failed (exit=${exit_code}) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
    log_error "See full log: ${LOGFILE}"
    exit "$exit_code"
}
trap on_error ERR

# =============================================================================
# Configuration
# =============================================================================

TAILSCALE_AUTH_KEY="${TAILSCALE_AUTH_KEY:-}"
TAILSCALE_AUTH_KEY_FILE="${TAILSCALE_AUTH_KEY_FILE:-}"
TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-qwen3-reranker}"
TAILSCALE_TAGS="${TAILSCALE_TAGS:-}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/yourusername/qwen3-reranker-multi.git}"
GIT_BRANCH="${GIT_BRANCH:-master}"
RERANKER_PROFILE="${RERANKER_PROFILE:-qwen3_4b_cuda}"
RERANKER_PORT="${RERANKER_PORT:-9003}"
RERANKER_BACKEND="${RERANKER_BACKEND:-pytorch}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
FILESYSTEM_NAME="${FILESYSTEM_NAME:-}"

INSTALL_DIR="/home/ubuntu/qwen3-reranker-multi"
VENV_DIR="${INSTALL_DIR}/.venv"

FILESYSTEM_MOUNT=""
HF_CACHE_DIR=""

# =============================================================================
# Validation
# =============================================================================

# Support reading secrets from a file so Terraform can keep logs visible.
if [ -z "$TAILSCALE_AUTH_KEY" ] && [ -n "$TAILSCALE_AUTH_KEY_FILE" ]; then
    if [ ! -f "$TAILSCALE_AUTH_KEY_FILE" ]; then
        log_error "TAILSCALE_AUTH_KEY_FILE was set but file not found: ${TAILSCALE_AUTH_KEY_FILE}"
        exit 1
    fi
    TAILSCALE_AUTH_KEY="$(tr -d '\n' < "$TAILSCALE_AUTH_KEY_FILE")"
fi

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
    log_info "  Filesystem: ${FILESYSTEM_NAME} (mount point auto-detected)"
else
    log_info "  Filesystem: none (using ephemeral storage)"
fi

# =============================================================================
# Step 1: System Updates
# =============================================================================

log_info "Step 1/7: Updating system packages..."

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    pkg-config \
    jq

log_info "System packages updated."

# =============================================================================
# Step 2: Install Tailscale
# =============================================================================

log_info "Step 2/7: Installing Tailscale..."

# Install Tailscale if needed
if ! command -v tailscale >/dev/null 2>&1; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# Ensure tailscaled is running
sudo systemctl enable --now tailscaled

# Skip auth if already connected
EXISTING_IP="$(tailscale ip -4 2>/dev/null || true)"
if [ -n "$EXISTING_IP" ]; then
    log_info "Tailscale already connected. IP: ${EXISTING_IP}"
else
    # Build the tailscale up command
    TAILSCALE_ARGS=(up "--auth-key=${TAILSCALE_AUTH_KEY}" "--hostname=${TAILSCALE_HOSTNAME}" "--ssh")

    # Add tags if specified
    if [ -n "$TAILSCALE_TAGS" ]; then
        TAILSCALE_ARGS+=("--advertise-tags=${TAILSCALE_TAGS}")
    fi

    # Join the tailnet
    log_info "Joining Tailscale network..."
    sudo tailscale "${TAILSCALE_ARGS[@]}"

    # Wait for Tailscale to connect
    for _ in {1..15}; do
        TAILSCALE_IP="$(tailscale ip -4 2>/dev/null || true)"
        if [ -n "$TAILSCALE_IP" ]; then
            break
        fi
        sleep 2
    done

    TAILSCALE_IP="${TAILSCALE_IP:-pending}"
    log_info "Tailscale connected. IP: ${TAILSCALE_IP}"
fi

# Cleanup one-time auth key material if we were given a temp file.
if [ -n "$TAILSCALE_AUTH_KEY_FILE" ] && [[ "$TAILSCALE_AUTH_KEY_FILE" == /tmp/* ]]; then
    rm -f "$TAILSCALE_AUTH_KEY_FILE" || true
fi
unset TAILSCALE_AUTH_KEY

# =============================================================================
# Step 3: Clone Repository
# =============================================================================

log_info "Step 3/7: Cloning repository..."

if [[ "$GIT_REPO_URL" == "https://github.com/yourusername/qwen3-reranker-multi.git" ]]; then
    log_warn "GIT_REPO_URL is still the placeholder value. Set git_repo_url in terraform.tfvars."
fi

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

log_info "Step 4/7: Setting up Python environment..."

PYTHON_CMD="python3"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade packaging tooling
pip install --upgrade pip setuptools wheel

# Choose dependency extras based on backend (avoid flash-attn by default for reliability)
EXTRAS="pytorch"
case "$RERANKER_BACKEND" in
    vllm) EXTRAS="vllm" ;;
    pytorch|auto) EXTRAS="pytorch" ;;
    *)
        log_warn "Unknown RERANKER_BACKEND '${RERANKER_BACKEND}', defaulting to pytorch"
        EXTRAS="pytorch"
        ;;
esac

log_info "Installing dependencies: .[${EXTRAS}] (this may take a few minutes)..."
pip install -e ".[${EXTRAS}]"

# If a GPU is present, ensure we have a CUDA-enabled PyTorch build.
if command -v nvidia-smi >/dev/null 2>&1; then
    log_info "NVIDIA GPU detected:"
    nvidia-smi -L || true

    TORCH_CUDA_OK="$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')")"
    if [ "$TORCH_CUDA_OK" != "1" ]; then
        log_warn "PyTorch CUDA is not available; reinstalling torch from PyTorch CUDA wheels (cu121)..."
        pip install --upgrade --force-reinstall 'torch>=2.4.0' --index-url https://download.pytorch.org/whl/cu121

        TORCH_CUDA_OK="$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')")"
        if [ "$TORCH_CUDA_OK" != "1" ]; then
            log_error "PyTorch CUDA is still not available after reinstall. Aborting to avoid a silent CPU deployment."
            python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', getattr(torch.version, 'cuda', None))"
            exit 1
        fi
    fi
else
    log_warn "No NVIDIA GPU detected (nvidia-smi missing). Continuing without CUDA verification."
fi

# Optional: attempt Flash Attention install (non-fatal)
if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
    log_info "INSTALL_FLASH_ATTN=1; attempting to install flash-attn (best-effort)..."
    pip install --no-build-isolation "flash-attn>=2.6.0" || log_warn "flash-attn install failed; continuing without it"
fi

log_info "Python environment ready."

# =============================================================================
# Step 5: Configure Persistent Storage (if filesystem attached)
# =============================================================================

if [ -n "$FILESYSTEM_NAME" ]; then
    log_info "Step 5/7: Configuring persistent storage..."

    # Wait for filesystem to be mounted (Lambda mounts automatically, but mountpoint can vary)
    MOUNT_WAIT=0
    while [ $MOUNT_WAIT -lt 60 ]; do
        for candidate in "/home/ubuntu/${FILESYSTEM_NAME}" "/mnt/${FILESYSTEM_NAME}"; do
            if [ -d "$candidate" ]; then
                FILESYSTEM_MOUNT="$candidate"
                break 2
            fi
        done

        if command -v findmnt >/dev/null 2>&1; then
            FOUND="$(findmnt -rn -o TARGET | grep -E "/${FILESYSTEM_NAME}$" | head -n1 || true)"
            if [ -n "$FOUND" ] && [ -d "$FOUND" ]; then
                FILESYSTEM_MOUNT="$FOUND"
                break
            fi
        fi

        log_info "Waiting for filesystem to mount..."
        sleep 2
        MOUNT_WAIT=$((MOUNT_WAIT + 2))
    done

    if [ -n "$FILESYSTEM_MOUNT" ] && [ -d "$FILESYSTEM_MOUNT" ]; then
        HF_CACHE_DIR="${FILESYSTEM_MOUNT}/huggingface"
        mkdir -p "${HF_CACHE_DIR}"
        sudo chown -R ubuntu:ubuntu "${HF_CACHE_DIR}" || true
        log_info "Filesystem mounted at ${FILESYSTEM_MOUNT}"
        log_info "HuggingFace cache directory: ${HF_CACHE_DIR}"

        # Persistence verification sentinel (survives instance replacement)
        SENTINEL_FILE="${HF_CACHE_DIR}/.persistence_sentinel"
        if [ -f "$SENTINEL_FILE" ]; then
            log_info "Persistence sentinel found: $(cat "$SENTINEL_FILE" 2>/dev/null || echo "<unreadable>")"
        else
            echo "created_at=$(date -Is) hostname=$(hostname) filesystem=${FILESYSTEM_NAME}" > "$SENTINEL_FILE"
            log_info "Persistence sentinel written: ${SENTINEL_FILE}"
        fi
    else
        log_warn "Filesystem not detected after 60s, using ephemeral storage"
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

# Wait for systemd to report the unit active
for _ in {1..30}; do
    if sudo systemctl is-active --quiet qwen3-reranker; then
        break
    fi
    sleep 2
done

if ! sudo systemctl is-active --quiet qwen3-reranker; then
    log_error "Service failed to start. Checking logs..."
    sudo journalctl -u qwen3-reranker -n 200 --no-pager
    exit 1
fi

log_info "Service process is active. Waiting for readiness (model download + warmup can take several minutes)..."
READY_URL="http://127.0.0.1:${RERANKER_PORT}/ready"
READY_OK=0
for attempt in {1..60}; do
    if curl -sf --max-time 5 "$READY_URL" >/dev/null 2>&1; then
        READY_OK=1
        break
    fi

    if ! sudo systemctl is-active --quiet qwen3-reranker; then
        log_error "Service stopped while waiting for readiness. Checking logs..."
        sudo journalctl -u qwen3-reranker -n 200 --no-pager
        exit 1
    fi

    log_info "Not ready yet (attempt ${attempt}/60). Retrying..."
    sleep 20
done

if [ "$READY_OK" != "1" ]; then
    log_error "Service did not become ready in time. Checking logs..."
    sudo journalctl -u qwen3-reranker -n 200 --no-pager
    exit 1
fi

log_info "Reranker service is ready!"

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
echo "  Sentinel:           ${HF_CACHE_DIR}/.persistence_sentinel"
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

log_info "Setup complete! The service is ready."
