#!/bin/bash
# =============================================================================
# Tailscale Installation Script
# =============================================================================
# Standalone script to install and configure Tailscale on Ubuntu
#
# Usage:
#   ./install-tailscale.sh <auth-key> [hostname] [tags]
#
# Arguments:
#   auth-key  - Tailscale auth key (required)
#   hostname  - Hostname for the node (default: hostname of machine)
#   tags      - Comma-separated tags (optional, e.g., "tag:servers,tag:gpu")
#
# Example:
#   ./install-tailscale.sh tskey-auth-xxx qwen3-reranker tag:servers
# =============================================================================

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
AUTH_KEY="${1:-}"
HOSTNAME="${2:-$(hostname)}"
TAGS="${3:-}"

if [ -z "$AUTH_KEY" ]; then
    log_error "Usage: $0 <auth-key> [hostname] [tags]"
    log_error "  auth-key: Tailscale auth key (required)"
    log_error "  hostname: Hostname for the node (default: machine hostname)"
    log_error "  tags:     Comma-separated tags (optional)"
    exit 1
fi

log_info "Installing Tailscale..."
log_info "  Hostname: ${HOSTNAME}"
log_info "  Tags: ${TAGS:-none}"

# Check if Tailscale is already installed
if command -v tailscale &> /dev/null; then
    log_warn "Tailscale is already installed."
    TAILSCALE_VERSION=$(tailscale version 2>/dev/null | head -n1 || echo "unknown")
    log_info "Current version: ${TAILSCALE_VERSION}"
else
    # Install Tailscale using the official script
    log_info "Downloading and installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# Ensure tailscaled is running
log_info "Starting tailscaled service..."
sudo systemctl enable tailscaled
sudo systemctl start tailscaled

# jq is optional but improves status parsing
if ! command -v jq &> /dev/null; then
    log_info "Installing jq for status parsing..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq jq
fi

# Wait for tailscaled to be ready
sleep 2

# Build the tailscale up command
TAILSCALE_CMD="sudo tailscale up --auth-key=${AUTH_KEY} --hostname=${HOSTNAME}"

# Add tags if specified
if [ -n "$TAGS" ]; then
    TAILSCALE_CMD="${TAILSCALE_CMD} --advertise-tags=${TAGS}"
fi

# Enable Tailscale SSH for easier management
TAILSCALE_CMD="${TAILSCALE_CMD} --ssh"

# Join the tailnet
log_info "Joining Tailscale network..."
eval $TAILSCALE_CMD

# Wait for connection
sleep 3

# Get connection status
STATUS=$(tailscale status --json 2>/dev/null | jq -r '.BackendState // "unknown"' 2>/dev/null || echo "unknown")
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "pending")

if [ "$STATUS" = "Running" ] || [ "$TAILSCALE_IP" != "pending" ]; then
    log_info "Tailscale connected successfully!"
    echo ""
    echo "============================================================"
    echo "  Tailscale Connection Details"
    echo "============================================================"
    echo "  Hostname: ${HOSTNAME}"
    echo "  IPv4:     ${TAILSCALE_IP}"
    echo "  Status:   ${STATUS}"
    echo ""
    echo "  View in admin console:"
    echo "    https://login.tailscale.com/admin/machines"
    echo "============================================================"
else
    log_warn "Tailscale may still be connecting. Check status with: tailscale status"
fi
