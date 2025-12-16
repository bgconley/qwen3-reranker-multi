# Input variables for Lambda Cloud deployment
# Qwen3-Reranker with Tailscale networking

# =============================================================================
# Required Variables
# =============================================================================

variable "lambda_api_key" {
  description = "Lambda Labs API key (can also be set via LAMBDALABS_API_KEY env var)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "ssh_key_name" {
  description = "Name of the SSH key registered in Lambda Cloud (required)"
  type        = string
}

variable "tailscale_auth_key" {
  description = "Tailscale auth key for unattended setup (from https://login.tailscale.com/admin/settings/keys)"
  type        = string
  sensitive   = true
}

variable "ssh_private_key_path" {
  description = "Path to the SSH private key file for provisioning"
  type        = string
  default     = "~/.ssh/id_ed25519"
}

# =============================================================================
# Optional Variables - Instance Configuration
# =============================================================================

variable "instance_name" {
  description = "Name for the Lambda Cloud instance"
  type        = string
  default     = "qwen3-reranker"
}

variable "instance_type" {
  description = "Lambda Cloud instance type (gpu_1x_a10, gpu_1x_h100_pcie, gpu_1x_a100_sxm, etc.)"
  type        = string
  default     = "gpu_1x_a10"
}

variable "region" {
  description = "Lambda Cloud region (us-west-1, us-east-1, us-south-1, europe-central-1, asia-northeast-1)"
  type        = string
  default     = "us-west-1"
}

# =============================================================================
# Optional Variables - Reranker Configuration
# =============================================================================

variable "reranker_profile" {
  description = "Reranker profile to use (qwen3_4b_cuda, qwen3_4b_vllm, etc.)"
  type        = string
  default     = "qwen3_4b_cuda"
}

variable "reranker_port" {
  description = "Port for the reranker service"
  type        = number
  default     = 9003
}

variable "reranker_backend" {
  description = "Backend to use (pytorch, vllm, auto)"
  type        = string
  default     = "pytorch"
}

# =============================================================================
# Optional Variables - Git Repository
# =============================================================================

variable "git_repo_url" {
  description = "Git repository URL to clone"
  type        = string
  default     = "https://github.com/yourusername/qwen3-reranker-multi.git"
}

variable "git_branch" {
  description = "Git branch to checkout"
  type        = string
  default     = "master"
}

# =============================================================================
# Optional Variables - Tailscale
# =============================================================================

variable "tailscale_hostname" {
  description = "Hostname for the Tailscale node"
  type        = string
  default     = "qwen3-reranker"
}

variable "tailscale_tags" {
  description = "Tailscale tags to apply (e.g., tag:servers)"
  type        = list(string)
  default     = []
}

# =============================================================================
# Optional Variables - Persistent Storage
# =============================================================================

variable "create_filesystem" {
  description = "Whether to create a persistent filesystem for model cache"
  type        = bool
  default     = true
}

variable "filesystem_name" {
  description = "Name for the Lambda persistent filesystem"
  type        = string
  default     = "qwen3-reranker-cache"
}

