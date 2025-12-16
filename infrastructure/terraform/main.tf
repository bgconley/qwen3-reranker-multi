# Lambda Cloud GPU Instance for Qwen3-Reranker
# With Tailscale networking for secure, direct access

# =============================================================================
# Persistent Filesystem (for model cache)
# =============================================================================
#
# The filesystem persists across instance destroy/recreate cycles.
#
# To destroy ONLY the instance (keep filesystem):
#   terraform destroy -target=lambdalabs_instance.reranker
#
# To destroy the filesystem (when you really want to):
#   1. Set prevent_destroy = false below
#   2. terraform destroy
#   3. Set prevent_destroy = true again
#
resource "lambdalabs_filesystem" "model_cache" {
  count  = var.create_filesystem ? 1 : 0
  name   = var.filesystem_name
  region = var.region

  lifecycle {
    prevent_destroy = true
  }
}

# =============================================================================
# Lambda Cloud Instance
# =============================================================================

resource "lambdalabs_instance" "reranker" {
  name               = var.instance_name
  region_name        = var.region
  instance_type_name = var.instance_type
  ssh_key_names      = [var.ssh_key_name]
  file_system_names  = var.create_filesystem ? [lambdalabs_filesystem.model_cache[0].name] : []

  # SSH connection for provisioning
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file(pathexpand(var.ssh_private_key_path))
    host        = self.ip
    timeout     = "10m"
  }

  # Wait for cloud-init to complete
  provisioner "remote-exec" {
    inline = [
      "echo 'Waiting for cloud-init to complete...'",
      "cloud-init status --wait || true",
      "echo 'Cloud-init complete.'",
    ]
  }

  # Copy setup scripts to the instance
  provisioner "file" {
    source      = "${path.module}/../scripts/setup-instance.sh"
    destination = "/tmp/setup-instance.sh"
  }

  # Run the setup script with environment variables
  # on_failure = continue keeps instance alive for debugging if provisioning fails
  provisioner "remote-exec" {
    on_failure = continue
    inline = [
      "chmod +x /tmp/setup-instance.sh",
      "TAILSCALE_AUTH_KEY='${var.tailscale_auth_key}' TAILSCALE_HOSTNAME='${var.tailscale_hostname}' TAILSCALE_TAGS='${join(",", var.tailscale_tags)}' GIT_REPO_URL='${var.git_repo_url}' GIT_BRANCH='${var.git_branch}' RERANKER_PROFILE='${var.reranker_profile}' RERANKER_PORT='${var.reranker_port}' RERANKER_BACKEND='${var.reranker_backend}' FILESYSTEM_NAME='${var.create_filesystem ? var.filesystem_name : ""}' /tmp/setup-instance.sh",
    ]
  }
}

# =============================================================================
# Data Sources (for reference)
# =============================================================================

# You can use this data source to see available instance types
# data "lambdalabs_instance_types" "available" {}

# =============================================================================
# Local Values
# =============================================================================

locals {
  # Tailscale MagicDNS hostname (assumes default tailnet domain)
  tailscale_fqdn = var.tailscale_hostname

  # Reranker service URL via Tailscale
  reranker_url = "http://${local.tailscale_fqdn}:${var.reranker_port}"
}
