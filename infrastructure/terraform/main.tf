# Lambda Cloud GPU Instance for Qwen3-Reranker
# With Tailscale networking for secure, direct access

# =============================================================================
# Lambda Cloud Instance
# =============================================================================

resource "lambdalabs_instance" "reranker" {
  name               = var.instance_name
  region_name        = var.region
  instance_type_name = var.instance_type
  ssh_key_names      = [var.ssh_key_name]
  file_system_names  = local.attach_filesystem ? [var.filesystem_name] : []

  # SSH connection for provisioning
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file(pathexpand(var.ssh_private_key_path))
    host        = self.ip
    timeout     = "30m"
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

  # Upload the Tailscale auth key to the instance (content never appears in logs).
  provisioner "file" {
    content     = local.tailscale_auth_key
    destination = "/tmp/tailscale-auth-key"
  }

  # Run the setup script with environment variables
  # on_failure = continue keeps instance alive for debugging if provisioning fails
  provisioner "remote-exec" {
    on_failure = continue
    inline = [
      "chmod +x /tmp/setup-instance.sh",
      "chmod 600 /tmp/tailscale-auth-key",
      "TAILSCALE_AUTH_KEY_FILE='/tmp/tailscale-auth-key' TAILSCALE_HOSTNAME='${var.tailscale_hostname}' TAILSCALE_TAGS='${local.tailscale_tags_csv}' GIT_REPO_URL='${var.git_repo_url}' GIT_BRANCH='${var.git_branch}' RERANKER_PROFILE='${var.reranker_profile}' RERANKER_PORT='${var.reranker_port}' RERANKER_BACKEND='${var.reranker_backend}' FILESYSTEM_NAME='${local.filesystem_name_for_script}' /tmp/setup-instance.sh",
    ]
  }

  # Always show the tail of the setup log to make failures actionable.
  provisioner "remote-exec" {
    on_failure = continue
    inline = [
      "echo '--- /tmp/setup-instance.log (tail) ---'",
      "sudo tail -n 200 /tmp/setup-instance.log || true",
      "echo '--- end /tmp/setup-instance.log ---'",
    ]
  }
}

# Fail the apply if the service isn't actually up, while still keeping the instance
# around for debugging (this is a separate resource, so failures here don't destroy
# the instance).
resource "null_resource" "verify_reranker" {
  depends_on = [lambdalabs_instance.reranker]

  triggers = {
    instance_id = lambdalabs_instance.reranker.id
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file(pathexpand(var.ssh_private_key_path))
    host        = lambdalabs_instance.reranker.ip
    timeout     = "10m"
  }

  provisioner "remote-exec" {
    inline = [
      "echo 'Verifying reranker readiness on localhost...'",
      "command -v curl >/dev/null 2>&1 || (echo 'curl missing; setup likely failed early' && exit 1)",
      "curl -sf --max-time 5 http://127.0.0.1:${var.reranker_port}/ready >/dev/null || (echo 'Reranker not ready; dumping logs.'; sudo tail -n 200 /tmp/setup-instance.log || true; sudo journalctl -u qwen3-reranker -n 200 --no-pager || true; exit 1)",
      "echo 'OK: /ready returned 200'",
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
  # Keep secrets out of provisioner commands so Terraform won't suppress logs.
  # Prefer tailscale_auth_key_path to avoid storing the auth key in Terraform state.
  tailscale_auth_key = var.tailscale_auth_key != "" ? var.tailscale_auth_key : sensitive(trimspace(file(pathexpand(var.tailscale_auth_key_path))))

  # Tailscale MagicDNS hostname (assumes default tailnet domain)
  tailscale_fqdn = var.tailscale_hostname

  tailscale_tags_csv         = join(",", var.tailscale_tags)
  attach_filesystem          = coalesce(var.attach_filesystem, var.create_filesystem)
  filesystem_name_for_script = local.attach_filesystem ? var.filesystem_name : ""

  # Reranker service URL via Tailscale
  reranker_url = "http://${local.tailscale_fqdn}:${var.reranker_port}"
}
