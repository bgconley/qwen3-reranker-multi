# Output values for Lambda Cloud deployment
# Use these to connect to your Qwen3-Reranker instance

# =============================================================================
# Instance Information
# =============================================================================

output "instance_id" {
  description = "Lambda Cloud instance ID"
  value       = lambdalabs_instance.reranker.id
}

output "instance_name" {
  description = "Lambda Cloud instance name"
  value       = lambdalabs_instance.reranker.name
}

output "instance_type" {
  description = "Lambda Cloud instance type"
  value       = var.instance_type
}

output "region" {
  description = "Lambda Cloud region"
  value       = var.region
}

# =============================================================================
# Network Information
# =============================================================================

output "public_ip" {
  description = "Public IP address (use for SSH)"
  value       = lambdalabs_instance.reranker.ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh ubuntu@${lambdalabs_instance.reranker.ip}"
}

output "setup_log_tail_command" {
  description = "SSH command to view the setup script log on the instance"
  value       = "ssh ubuntu@${lambdalabs_instance.reranker.ip} 'sudo tail -n 200 /tmp/setup-instance.log'"
}

# =============================================================================
# Tailscale Information
# =============================================================================

output "tailscale_hostname" {
  description = "Tailscale hostname (use this to access the reranker)"
  value       = var.tailscale_hostname
}

output "tailscale_url" {
  description = "Tailscale URL for the reranker service (uses requested hostname; actual MagicDNS name may differ if there is a hostname conflict)"
  value       = "http://${var.tailscale_hostname}:${var.reranker_port}"
}

output "tailscale_discovery_command" {
  description = "SSH command to discover the actual Tailscale DNS name + IP on the instance"
  value       = "ssh ubuntu@${lambdalabs_instance.reranker.ip} 'tailscale status --json | jq -r .Self.DNSName; tailscale ip -4'"
}

# =============================================================================
# Reranker Configuration
# =============================================================================

output "reranker_url" {
  description = "URL to access the reranker service via Tailscale"
  value       = local.reranker_url
}

output "reranker_health_check" {
  description = "Command to check reranker health via Tailscale"
  value       = "curl http://${var.tailscale_hostname}:${var.reranker_port}/health"
}

output "wekadocs_config" {
  description = "Environment variable to configure wekadocs-matrix"
  value       = "export RERANKER_BASE_URL=http://${var.tailscale_hostname}:${var.reranker_port}"
}

# =============================================================================
# Persistent Storage
# =============================================================================

output "filesystem_name" {
  description = "Name of the persistent filesystem (if attached)"
  value       = local.attach_filesystem ? var.filesystem_name : "none"
}

output "filesystem_id" {
  description = "ID of the persistent filesystem (not tracked by this module)"
  value       = "not_tracked"
}

output "filesystem_mount_point" {
  description = "Mount point for the persistent filesystem"
  value       = local.attach_filesystem ? "auto-detected on instance (see /tmp/setup-instance.log)" : "none"
}

# =============================================================================
# Quick Reference
# =============================================================================

output "quick_reference" {
  description = "Quick reference commands"
  value       = <<-EOT

    ============================================================
    Qwen3-Reranker Deployment Complete!
    ============================================================

    SSH Access (public IP):
      ssh ubuntu@${lambdalabs_instance.reranker.ip}

    Tailscale Access (after joining your tailnet):
      Hostname: ${var.tailscale_hostname}
      URL: http://${var.tailscale_hostname}:${var.reranker_port}
      If the hostname doesn't resolve, discover the actual MagicDNS name:
        ssh ubuntu@${lambdalabs_instance.reranker.ip} 'tailscale status --json | jq -r .Self.DNSName; tailscale ip -4'

    Persistent Storage:
      Filesystem: ${local.attach_filesystem ? var.filesystem_name : "none (ephemeral)"}
      Mount Point: ${local.attach_filesystem ? "auto-detected on instance (see /tmp/setup-instance.log)" : "N/A"}
      HF Cache: ${local.attach_filesystem ? "auto-detected on instance (see /tmp/setup-instance.log)" : "~/.cache/huggingface (ephemeral)"}

    Health Check:
      curl http://${var.tailscale_hostname}:${var.reranker_port}/health

    Detailed Health:
      curl http://${var.tailscale_hostname}:${var.reranker_port}/healthz

    Test Rerank:
      curl -X POST http://${var.tailscale_hostname}:${var.reranker_port}/v1/rerank \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "documents": ["doc1", "doc2"]}'

    Configure wekadocs-matrix:
      export RERANKER_BASE_URL=http://${var.tailscale_hostname}:${var.reranker_port}

    View service logs:
      ssh ubuntu@${lambdalabs_instance.reranker.ip} 'sudo journalctl -u qwen3-reranker -f'

    View setup logs (provisioning):
      ssh ubuntu@${lambdalabs_instance.reranker.ip} 'sudo tail -n 200 /tmp/setup-instance.log'

    Verify filesystem persistence (sentinel):
      ssh ubuntu@${lambdalabs_instance.reranker.ip} 'grep -E \"Persistence sentinel\" /tmp/setup-instance.log | tail -n 5'

    If you upgraded from an older version that managed the filesystem resource:
      terraform state rm lambdalabs_filesystem.model_cache[0]

    ============================================================
  EOT
}
