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

# =============================================================================
# Tailscale Information
# =============================================================================

output "tailscale_hostname" {
  description = "Tailscale hostname (use this to access the reranker)"
  value       = var.tailscale_hostname
}

output "tailscale_url" {
  description = "Tailscale URL for the reranker service"
  value       = "http://${var.tailscale_hostname}:${var.reranker_port}"
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

    ============================================================
  EOT
}
