# Terraform version and provider requirements
# Lambda Cloud deployment for Qwen3-Reranker

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    lambdalabs = {
      source  = "elct9620/lambdalabs"
      version = "~> 0.8"
    }
  }
}

# Configure the Lambda Labs provider
provider "lambdalabs" {
  # API key can be set via:
  # 1. This attribute (not recommended for security)
  # 2. LAMBDALABS_API_KEY environment variable (recommended)
  # api_key = var.lambda_api_key
}
