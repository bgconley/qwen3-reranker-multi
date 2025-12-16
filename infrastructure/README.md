# Qwen3-Reranker Infrastructure

Deploy Qwen3-Reranker to Lambda Cloud with Tailscale networking for secure, direct access from your local machine.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────────┐
│   Your Mac (dev)    │         │   Lambda Cloud GPU       │
│   ┌─────────────┐   │         │   ┌──────────────────┐   │
│   │ wekadocs-   │   │ Tailnet │   │ qwen3-reranker   │   │
│   │ matrix      │◄──┼─────────┼──►│ :9003            │   │
│   └─────────────┘   │         │   └──────────────────┘   │
│   ┌─────────────┐   │         │   ┌──────────────────┐   │
│   │ Tailscale   │   │         │   │ Tailscale        │   │
│   │ 100.x.x.x   │◄──┼─────────┼──►│ 100.x.x.x        │   │
│   └─────────────┘   │         │   └──────────────────┘   │
└─────────────────────┘         └──────────────────────────┘
```

**Key Benefits:**
- No public port exposure (port 9003 only accessible via Tailscale)
- End-to-end encrypted traffic between your machine and the GPU
- Easy DNS name (`qwen3-reranker`) instead of managing IP addresses
- Auto-cleanup of stale devices with ephemeral auth keys

## Prerequisites

### 1. Terraform

Install Terraform 1.0 or later:

```bash
# macOS
brew install terraform

# Verify installation
terraform version
```

### 2. Lambda Cloud Account

1. Create an account at [Lambda Cloud](https://cloud.lambdalabs.com/)
2. Add payment method
3. Generate an API key:
   - Go to [API Keys](https://cloud.lambdalabs.com/api-keys)
   - Click "Generate API Key"
   - Save the key securely

### 3. SSH Key in Lambda Cloud

1. Go to [SSH Keys](https://cloud.lambdalabs.com/ssh-keys)
2. Either:
   - Upload your existing public key (`~/.ssh/id_ed25519.pub`), or
   - Generate a new key pair
3. Note the **key name** (you'll need it for Terraform)

### 4. Tailscale Account

1. Sign up at [Tailscale](https://login.tailscale.com/start)
2. Install Tailscale on your Mac:
   ```bash
   brew install tailscale
   # or download from https://tailscale.com/download
   ```
3. Connect your Mac to your tailnet:
   ```bash
   tailscale up
   ```

### 5. Tailscale Auth Key

Generate an auth key for the Lambda instance:

1. Go to [Tailscale Keys](https://login.tailscale.com/admin/settings/keys)
2. Click "Generate auth key"
3. Configure the key:
   - **Description**: `qwen3-reranker-lambda`
   - **Reusable**: No (one-time use)
   - **Ephemeral**: Yes (auto-remove when instance terminates)
   - **Pre-approved**: Yes (skip manual device approval)
   - **Tags**: Optional (e.g., `tag:servers` for ACL policies)
4. Copy the key (starts with `tskey-auth-`)

## Quick Start

### 1. Configure Variables

```bash
cd infrastructure/terraform

# Copy the example config
cp terraform.tfvars.example terraform.tfvars

# Edit with your credentials
# REQUIRED: ssh_key_name, and either tailscale_auth_key_path (preferred) or tailscale_auth_key
nano terraform.tfvars  # or use your preferred editor
```

If you set `attach_filesystem = true`, create the filesystem in the Lambda Cloud UI first (same `filesystem_name` and `region`).

### 2. Set Lambda API Key

```bash
# Option 1: Environment variable (recommended)
export LAMBDALABS_API_KEY="your-api-key-here"

# Option 2: Add to terraform.tfvars (less secure)
# lambda_api_key = "your-api-key-here"
```

### 3. Deploy

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy (takes ~5-10 minutes)
terraform apply
```

### 4. Verify Deployment

After deployment, Terraform outputs the connection details:

```bash
# Test the health endpoint
curl http://qwen3-reranker:9003/health

# Detailed health check
curl http://qwen3-reranker:9003/healthz

# Test reranking
curl -X POST http://qwen3-reranker:9003/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI.",
      "The weather is nice today.",
      "Deep learning uses neural networks."
    ]
  }'
```

If `curl` to the short hostname fails, discover the actual MagicDNS name and/or Tailscale IP (Tailscale may append a suffix if the hostname is already taken):

```bash
ssh ubuntu@$(terraform output -raw public_ip) 'tailscale status --json | jq -r .Self.DNSName; tailscale ip -4'
```

### 4b. Verify Persistent Filesystem (Recommended)

If `attach_filesystem = true` (or the deprecated `create_filesystem = true`), the setup script writes a sentinel file into the
HuggingFace cache directory on the persistent filesystem.

1. On the first deploy, confirm the sentinel was written:
   - SSH in and check `/tmp/setup-instance.log` for `Persistence sentinel written`
2. Destroy only the instance (keep the filesystem):
   ```bash
   terraform destroy -target=lambdalabs_instance.reranker
   ```
3. Recreate the instance:
   ```bash
   terraform apply
   ```
4. Confirm persistence:
   - `/tmp/setup-instance.log` should contain `Persistence sentinel found`

### 5. Configure wekadocs-matrix

```bash
export RERANKER_BASE_URL=http://qwen3-reranker:9003
```

## Instance Types

| Type | GPU | VRAM | Cost | Best For |
|------|-----|------|------|----------|
| `gpu_1x_a10` | 1x A10 | 24GB | ~$0.75/hr | Budget inference |
| `gpu_1x_h100_pcie` | 1x H100 | 80GB | ~$2.49/hr | High performance |
| `gpu_1x_a100_sxm` | 1x A100 | 40GB | ~$1.29/hr | Balanced |

To use a different instance type:

```hcl
# terraform.tfvars
instance_type = "gpu_1x_h100_pcie"
```

## Regions

Available Lambda Cloud regions:

| Region | Location |
|--------|----------|
| `us-west-1` | California, USA |
| `us-east-1` | Virginia, USA |
| `us-south-1` | Texas, USA |
| `europe-central-1` | Germany |
| `asia-northeast-1` | Tokyo, Japan |

## Management Commands

### SSH Access

```bash
# Direct SSH (public IP)
ssh ubuntu@$(terraform output -raw public_ip)

# Via Tailscale SSH (if enabled)
ssh ubuntu@qwen3-reranker
```

### Service Management

```bash
# On the Lambda instance:

# Check status
sudo systemctl status qwen3-reranker

# View logs (follow)
sudo journalctl -u qwen3-reranker -f

# View recent logs
sudo journalctl -u qwen3-reranker -n 100

# Restart service
sudo systemctl restart qwen3-reranker

# Stop service
sudo systemctl stop qwen3-reranker
```

### Update Reranker

```bash
# SSH into instance
ssh ubuntu@qwen3-reranker

# Pull latest code
cd ~/qwen3-reranker-multi
git pull

# Restart service
sudo systemctl restart qwen3-reranker
```

## Teardown

To destroy all resources:

```bash
terraform destroy
```

This will:
1. Terminate the Lambda Cloud instance
2. The Tailscale device will be automatically removed (if using ephemeral key)
3. Leave the filesystem alone (recommended)

To destroy only the instance (keep the filesystem/model cache):

```bash
terraform destroy -target=lambdalabs_instance.reranker
```

### Deleting the filesystem

The Lambda filesystem API can return “still mounted” for a short time after an instance is destroyed.
If you need to delete the filesystem, do it manually in the Lambda console after confirming it is not
attached to any instance.

If you previously managed the filesystem in this Terraform state, stop Terraform from trying to delete it:

```bash
terraform state rm lambdalabs_filesystem.model_cache[0]
```

## Troubleshooting

### Instance Not Available

Lambda Cloud instances are subject to availability. If you get an error about instance availability:

1. Try a different region:
   ```hcl
   region = "us-east-1"
   ```

2. Try a different instance type:
   ```hcl
   instance_type = "gpu_1x_a100_sxm"
   ```

### Tailscale Not Connecting

1. Check Tailscale status on the instance:
   ```bash
   ssh ubuntu@$(terraform output -raw public_ip)
   tailscale status
   ```

2. Check if the device appears in [Tailscale Admin](https://login.tailscale.com/admin/machines)

3. Verify your auth key hasn't expired

### Service Not Starting

1. Check service logs:
   ```bash
   ssh ubuntu@$(terraform output -raw public_ip)
   sudo journalctl -u qwen3-reranker -n 100
   ```

2. Check if the model is downloading (can take several minutes on first start)

3. Verify GPU is available:
   ```bash
   nvidia-smi
   ```

### Model Loading Issues

The Qwen3-Reranker-4B model (~8GB) needs to be downloaded on first start. This can take 5-10 minutes depending on network speed.

Check download progress:
```bash
ssh ubuntu@qwen3-reranker
sudo journalctl -u qwen3-reranker -f
```

## Security Considerations

1. **Tailscale Auth Keys**: Use ephemeral keys that auto-expire
2. **terraform.tfvars**: Never commit to version control (already in .gitignore)
3. **Lambda API Key**: Use environment variables, not files
4. **Network**: Port 9003 is only accessible via Tailscale, not public internet

## Cost Estimation

| Component | Cost |
|-----------|------|
| Lambda GPU (A10) | ~$0.75/hr (~$540/mo if 24/7) |
| Tailscale | Free (Personal) or $6/user/mo (Teams) |
| Data Transfer | Included in Lambda pricing |

**Tip**: Remember to `terraform destroy` when not using the instance!

## Provider Notes (LambdaLabs Terraform)

The `elct9620/lambdalabs` provider can be flaky around filesystem lifecycle (delete/update often fails if the API still
considers it “mounted”).

This repo now treats the filesystem as an external persistent resource and only **attaches it by name** to the instance.
If you upgraded from an older version that managed the filesystem in Terraform state, run:

```bash
terraform -chdir=infrastructure/terraform state rm lambdalabs_filesystem.model_cache[0]
```

## Files Reference

```
infrastructure/
├── terraform/
│   ├── main.tf                    # Instance resource and provisioners
│   ├── variables.tf               # Input variable definitions
│   ├── outputs.tf                 # Output values
│   ├── versions.tf                # Provider configuration
│   └── terraform.tfvars.example   # Example configuration
├── scripts/
│   ├── setup-instance.sh          # Main setup script
│   └── install-tailscale.sh       # Standalone Tailscale installer
└── README.md                      # This file
```
