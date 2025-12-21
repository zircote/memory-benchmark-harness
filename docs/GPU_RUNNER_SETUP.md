# GPU Runner Setup Guide

This document describes how to configure self-hosted GitHub Actions runners with GPU support for the benchmark harness.

## Overview

GPU-accelerated benchmarks require:
1. A machine with NVIDIA GPU(s)
2. NVIDIA drivers and CUDA toolkit
3. Docker with NVIDIA Container Toolkit
4. GitHub Actions runner configured with GPU labels

## Hardware Requirements

### Minimum Specifications
- NVIDIA GPU with CUDA Compute Capability 7.0+ (Volta or newer)
- 16GB GPU memory recommended for large model embeddings
- 32GB system RAM
- 100GB free disk space

### Recommended GPUs
| GPU | VRAM | CUDA Cores | Best For |
|-----|------|------------|----------|
| NVIDIA A100 | 40/80GB | 6912 | Production benchmarks |
| NVIDIA A10 | 24GB | 9216 | Cloud instances |
| NVIDIA RTX 4090 | 24GB | 16384 | Development |
| NVIDIA RTX 3090 | 24GB | 10496 | Budget option |

## Software Setup

### 1. Install NVIDIA Drivers

```bash
# Ubuntu 22.04/24.04
sudo apt update
sudo apt install -y nvidia-driver-550  # Or latest stable version

# Verify installation
nvidia-smi
```

### 2. Install CUDA Toolkit

```bash
# Install CUDA 12.2 (matches Dockerfile.gpu)
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --toolkit --silent

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### 3. Install Docker with NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 4. Configure GitHub Actions Runner

```bash
# Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download runner (check for latest version)
curl -o actions-runner-linux-x64.tar.gz -L \
    https://github.com/actions/runner/releases/download/v2.320.0/actions-runner-linux-x64-2.320.0.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

# Configure with GPU labels
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO \
    --token YOUR_RUNNER_TOKEN \
    --labels self-hosted,gpu,cuda-12.2

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

## Workflow Configuration

Enable GPU tests in `.github/workflows/ci.yml` by uncommenting:

```yaml
gpu-test:
  name: GPU Tests
  runs-on: [self-hosted, gpu]
  needs: [test]
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v4

    - name: Build GPU image
      run: docker build -t benchmark-harness:gpu-latest -f docker/Dockerfile.gpu .

    - name: Run GPU tests
      run: |
        docker run --rm --gpus all \
          benchmark-harness:gpu-latest \
          pytest tests/ -v -m gpu
```

## Cloud GPU Options

### AWS EC2
- **Instance types**: `g4dn.xlarge` (T4), `p3.2xlarge` (V100), `p4d.24xlarge` (A100)
- Use the [Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) for pre-configured CUDA

### Google Cloud
- **Instance types**: `n1-standard-4` + `nvidia-tesla-t4`, `a2-highgpu-1g` (A100)
- Use [Deep Learning VM Image](https://cloud.google.com/deep-learning-vm)

### Azure
- **Instance types**: `Standard_NC6s_v3` (V100), `Standard_ND96asr_v4` (A100)
- Use [Data Science Virtual Machine](https://azure.microsoft.com/en-us/products/virtual-machines/data-science-virtual-machines/)

## Troubleshooting

### Common Issues

**GPU not visible in Docker:**
```bash
# Check NVIDIA Container Runtime is configured
docker info | grep -i nvidia

# If not present, reconfigure
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**CUDA version mismatch:**
```bash
# Check driver's maximum supported CUDA version
nvidia-smi  # Shows "CUDA Version: X.X" in top right

# Ensure Docker image CUDA version <= driver's maximum
```

**Out of memory errors:**
```bash
# Monitor GPU memory during tests
watch -n 1 nvidia-smi

# Reduce batch sizes in benchmark configuration
export BENCHMARK_BATCH_SIZE=16  # Default is 32
```

**Runner not picking up jobs:**
```bash
# Check runner status
cd ~/actions-runner
./svc.sh status

# Check logs
journalctl -u actions.runner.YOUR_ORG-YOUR_REPO.YOUR_RUNNER_NAME.service -f
```

## Security Considerations

1. **Isolate GPU runners**: Use dedicated machines for CI, not shared workstations
2. **Limit repository access**: Configure runners for specific repositories only
3. **Use ephemeral runners**: Consider [ephemeral runners](https://docs.github.com/en/actions/hosting-your-own-runners/autoscaling-with-self-hosted-runners) for better isolation
4. **Regular updates**: Keep NVIDIA drivers and CUDA toolkit updated for security patches

## Performance Optimization

### Docker Layer Caching
```bash
# Pre-pull base images to speed up builds
docker pull nvidia/cuda:12.2.0-devel-ubuntu22.04
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
```

### Build Cache
```bash
# Enable Docker BuildKit
export DOCKER_BUILDKIT=1

# Use cache mounts in Dockerfile (already configured in Dockerfile.gpu)
```

### Persistent Model Cache
```bash
# Mount model cache directory to avoid re-downloading
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    benchmark-harness:gpu-latest pytest tests/ -v -m gpu
```
