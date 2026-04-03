# UCloud GPU 服务器快速开始

## 5 分钟快速部署

### 前提条件

- 已购买 UCloud GPU 云主机 (A10 推荐)
- 服务器操作系统: Ubuntu 22.04 LTS
- 已配置安全组开放 80、443 端口

### 快速部署命令

```bash
# 1. 连接到服务器
ssh root@your-server-ip

# 2. 安装 Docker 和 NVIDIA Container Toolkit
curl -fsSL https://get.docker.com | sh
systemctl start docker && systemctl enable docker

distribution=$(. /etc/os-release;echo $VERSION_CODENAME)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sed 's#deb https://#deb https://#g' | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime install --runtime=docker --version=5.1.0
systemctl restart docker

# 3. 安装 Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 4. 克隆代码并部署
cd /opt
git clone https://github.com/Soul-AILab/SoulX-Singer.git
cd SoulX-Singer

# 5. 下载模型（重要！）
pip3 install -U huggingface_hub
mkdir -p models
cd models
hf download Soul-AILab/SoulX-Singer --local-dir SoulX-Singer
hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir SoulX-Singer-Preprocess
cd ..

# 6. 部署服务
bash deploy.sh
```

### 验证部署

```bash
# 检查服务状态
docker-compose ps

# 查看 GPU 状态
nvidia-smi

# 测试 API
bash test_api.sh
```

### 访问服务

- **API 文档**: http://your-server-ip:8088/docs
- **健康检查**: http://your-server-ip:8088/health
- **API 地址**: http://your-server-ip:8088/v1/svc

---

## API 使用示例

### cURL 示例

```bash
curl -X POST "http://your-server-ip:8088/v1/svc" \
  -F "prompt_audio=@prompt.wav" \
  -F "target_audio=@target.wav" \
  -F "auto_shift=true" \
  -F "n_steps=32" \
  -F "cfg=1.0" \
  -o output.wav
```

### Python 示例

```python
import requests

url = "http://your-server-ip:8088/v1/svc"

files = {
    'prompt_audio': open('prompt.wav', 'rb'),
    'target_audio': open('target.wav', 'rb'),
    'auto_shift': True,
    'n_steps': 32,
    'cfg': 1.0,
}

response = requests.post(url, files=files)
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

---

## 环境变量配置

### CPU 限制模式（节省资源）

```bash
# 在 docker-compose.yml 中添加
environment:
  - SVC_CPU_THREADS=2  # 推荐 2 核
```

### FP16 模式（加速推理）

```bash
environment:
  - SVC_FP16=1
```

---

## 故障排查

### 问题 1: GPU 不可用

```bash
nvidia-smi  # 检查 GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 问题 2: 服务启动失败

```bash
docker-compose logs svc-api  # 查看日志
docker-compose ps            # 检查状态
```

### 问题 3: 内存不足

```bash
free -h                      # 检查内存
docker system prune -a      # 清理 Docker 缓存
```

---

## 下一步

- [ ] 配置 SSL 证书 (HTTPS)
- [ ] 设置域名
- [ ] 配置监控告警
- [ ] 设置自动备份

详见完整部署指南: [UCLOUD_DEPLOYMENT.md](./UCLOUD_DEPLOYMENT.md)
