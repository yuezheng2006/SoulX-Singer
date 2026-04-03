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

## 常见配置参数

### API 参数说明

| 参数 | 类型 | 默认值 | 推荐值 | 说明 |
|------|------|--------|--------|------|
| `prompt_vocal_sep` | bool | false | false | 是否对参考音频做人声分离（通常不需要，参考音频应已是纯净人声） |
| `target_vocal_sep` | bool | true | true | 是否对目标音频做人声分离（如果目标是带伴奏的歌曲，推荐 true） |
| `auto_shift` | bool | true | true | 是否自动调整音高（让转换后的歌声更贴合参考音色） |
| `auto_mix_acc` | bool | true | true | 是否自动混音伴奏（仅在 target_vocal_sep=true 时有效） |
| `pitch_shift` | int | 0 | 0 | 手动音高偏移（半音，正数升高，负数降低，auto_shift=true 时建议 0） |
| `n_steps` | int | 32 | 16-32 | 采样步数（16 速度快，32 质量好，推荐 16 用于生产） |
| `cfg` | float | 1.0 | 1.0-2.0 | 分类器自由引导强度（值越高音色相似度越高，但可能失真） |
| `seed` | int | 42 | 42 | 随机种子（固定值保证结果可复现） |
| `prompt_max_sec` | int | 30 | 30 | 参考音频最大时长（秒） |
| `target_max_sec` | int | 600 | 300 | 目标音频最大时长（秒，建议 < 5 分钟） |

### 推荐配置组合

#### 快速转换（生产环境）
```bash
n_steps=16
cfg=1.0
auto_shift=true
auto_mix_acc=true
```

#### 高质量转换
```bash
n_steps=32
cfg=1.5
auto_shift=true
auto_mix_acc=true
```

#### 干声转换（无伴奏）
```bash
target_vocal_sep=false
auto_mix_acc=false
auto_shift=true
n_steps=32
```

## API 使用示例

### cURL 示例

```bash
# 使用推荐配置
curl -X POST "http://your-server-ip:8088/v1/svc" \
  -F "prompt_audio=@prompt.wav" \
  -F "target_audio=@target.wav" \
  -F "auto_shift=true" \
  -F "n_steps=16" \
  -F "cfg=1.0" \
  -o output.wav
```

### Python 示例

```python
import requests

url = "http://your-server-ip:8088/v1/svc"

# 使用 URL 方式
data = {
    'prompt_audio_url': 'https://example.com/prompt.wav',
    'target_audio_url': 'https://example.com/target.mp3',
    'auto_shift': True,
    'n_steps': 16,
    'cfg': 1.0,
}

response = requests.post(url, data=data)
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
