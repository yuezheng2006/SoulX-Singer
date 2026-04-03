# SoulX-Singer SVC API 部署指南

## 环境要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA 8GB 显存 | NVIDIA 12GB+ 显存 |
| CPU | 4 核 | 8+ 核 |
| 内存 | 16GB | 32GB+ |
| 存储 | 20GB | 50GB+ SSD |
| CUDA | 12.1+ | 12.1+ |
| Python | 3.10 | 3.10 |

## 快速部署

### 方式一：Docker 部署（推荐）

```bash
# 1. 构建镜像
docker build -f soulx_svc/Dockerfile -t soulx-svc:latest .

# 2. 下载模型到本地（避免每次启动重新下载）
mkdir -p pretrained_models
huggingface-cli download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer
huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess

# 3. 启动服务
docker run -d \
  --name soulx-svc \
  --gpus all \
  -p 8088:8088 \
  -v $(pwd)/pretrained_models:/app/pretrained_models:ro \
  -e SVC_FP16=1 \
  -e SVC_CPU_THREADS=2 \
  soulx-svc:latest

# 4. 验证服务
curl http://localhost:8088/health
```

### 方式二：原生部署

```bash
# 1. 创建环境
conda create -n soulx-svc python=3.10 -y
conda activate soulx-svc

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 3. 安装依赖
pip install -r requirements.txt
pip install -r soulx_svc/requirements-api.txt

# 4. 下载模型
pip install -U huggingface_hub
huggingface-cli download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer
huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess

# 5. 启动服务
export PYTHONPATH=$PWD:$PYTHONPATH
python -m soulx_svc.api --host 0.0.0.0 --port 8088
```

## 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SVC_FP16` | `0` | 启用 FP16 推理（GPU 推荐开启） |
| `SVC_CPU_THREADS` | - | CPU 线程数限制 |
| `SVC_LIMIT_CPU` | `0` | 限制为单线程（最省 CPU，最慢） |
| `SVC_CORS_ORIGINS` | - | CORS 允许的源，逗号分隔，`*` 允许所有 |
| `SVC_KEEP_SESSIONS` | `0` | 保留会话目录（调试用） |
| `CUDA_VISIBLE_DEVICES` | `0` | 指定 GPU 设备 |

### 常用配置示例

```bash
# GPU 部署（推荐）
export SVC_FP16=1
export SVC_CPU_THREADS=2
export CUDA_VISIBLE_DEVICES=0

# CPU 部署（开发测试）
export SVC_CPU_THREADS=4
export DEVICE=cpu

# 开放 CORS（前端调用）
export SVC_CORS_ORIGINS="https://your-frontend.com"
# 或允许所有
export SVC_CORS_ORIGINS="*"
```

## API 使用

### 健康检查

```bash
curl http://localhost:8088/health
# {"status": "ok", "service": "svc"}
```

### SVC 转换

```bash
curl -X POST http://localhost:8088/v1/svc \
  -F "prompt_audio=@prompt.wav" \
  -F "target_audio=@target.wav" \
  -o generated.wav
```

### 完整参数

```bash
curl -X POST http://localhost:8088/v1/svc \
  -F "prompt_audio=@prompt.wav" \
  -F "target_audio=@target.wav" \
  -F "prompt_vocal_sep=false" \
  -F "target_vocal_sep=true" \
  -F "auto_shift=true" \
  -F "auto_mix_acc=true" \
  -F "pitch_shift=0" \
  -F "n_steps=32" \
  -F "cfg=1.0" \
  -F "seed=42" \
  -o generated.wav
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt_audio` | file | 必填 | 参考音色音频（歌手音色） |
| `target_audio` | file | 必填 | 目标演唱音频（要转换的内容） |
| `prompt_vocal_sep` | bool | `false` | 对参考音频做人声分离 |
| `target_vocal_sep` | bool | `true` | 对目标音频做人声分离 |
| `auto_shift` | bool | `true` | 自动音高偏移对齐 |
| `auto_mix_acc` | bool | `true` | 自动混合伴奏 |
| `pitch_shift` | int | `0` | 音高偏移（半音） |
| `n_steps` | int | `32` | Flow matching 步数 |
| `cfg` | float | `1.0` | Classifier-free guidance scale |
| `seed` | int | `42` | 随机种子 |

## 生产部署建议

### 1. 使用 Gunicorn + Uvicorn

```bash
pip install gunicorn

gunicorn soulx_svc.api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8088 \
  --timeout 300 \
  --keep-alive 5
```

> **注意**：由于 GPU 显存限制，`--workers` 建议设为 1。如需并发，考虑多实例 + 负载均衡。

### 2. Nginx 反向代理

```nginx
upstream soulx_svc {
    server 127.0.0.1:8088;
}

server {
    listen 80;
    server_name api.example.com;

    client_max_body_size 100M;
    proxy_read_timeout 300s;

    location / {
        proxy_pass http://soulx_svc;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Systemd 服务

```ini
# /etc/systemd/system/soulx-svc.service
[Unit]
Description=SoulX-Singer SVC API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/SoulX-Singer
Environment="PYTHONPATH=/opt/SoulX-Singer"
Environment="SVC_FP16=1"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/conda/envs/soulx-svc/bin/python -m soulx_svc.api --host 127.0.0.1 --port 8088
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable soulx-svc
sudo systemctl start soulx-svc
```

### 4. Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: soulx-svc
spec:
  replicas: 1
  selector:
    matchLabels:
      app: soulx-svc
  template:
    metadata:
      labels:
        app: soulx-svc
    spec:
      containers:
      - name: soulx-svc
        image: soulx-svc:latest
        ports:
        - containerPort: 8088
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
        env:
        - name: SVC_FP16
          value: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8088
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8088
          initialDelaySeconds: 120
          periodSeconds: 10
        volumeMounts:
        - name: models
          mountPath: /app/pretrained_models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: soulx-models-pvc
```

## 性能调优

### GPU 优化

```bash
# 启用 FP16（显存减半，速度提升）
export SVC_FP16=1

# 限制 GPU 显存占用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### CPU 优化（无 GPU 环境）

```bash
# 限制 CPU 线程（避免占满）
export SVC_CPU_THREADS=4

# 或极限省 CPU（最慢）
export SVC_LIMIT_CPU=1
```

## 故障排查

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `CUDA out of memory` | 显存不足 | 启用 `SVC_FP16=1`，或减少 `n_steps` |
| `pretrained_models not found` | 模型未下载 | 执行 `huggingface-cli download` |
| `preprocess failed` | 音频格式不支持 | 转换为标准 WAV/MP3 |
| `503 SVC 推理依赖未安装` | 依赖缺失 | 安装 `requirements.txt` |

### 日志调试

```bash
# 保留会话目录（查看中间结果）
export SVC_KEEP_SESSIONS=1

# 查看会话输出
ls outputs/soulx_svc/sessions/
```
