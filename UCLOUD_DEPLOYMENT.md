# UCloud GPU 服务器 SVC API 部署指南

## 概述

本指南提供完整的 UCloud GPU 服务器部署方案，用于提供 SoulX-Singer SVC API 服务。

---

## 1. UCloud 服务器配置

### 推荐配置

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| **GPU** | NVIDIA A10 24GB | 性价比最高，支持 5-6 并发 |
| **CPU** | 8 核+ | 推荐 16 核 |
| **内存** | 32GB+ | 推荐 64GB |
| **系统盘** | 100GB SSD | |
| **数据盘** | 500GB SSD | 存储模型和音频 |
| **带宽** | 10Mbps+ | 推荐 20Mbps |
| **操作系统** | Ubuntu 22.04 LTS | |

### 成本估算

| 配置 | 月成本 (按需) | 月成本 (预留) |
|------|--------------|--------------|
| A10 + 8核 + 32GB | ¥15,000-25,000 | ¥10,000-18,000 (6-7 折) |

---

## 2. 服务器购买步骤

### 2.1 购买 GPU 云主机

1. 登录 UCloud 控制台
2. 选择「产品」→「GPU 型云主机」
3. 选择地域（建议选择离用户最近的）
4. 配置如下：
   - **GPU 类型**: NVIDIA A10
   - **GPU 数量**: 1
   - **CPU**: 8 核
   - **内存**: 32GB
   - **系统盘**: 100GB SSD
   - **数据盘**: 500GB SSD
   - **操作系统**: Ubuntu 22.04 LTS
5. 点击「立即购买」

### 2.2 配置安全组

在 UCloud 控制台，配置安全组规则：

| 协议 | 端口 | 来源 | 说明 |
|------|------|------|------|
| TCP | 22 | 0.0.0.0/0 | SSH |
| TCP | 80 | 0.0.0.0/0 | HTTP |
| TCP | 443 | 0.0.0.0/0 | HTTPS |
| TCP | 8088 | 0.0.0.0/0 | API (可选，直接暴露) |

---

## 3. 服务器环境配置

### 3.1 连接服务器

```bash
# 使用 SSH 连接到服务器
ssh root@your-server-ip

# 或使用 UCloud 提供的 VNC 连接
```

### 3.2 安装 Docker 和 NVIDIA Container Toolkit

```bash
# 更新系统
apt-get update && apt-get upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 启动 Docker
systemctl start docker
systemctl enable docker

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $VERSION_CODENAME)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sed 's#deb https://#deb https://#g' | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime install --runtime=docker --version=5.1.0
systemctl restart docker

# 验证安装
docker info | grep nvidia
```

### 3.3 安装 Docker Compose

```bash
# 下载 Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# 验证安装
docker-compose --version
```

---

## 4. 部署应用

### 4.1 克隆代码

```bash
# 安装 Git
apt-get install -y git

# 克隆代码
cd /opt
git clone https://github.com/Soul-AILab/SoulX-Singer.git
cd SoulX-Singer
```

### 4.2 下载模型

```bash
# 安装 Hugging Face Hub
pip3 install -U huggingface_hub

# 下载主模型
mkdir -p models
cd models
hf download Soul-AILab/SoulX-Singer --local-dir SoulX-Singer

# 下载预处理模型
hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir SoulX-Singer-Preprocess
```

### 4.3 部署服务

```bash
# 返回项目根目录
cd /opt/SoulX-Singer

# 运行部署脚本
chmod +x deploy.sh
bash deploy.sh
```

---

## 5. 验证部署

### 5.1 检查服务状态

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f svc-api
```

### 5.2 测试 API

```bash
# 运行测试脚本
chmod +x test_api.sh
bash test_api.sh
```

### 5.3 访问服务

- **API 文档**: http://your-server-ip:8088/docs
- **健康检查**: http://your-server-ip:8088/health
- **API 端点**: http://your-server-ip:8088/v1/svc

---

## 6. 性能优化

### 6.1 环境变量优化

编辑 `docker-compose.yml`，添加以下环境变量：

```yaml
environment:
  - SVC_CPU_THREADS=2
  - SVC_FP16=1
  - SVC_LIMIT_CPU=0  # 取消 CPU 限制
```

### 6.2 GPU 监控

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看详细 GPU 信息
nvidia-smi --query-gpu=index,name,temperature.gpu_utilization,memory.used,memory.total --format=csv
```

### 6.3 日志监控

```bash
# 查看实时日志
docker-compose logs -f svc-api

# 查看最近 100 行日志
docker-compose logs --tail=100 svc-api
```

---

## 7. 常用命令

### 7.1 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看状态
docker-compose ps
```

### 7.2 日志管理

```bash
# 实时日志
docker-compose logs -f svc-api

# 最近日志
docker-compose logs --tail=100 svc-api

# 导出日志
docker-compose logs svc-api > svc.log
```

### 7.3 更新部署

```bash
# 拉取最新代码
cd /opt/SoulX-Singer
git pull

# 重新构建并启动
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 8. 故障排查

### 8.1 服务无法启动

```bash
# 检查容器状态
docker-compose ps

# 查看错误日志
docker-compose logs svc-api
```

### 8.2 GPU 不可用

```bash
# 检查 GPU
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 8.3 内存不足

```bash
# 查看内存使用
free -h

# 清理 Docker 缓存
docker system prune -a
```

---

## 9. 安全建议

### 9.1 配置防火墙

```bash
# 只开放必要的端口
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

### 9.2 配置 SSL/TLS

```bash
# 使用 Let's Encrypt 免费 SSL
apt-get install certbot python3-certbot-nginx
certbot --nginx -d your-domain.com
```

### 9.3 API 限流

编辑 `nginx/nginx.conf`，添加限流配置：

```nginx
# 在 http 块中添加
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# 在 location 块中使用
limit_req zone=api_limit burst=20 nodelay;
```

---

## 10. 监控告警

### 10.1 基础监控

```bash
# GPU 监控脚本
watch -n 5 nvidia-smi

# 服务监控脚本
watch -n 5 'curl -s http://localhost:8088/health'
```

### 10.2 日志轮转

```bash
# 配置日志轮转
cat > /etc/logrotate.d/soulx-svc << EOF
/var/log/nginx/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        docker-compose exec nginx nginx -s reload
    endscript
}
EOF
```

---

## 11. 成本优化

### 11.1 使用竞价实例

在非高峰时段使用竞价实例（价格低至 1-3 折）

### 11.2 按需启停

```bash
# 定时任务：自动停止（晚上 23 点）
0 23 * * * docker-compose down

# 定时任务：自动启动（早上 8 点）
0 8 * * * cd /opt/SoulX-Singer && docker-compose up -d
```

### 11.3 预留实例

购买预留实例可享受 6-7 折优惠。

---

## 12. 联系支持

如有问题，请检查：
1. 服务器日志: `docker-compose logs`
2. GPU 状态: `nvidia-smi`
3. API 文档: http://your-server-ip:8088/docs
