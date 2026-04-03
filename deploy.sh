#!/bin/bash

# UCloud GPU 服务器 SVC API 部署脚本
# 使用方法: bash deploy.sh

set -e

echo "=========================================="
echo "SoulX-Singer SVC API 部署脚本"
echo "=========================================="

# 配置变量
MODEL_DIR="./models"
OUTPUT_DIR="./outputs"
AUDIO_CACHE_DIR="./audios"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 Docker
echo -e "\n${YELLOW}[1/7] 检查 Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker 未安装，请先安装 Docker${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker 已安装${NC}"

# 检查 Docker Compose
echo -e "\n${YELLOW}[2/7] 检查 Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose 未安装，正在安装...${NC}"
    sudo apt-get update
    sudo apt-get install -y docker-compose
fi
echo -e "${GREEN}✓ Docker Compose 已安装${NC}"

# 检查 NVIDIA Docker Runtime
echo -e "\n${YELLOW}[3/7] 检查 NVIDIA Docker Runtime...${NC}"
if ! docker info | grep -q nvidia; then
    echo -e "${YELLOW}配置 NVIDIA Docker Runtime...${NC}"
    sudo nvidia-ctk runtime install --version=5.1.0
    sudo systemctl restart docker
    echo -e "${GREEN}✓ NVIDIA Docker Runtime 已配置${NC}"
else
    echo -e "${GREEN}✓ NVIDIA Docker Runtime 已配置${NC}"
fi

# 创建必要的目录
echo -e "\n${YELLOW}[4/7] 创建目录...${NC}"
mkdir -p $MODEL_DIR $OUTPUT_DIR $AUDIO_CACHE_DIR
echo -e "${GREEN}✓ 目录已创建${NC}"

# 检查模型文件
echo -e "\n${YELLOW}[5/7] 检查模型文件...${NC}"
if [ ! -d "$MODEL_DIR/SoulX-Singer" ] || [ ! -d "$MODEL_DIR/SoulX-Singer-Preprocess" ]; then
    echo -e "${YELLOW}模型文件未找到，请下载模型到 $MODEL_DIR 目录：${NC}"
    echo "1. SoulX-Singer: https://www.modelscope.cn/models/Soul-AILab/SoulX-Singer"
    echo "2. SoulX-Singer-Preprocess: https://www.modelscope.cn/models/Soul-AILab/SoulX-Singer-Preprocess"
    read -p "模型已下载? (y/n): " has_models
    if [ "$has_models" != "y" ]; then
        echo -e "${RED}请先下载模型文件后再运行此脚本${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ 模型文件已就绪${NC}"

# 构建并启动服务
echo -e "\n${YELLOW}[6/7] 构建并启动服务...${NC}"
docker-compose down  # 清理旧容器
docker-compose build
docker-compose up -d

# 等待服务启动
echo -e "\n${YELLOW}[7/7] 等待服务启动...${NC}"
sleep 10

# 检查服务状态
echo -e "\n${YELLOW}检查服务状态...${NC}"
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✓ 服务启动成功！${NC}"
    echo ""
    echo "=========================================="
    echo "部署完成！"
    echo "=========================================="
    echo ""
    echo "API 地址: http://localhost:8088"
    echo "文档地址: http://localhost:8088/docs"
    echo "健康检查: http://localhost:8088/health"
    echo ""
    echo "查看日志: docker-compose logs -f"
    echo "停止服务: docker-compose down"
    echo ""
else
    echo -e "${RED}✗ 服务启动失败${NC}"
    echo "请查看日志: docker-compose logs"
    exit 1
fi
