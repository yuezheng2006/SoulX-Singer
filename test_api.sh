#!/bin/bash

# UCloud GPU 服务器 API 测试脚本
# 使用方法: bash test_api.sh

API_URL="${API_URL:-http://localhost:8088}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "SoulX-Singer SVC API 测试"
echo "=========================================="
echo ""
echo "API 地址: $API_URL"
echo ""

# 测试 1: 健康检查
echo -e "${YELLOW}[测试 1/5] 健康检查...${NC}"
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 健康检查通过${NC}"
    echo "响应: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗ 健康检查失败${NC}"
    exit 1
fi
echo ""

# 测试 2: 访问文档
echo -e "${YELLOW}[测试 2/5] 访问 API 文档...${NC}"
DOCS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs")
if [ "$DOCS_STATUS" == "200" ]; then
    echo -e "${GREEN}✓ 文档可访问${NC}"
else
    echo -e "${RED}✗ 文档不可访问 (状态码: $DOCS_STATUS)${NC}"
fi
echo ""

# 测试 3: 上传测试音频（如果有）
echo -e "${YELLOW}[测试 3/5] 准备测试音频...${NC}"
if [ -f "test_prompt.wav" ] && [ -f "test_target.wav" ]; then
    echo -e "${GREEN}✓ 测试音频已存在${NC}"
else
    echo -e "${YELLOW}测试音频不存在，跳过上传测试${NC}"
    echo "如需测试上传，请准备 test_prompt.wav 和 test_target.wav"
fi
echo ""

# 测试 4: API 端点测试
echo -e "${YELLOW}[测试 4/5] API 端点测试...${NC}"
if [ -f "test_prompt.wav" ] && [ -f "test_target.wav" ]; then
    echo "上传测试音频到 API..."
    RESPONSE=$(curl -s -X POST "$API_URL/v1/svc" \
        -F "prompt_audio=@test_prompt.wav" \
        -F "target_audio=@test_target.wav" \
        -F "auto_shift=true" \
        -F "n_steps=16" \
        -F "cfg=1.0")

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ API 调用成功${NC}"
        echo "响应: $RESPONSE"
    else
        echo -e "${RED}✗ API 调用失败${NC}"
    fi
else
    echo -e "${YELLOW}跳过 API 端点测试（需要测试音频）${NC}"
fi
echo ""

# 测试 5: GPU 状态检查
echo -e "${YELLOW}[测试 5/5] GPU 状态检查...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader
else
    echo -e "${YELLOW}nvidia-smi 不可用，跳过 GPU 检查${NC}"
fi
echo ""

echo "=========================================="
echo "测试完成！"
echo "=========================================="
