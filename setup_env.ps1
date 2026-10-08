# ============================================================
# BEV Drone Project - 环境配置脚本 (Windows PowerShell)
# Python 3.9 + PyTorch 2.0 + CUDA 11.8
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BEV Drone 环境配置脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ---- 方式一：使用 conda ----
# conda create -n bev_drone python=3.9 -y
# conda activate bev_drone

# ---- 方式二：使用 venv ----
Write-Host "[1/3] 创建 Python 虚拟环境..." -ForegroundColor Yellow
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Write-Host "  虚拟环境已创建并激活" -ForegroundColor Green

Write-Host ""
Write-Host "[2/3] 升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "  pip 升级完成" -ForegroundColor Green

Write-Host ""
Write-Host "[3/3] 安装依赖包..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "  依赖包安装完成" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  验证安装..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
python -c "import torch; import cv2; import transformers; print('环境验证成功'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  环境配置完成!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
