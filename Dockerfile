# 建议使用轻量版镜像
FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 1. 基础环境
ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

# 2. 安装依赖
RUN python -m pip install \
    --user \
    --no-cache-dir \
    numpy \
    scipy \
    simpleitk

# 3. 复制 Python 代码 (注意这里改成 v3)
COPY --chown=user:user trackrad_unet_v3.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/

# 4. 直接复制模型 (7MB 不需要分卷拼接)
# 请确保 best_model.pth 已经在 resources 文件夹里
COPY --chown=user:user resources/best_model.pth /opt/app/resources/best_model.pth

# 5. 检查文件大小
# 【重要修改】只要大于 5MB 就算成功 (适配你的 7MB 模型)
RUN python -c "import os, sys; \
    size = os.path.getsize('/opt/app/resources/best_model.pth') / (1024*1024); \
    print(f'Final model size: {size:.2f} MB'); \
    sys.exit(1) if size < 5 else None"

ENTRYPOINT ["python", "inference.py"]