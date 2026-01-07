# 建议使用轻量版镜像，下载更快且稳定 (如果你不想换，保持原来的 pytorch/pytorch 也可以)
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

# 3. 复制 Python 代码
COPY --chown=user:user trackrad_unet_v2.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/

# 4. 复制分卷并拼接
COPY --chown=user:user resources/ /opt/app/resources/

RUN echo "🧩 Reassembling model weights..." && \
    cat /opt/app/resources/best_model.pth.part* > /opt/app/resources/best_model.pth && \
    echo "✅ Model reassembled successfully!"

# 删除碎片
RUN rm /opt/app/resources/best_model.pth.part*

# 5. 【修复点】检查文件大小 (修正了 Python 语法)
# 使用 "sys.exit(1) if ... else None" 这种写法是合法的
RUN python -c "import os, sys; \
    size = os.path.getsize('/opt/app/resources/best_model.pth') / (1024*1024); \
    print(f'Final model size: {size:.2f} MB'); \
    sys.exit(1) if size < 100 else None"

ENTRYPOINT ["python", "inference.py"]