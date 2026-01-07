FROM --platform=linux/amd64 pytorch/pytorch

# 1. 基础环境
ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

# 2. 安装依赖 (保留 Scipy 等)
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

# 4. 【核心步骤】复制分卷并拼接
# 先把 resources 文件夹里所有的 .part 文件复制进去
COPY --chown=user:user resources/ /opt/app/resources/

# 使用 cat 命令将碎片还原为 best_model.pth
# 这里的 * 通配符会自动按顺序 cat part0, part1...
RUN echo "🧩 Reassembling model weights..." && \
    cat /opt/app/resources/best_model.pth.part* > /opt/app/resources/best_model.pth && \
    echo "✅ Model reassembled successfully!"

# (可选) 拼完后删除碎片以减小镜像体积
RUN rm /opt/app/resources/best_model.pth.part*

# 5. 检查文件大小 (保险丝)
RUN python -c "import os, sys; \
    size = os.path.getsize('/opt/app/resources/best_model.pth') / (1024*1024); \
    print(f'Final model size: {size:.2f} MB'); \
    if size < 100: sys.exit(1);"

ENTRYPOINT ["python", "inference.py"]