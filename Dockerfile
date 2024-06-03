FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . /app
WORKDIR /app
ENTRYPOINT ["bash", "/app/test_all.sh"]