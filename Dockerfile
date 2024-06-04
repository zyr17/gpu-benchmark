FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir
COPY . /app
WORKDIR /app
ENTRYPOINT ["bash", "/app/test_all.sh"]