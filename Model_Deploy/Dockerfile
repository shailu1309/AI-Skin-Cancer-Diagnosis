FROM python:3.8-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/* \
    python3-pip
WORKDIR myapp
COPY . ./
RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["python","app.py"]
EXPOSE 5000