FROM python:3.8.16-buster

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

RUN chmod +x run_server.sh

ENTRYPOINT ["./run_server.sh"]