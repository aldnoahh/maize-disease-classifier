# Use an official Ubuntu image as the base
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Calcutta

# Update package list and install required packages
RUN <<EOF
# apt-get update
# apt-get install -y tzdata
# ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
# echo $TZ > /etc/timezone

apt-get update
apt-get install -y apt-utils

apt-get install -y  cmake pciutils cpio libnotify-bin wget pkg-config gcc default-libmysqlclient-dev software-properties-common python3-pip tmux curl htop 
apt-get clean
rm -rf /var/lib/apt/lists/
EOF

# Add deadsnakes repository and install Python 3.7
# RUN <<EOF
# apt update 
# add-apt-repository ppa:deadsnakes/ppa 
# apt-get update 
# apt-get install -y python3.7 python3.7-dev git python3.7-venv mplayer mariadb-server
# EOF

RUN apt-get update -y && apt-get install -y nvidia-docker2


# Install gdown and other Python packages
RUN pip3 install gdown --no-cache-dir && \
    pip3 install -U protobuf --no-cache-dir

# Copy your requirements.txt and install the Python packages
COPY requirements.txt /home/face_recognition/requirements.txt
WORKDIR /root/maize-issue-classifier/inference
RUN pip3 install -r requirements.txt --no-cache-dir
RUN pip3 install fastapi==0.101.1 uvicorn==0.23.2 onnx onnxruntime-gpu
COPY . /root/maize-issue-classifier/inference
# Start a bash shell when the container is run
CMD ["python3", "api.py"]
# CMD ["tail", "-f", "/dev/null"]