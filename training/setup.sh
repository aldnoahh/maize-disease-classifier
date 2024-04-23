sudo apt update -y
sudo apt upgrade -y
sudo apt install -y python3-pip
pip3 install virtualenv
python3 -m venv train
source train/bin/activate