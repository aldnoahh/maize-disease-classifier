#!/bin/bash

sudo apt update -y
sudo apt upgrade -y
sudo apt install -y python3-pip
pip3 install virtualenv
python3 -m venv inf
source inf/bin/activate
pip3 install -r requirements.txt



