# Installation

pip install -r requirements.txt

# Usage

streamlit run app.py

# for ocr_server:
pip install fastapi uvicorn pillow paddleocr
pip install paddlepaddle

User	root
Password	vqUHqXtewjTMfnrvRstW

## ssh into server
ssh -i /mnt/c/Users/marku/.ssh/id_rsa root@168.119.242.186
## Upload a file
scp -i /mnt/c/Users/marku/.ssh/id_rsa "/mnt/h/Meine Ablage/TU/Master/Implementation/ocr_server.py" root@168.119.242.186:/root/ocr_api/
## activate environment
cd ~/ocr_api
source venv/bin/activate
## start fastAPI
uvicorn ocr_server:app --host 0.0.0.0 --port 8500

## XIRSYS
1. Uni Email.
2. testerjuno