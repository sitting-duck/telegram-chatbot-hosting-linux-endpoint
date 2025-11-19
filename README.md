# telegram-chatbot-hosting-linux-endpoint

Always on Linux the first thing we do is: 
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv # must use 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "numpy<2"
sudo apt install -y python3-venv python3-pip git curl jq ffmpeg openssl
pip install -r requirements.txt
```

### Create your .env file
```bash
cp .env.tmp .env
```

### Build BM25 Index
```bash
python build_bm25.py --corpus ./corpus_clean.jsonl --out ./bm25.idx
```

### Register webhook
```bash
source scripts/load_env.sh # mandatory
scripts/register_webhook.sh

```

Install NGrok (source: https://dashboard.ngrok.com/get-started/setup/linux)
```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok
```

run the following cmd to add your auth token to the default ngrok.yml config file
```bash
ngrok config add-authtoken abc123 #paste your own token from ngrok.com setup page

Authtoken saved to configuration file: /home/nerp/.config/ngrok/ngrok.yml

```

run the following and go to the dev domain in the output and see your app. 

```bash
ngrok http 80
```

Example Output: 
```bash
ngrok                                                                                     (Ctrl+C to quit)
                                                                                                          
🐋 Create instant endpoints for local containers within Docker Desktop → https://ngrok.com/r/docker       
                                                                                                          
Session Status                online                                                                      
Account                       developer (Plan: Free)                                                      
Update                        update available (version 3.32.0, Ctrl-U to update)                         
Version                       3.30.0                                                                      
Region                        United States (us)                                                          
Latency                       38ms                                                                        
Web Interface                 http://127.0.0.1:4040                                                       
Forwarding                    https://river-uncomplying-unidealistically.ngrok-free.dev -> http://localhos
                                                                                                          
Connections                   ttl     opn     rt1     rt5     p50     p90                                 
                              0       0       0.00    0.00    0.00    0.00                                
                                                                                                          
                                                                            
```

### Check GPU Support for Ollama
Driver present?

```bash
$ nvidia-smi
Fri Nov 14 13:58:21 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        Off |   00000000:01:00.0  On |                  N/A |
|  0%   39C    P8             34W /  370W |     444MiB /  10240MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2035      G   /usr/lib/xorg/Xorg                      153MiB |
|    0   N/A  N/A            2196      G   /usr/bin/gnome-shell                     30MiB |
|    0   N/A  N/A           13572      G   .../7177/usr/lib/firefox/firefox        156MiB |
|    0   N/A  N/A          392241      G   .../teamviewer/tv_bin/TeamViewer         13MiB |
+-----------------------------------------------------------------------------------------+

```

### Ollama setup
```bash
scripts/run_ollama.sh
# check if ollama is running by navigating to 
# http://localhost:11434/
# in your browser

```
