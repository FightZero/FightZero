# prepare game for the environment
sudo apt install p7zip-full wget
pip3 install -r requirements.txt
wget https://edgeemu.net/down.php?id=12765 -nc -O game.7z
7z x game.7z -o./game -y
python3 -m retro.import game
