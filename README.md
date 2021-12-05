# FightZero
Learn to play fighting games from zero knowledge

------

### Set up and Run

1. Install Java SDK  
2. Download [FightingIce](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-2.html) version 4.5 and extract to `FTG4.50`  
3. Enter `FTG4.50` and run command (for Windows):
   ```cmd
   java -Xms1024m -Xmx1024m -cp "FightingICE.jar;lib/*;lib/lwjgl/*;lib/natives/windows/*;data/ai/*" Main --py4j --limithp 400 400 --inverted-player 1
   ```
4. Install python required packages:
   ```cmd
   pip install -r requirements.txt
   ```
5. Start running:
   ```cmd
   python main.py
   ```
6. For training AI:
   ```cmd
   python main.py --train
   ```
   And launch game server with:
   ```cmd
   java -Xms1024m -Xmx1024m -cp "FightingICE.jar;lib/*;lib/lwjgl/*;lib/natives/windows/*;data/ai/*" Main --py4j --limithp 400 400 --mute --fastmode --inverted-player 1 --disable-window
   ```

Command-line Arguments
```cmd
> python .\main.py --help
usage: main.py [-h] [-n NUMBER] [-p PORT] [--train]

optional arguments:
  -h, --help            show this help message and exit
  -n NUMBER, --number NUMBER
                        Number of rounds to play
  -p PORT, --port PORT  Game server port
  --train               Run in training mode (default is simulation)
```

------

For more information about the game, please visit [FightingIce](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-2.html) and its GitHub [repo](https://github.com/TeamFightingICE/FightingICE)