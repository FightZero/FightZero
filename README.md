# FightZero
Learn to play fighting games from zero knowledge

------

### Set up and Run

1. Install Java SDK  
2. Download [FightingIce](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-2.html) version 4.5 and extract to `FTG4.50`  
3. Enter `FTG4.50` and run command (for Windows):
   ```cmd
   java -cp "FightingICE.jar;lib/*;lib/lwjgl/*;lib/natives/windows/*" Main --py4j --limithp 400 400
   ```
4. Install python required packages:
   ```cmd
   pip install -r requirements.txt
   ```
5. Start running:
   ```cmd
   python main.py
   ```

------

For more information about the game, please visit [FightingIce](https://www.ice.ci.ritsumei.ac.jp/~ftgaic/index-2.html) and its GitHub [repo](https://github.com/TeamFightingICE/FightingICE)