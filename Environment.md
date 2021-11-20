# Game Environment Setup

The game we will be playing is _Street Fighter II - Special Champion Edition (USA)_, which already has its [configs](https://github.com/openai/retro/tree/master/retro/data/stable/StreetFighterIISpecialChampionEdition-Genesis) in `gym-retro` and the rom data can be downloaded from [edge emulation](https://edgeemu.net/details-12765.htm).

### Preparation Steps

* Download rom data and extract to a folder (with the `Street Fighter II' - Special Champion Edition (USA).md` file)
* Install [`Gym Retro`](https://github.com/openai/retro)
* Import game data:
  ```bash
    python3 -m retro.import your_extracted_folder
  ```
* Test run:
  ```bash
    python3 -m retro.examples.interactive --game StreetFighterIISpecialChampionEdition-Genesis
  ```
  And you should be able to run and interact with the game.

### Preparation with Scripts

* Alternatively, can use `prepare.sh` to prepare for the environment