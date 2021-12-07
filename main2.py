import os
import psutil
import argparse
import traceback
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
from py4j.protocol import Py4JError
from src.Agents.PPOAIMulti import PPOAI
import sys
sys.path.insert(0,"F:\\RL590Project\\FightZero\\FTG4.50\\python\\LTAI")
from Core.model import LTAI
sys.path.insert(0,"F:\\RL590Project\\FightZero\\FTG4.50\\python")
from WinOrGoHome import WinOrGoHome
import random
from os.path import exists
import numpy as np
import pickle

def run(args, AI, gateway: JavaGateway):
    manager = gateway.entry_point
    manager.registerAI("PPOPython", PPOAI(gateway, gameRounds=args.games, train=args.train, frameSkip=args.skip))
    # manager.registerAI("KickAIPython", KickAI(gateway))

    # Good Candidates:
    # 1. MctsAi
    # 2. BlackMambda
    # 3. FalzAI
    # 4. JayBot_GM
    # 5. LGIST_Bot
    # 6. UtalFighter
    
    game = manager.createGame("ZEN", "ZEN", "PPOPython", AI, args.games)
    manager.runGame(game)

def connect(args):
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=args.port), callback_server_parameters=CallbackServerParameters(port=0))
    python_port = gateway.get_callback_server().get_listening_port()
    gateway.java_gateway_server.resetCallbackClient(gateway.java_gateway_server.getCallbackClient().getAddress(), python_port)
    return gateway

def disconnect(gateway: JavaGateway):
    gateway.close(close_callback_server_connections=True)
    # gateway.shutdown()

def start_game(AI,gateway):
    manager = gateway.entry_point
    manager.registerAI("PPOPython", PPOAI(gateway, gameRounds=args.games, train=args.train, frameSkip=args.skip))
    if AI=='WinOrGoHome':
        p2=WinOrGoHome(gateway)
    else:
        p2 = LTAI(gateway)
    manager.registerAI(p2.__class__.__name__, p2)
    print("="*20)
    print("Starting Game")
    print(f"AI:{AI}")
    print("="*20)

    game = manager.createGame("ZEN", "ZEN",
                                "PPOPython",
                                p2.__class__.__name__,
                                GAME_NUM)
    manager.runGame(game)

    print("="*20)
    print("Game Stopped")
    print("="*20)
    sys.stdout.flush()

def main_process(AI):
    gateway = connect(args)
    start_game(AI, gateway)
    disconnect(gateway)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--games", type=int, help="Number of games to play for each iteration", default=1)
    parser.add_argument("-i", "--iters", type=int, help="Number of iterations", default=100)
    parser.add_argument("-p", "--port", type=int, help="Game server port", default=4242)
    parser.add_argument("--train", help="Run in training mode (default is simulation)", action="store_true", default=False)
    parser.add_argument("--er", help="exploration rate", type=float, default=0.3)
    parser.add_argument("--skip", help="Whether to skip frames (default is false)", action="store_true", default=False)

    args = parser.parse_args()
    if os.path.exists('AIs.pkl'):
        with open('AIs.pkl','rb') as f:
            AIs = pickle.load(f)
        f.close()
    else:
        AIs=dict()
        AIs['AI']=['BlackMamba',
        'ERHEA_PPO_PG',
        'MctsAi',
        'WinOrGoHome',
        'LTAI']
        AIs['platform_list']=['Java']*3+['Python']*2
        AIs['num_win']=np.array([0]*len(AIs['AI']))
        AIs['num_round']=np.array([0]*len(AIs['AI']))
    
    for iter in range(args.iters):
        print(f'training percentage:{iter/args.iters*100}%')
        if exists("/win_lose.txt") and random.uniform(0, 1)>=args.er:
            with open("/win_lose.txt",'r') as file1:
                lines = file1.readlines()
                for line in lines:
                    win+=if_win*int(line)
                    lose+=(1-if_win)*int(line)
                    if_win=1-if_win
                AIs['num_win'][idx]+=win
                AIs['num_round'][idx]+=win+lose
            file1.close()
            win_rate=AIs['num_win']/AIs['num_round']
            idx=np.where(win_rate==np.min(win_rate))[0]
        else:
            idx=random.randrange(len(AIs['AI']))
        AI=AIs['AI'][idx]
        if AIs['platform_list'][idx]=='Java':
            gateway = connect(args)

            pid = os.getpid()
            prog = psutil.Process(pid)

            print("="*20)
            print("Starting Game")
            print(f"AI:{AI}")
            print("="*20)
            try:
                run(args, AI, gateway)
            except Py4JError:
                print(traceback.format_exc())
            except Exception:
                print(traceback.format_exc())
            finally:
                disconnect(gateway)
                print("="*20)
                print("Game Stopped")
                print("="*20)
                # prog.terminate()
        else:
            GAME_NUM = args.games
            gateway = connect(args)
            manager = gateway.entry_point
            main_process(AI)
        win=0
        lose=0
        if_win=1
        with open("win_lose.txt",'r') as file1:
            lines = file1.readlines()
            for line in lines:
                win+=if_win*int(line)
                lose+=(1-if_win)*int(line)
                if_win=1-if_win
            AIs['num_win'][idx]+=win
            AIs['num_round'][idx]+=win+lose
        file1.close()
        with open('AIs.pkl','wb') as f:
            pickle.dump(AIs, f)
        f.close()