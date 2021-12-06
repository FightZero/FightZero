import os
import psutil
import argparse
import traceback
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
from py4j.protocol import Py4JError
from src.Agents.PPOAI import PPOAI
from src.Agents.PPOAI2 import PPOAI2
from src.Agents.PPOAIBM import PPOAIBM
from src.Agents.KickAI import KickAI
from tmp.WinOrGoHome import WinOrGoHome

def run(args, gateway: JavaGateway):
    manager = gateway.entry_point
    manager.registerAI("PPOPython", PPOAI(gateway, gameRounds=args.number,
        train=args.train, frameSkip=args.skip))
    manager.registerAI("PPO2Python", PPOAI2(gateway, gameRounds=args.number,
        train=args.train, frameSkip=args.skip))
    manager.registerAI("PPOBMPython", PPOAIBM(gateway, gameRounds=args.number,
        train=args.train, frameSkip=args.skip))
    # manager.registerAI("KickAIPython", KickAI(gateway))

    # Good Candidates:
    # 1. MctsAi
    # 2. BlackMamba
    # 3. FalzAI
    # 4. JayBot_GM
    # 5. LGIST_Bot
    # 6. UtalFighter


    game = manager.createGame("ZEN", "ZEN", "PPOBMPython", "MctsAi", args.number)
    # game = manager.createGame("ZEN", "ZEN", "KickAI", "MctsAi", args.number)
    manager.runGame(game)

def connect(args):
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=args.port), callback_server_parameters=CallbackServerParameters(port=0))
    python_port = gateway.get_callback_server().get_listening_port()
    gateway.java_gateway_server.resetCallbackClient(gateway.java_gateway_server.getCallbackClient().getAddress(), python_port)
    return gateway

def disconnect(gateway: JavaGateway):
    gateway.close(close_callback_server_connections=True)
    gateway.shutdown()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, help="Number of rounds to play", default=2)
    parser.add_argument("-p", "--port", type=int, help="Game server port", default=4242)
    parser.add_argument("--train", help="Run in training mode (default is simulation)", action="store_true", default=False)
    parser.add_argument("--skip", help="Whether to skip frames (default is false)", action="store_true", default=False)

    args = parser.parse_args()

    gateway = connect(args)

    pid = os.getpid()
    prog = psutil.Process(pid)

    print("="*20)
    print("Starting Game")
    print("="*20)
    try:
        run(args, gateway)
    except Py4JError:
        print(traceback.format_exc())
    except Exception:
        print(traceback.format_exc())
    finally:
        disconnect(gateway)
        print("="*20)
        print("Game Stopped")
        print("="*20)
        prog.terminate()