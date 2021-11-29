import argparse
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field
from KickAI import KickAI
from DisplayInfo import DisplayInfo

def run(args, gateway):
    p1 = KickAI(gateway)
    p2 = DisplayInfo(gateway)
    manager = gateway.entry_point
    manager.registerAI(p1.__class__.__name__, p1)
    manager.registerAI(p2.__class__.__name__, p2)

    game = manager.createGame("ZEN", "ZEN",
                                  p1.__class__.__name__,
                                  p2.__class__.__name__,
                                  args.number)
    manager.runGame(game)

def connect(args):
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=args.port), callback_server_parameters=CallbackServerParameters())
    return gateway

def disconnect(gateway):
	gateway.close_callback_server()
	gateway.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, help="Number of rounds to play", default=2)
    parser.add_argument("-p", "--port", type=int, help="Game server port", default=4242)

    args = parser.parse_args()

    gateway = connect(args)

    try:
        run(args, gateway)
    except Exception as e:
        print(e)
    finally:
        disconnect(gateway)