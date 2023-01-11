import zmq
import time

"""
Server is paired to `audio_zmq.py`
"""

context = zmq.Context()


def simple_server(port: int = 5578, topic: str = "tt_probs"):
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    i = 0
    while True:
        topic = socket.recv_string()
        d = socket.recv()
        print("received: ", type(d), d)
        i += 1


if __name__ == "__main__":

    try:
        simple_server()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        context.destroy()
        print("Context destroyed")
