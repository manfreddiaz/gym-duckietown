#!/usr/bin/env python

import signal
import sys
import time
import zmq
import numpy as np

def signal_handler(signal, frame):
    print ("exiting")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

##############################################################################

portNo = int(sys.argv[1])

serverAddr = "tcp://*:%s" % portNo
print('Starting server at %s' % serverAddr)
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind(serverAddr)



image = np.zeros((160, 120, 3), dtype=np.uint8)



def sendArray(socket, array):
    """Send a numpy array with metadata over zmq"""
    md = dict(
        dtype=str(array.dtype),
        shape=array.shape,
    )
    # SNDMORE flag specifies this is a multi-part message
    socket.send_json(md, flags=zmq.SNDMORE)
    return socket.send(array, flags=0, copy=True, track=False)

def poll_socket(socket, timetick = 10):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 10msec
    try:
        print("poller ready")
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv_json()
                #yield socket.recv_string()
    except KeyboardInterrupt:
        print ("stopping server")
        quit()

def handle_message(msg):
    #socket.send_string(msg)

    #socket.send_json(msg)

    sendArray(socket, image)


for message in poll_socket(socket):
    handle_message(message)
