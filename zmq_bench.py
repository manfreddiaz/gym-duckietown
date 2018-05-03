#!/usr/bin/env python3

import signal
import os
import sys
import time
import zmq
import numpy as np

import subprocess

# For Python 3 compatibility
import sys
if sys.version_info > (3,):
    buffer = memoryview

def recvArray(socket):
    """Receive a numpy array over zmq"""
    md = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    buf = buffer(msg)

    A = np.frombuffer(buf, dtype=np.uint8)
    A = A.reshape(md['shape'])
    return A


##############################################################################

num_processes = 1

procs = []

context = zmq.Context()

"""
    portNo = 5100 + i
    addr_str = "tcp://localhost:%s" % portNo

    print("starting subprocess")
    subprocess.call(["/home/maxime-mila/Desktop/gym-duckietown/zmq_server.py", str(portNo)], close_fds=True)
    pid = os.spawnl(os.P_NOWAIT, './zmq_server.py %d' % portNo)
    print('pid = %d' % pid)

    procs.append((pid, portNo, socket))

"""

portNo = 5100
addr_str = "tcp://localhost:%s" % portNo

print("connecting to %s ..." % addr_str)
socket = context.socket(zmq.PAIR)
ret = socket.connect(addr_str)
assert ret == None, ret


for i in range(0, 8000):

    #socket.send_string("ping")
    #recv = socket.recv()



    #socket.send_json({
    #    "command":"ping",
    #    "array": ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    #})
    #recv = socket.recv_json()



    #socket.send_string("ping")

    socket.send_json({
        "command":"action",
        "values": [ 0.2, 0.3 ]
    })

    recvArray(socket)
