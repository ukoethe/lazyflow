import time
from lazyflow.zmq_worker import ZMQWorker

import numpy, random, time

for i in range(1):
    w = ZMQWorker()
    w.start()


# loop until eternity
while True:
    time.sleep(100)
