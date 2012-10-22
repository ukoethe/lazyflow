import yappi
import threading
import time
import vigra
import lazyflow
from lazyflow.graph import *
from lazyflow import operators
from lazyflow.request import Request, Pool
import functools

doProfile = False
if doProfile:
    yappi.start()

g = Graph()

mcount = 20000
mcountf = 500

lock = threading.Lock()
eventA = threading.Event()
eventB = threading.Event()
lockA = threading.Lock()
lockB = threading.Lock()
lockA.acquire()

class A(threading.Thread):
    def run(self):
        for i in range(1,mcount):
            lockA.acquire()
            lockB.release()

class B(threading.Thread):
    def run(self):
        for i in range(1,mcount):
            lockB.acquire()
            lockA.release()

# import zmq
# context = zmq.Context(1)
#
# class ZmqA(threading.Thread):
#   def run(self):
#     socket = zmq.Socket(context,zmq.PAIR)
#     socket.bind("inproc://pipe")
#     for i in range(1,mcount):
#       socket.send("")#, flags = zmq.NOBLOCK)
#       msg = socket.recv()
#
#
# class ZmqB(threading.Thread):
#   def run(self):
#     socket = zmq.Socket(context,zmq.PAIR)
#     socket.connect("inproc://pipe")
#     for i in range(1,mcount):
#       msg = socket.recv()
#       socket.send("")#, flags = zmq.NOBLOCK)
#

class C(object):
    def __init__(self,shape,dtype=numpy.uint8):
        self.array = numpy.ndarray(shape,dtype)

    def __getitem__(self,key):
        return self.array[key]


cache = operators.OpArrayCache(graph=g)
p = operators.OpArrayPiper(graph=g)


arr = numpy.ndarray((100,100,100,1),numpy.uint8)
arr = arr.view(vigra.VigraArray)
arr.axistags = vigra.defaultAxistags('xyzc')

cache.inputs["Input"].setValue(arr)
p.Input.connect(cache.outputs["Output"])

features = operators.OpPixelFeaturesPresmoothed(graph=g)
matrix = numpy.ndarray((6,2), numpy.uint8)
matrix[:] = 0
matrix[0,:] = 1
features.inputs["Scales"].setValue((1.0,3.0))
features.inputs["Input"].connect(cache.outputs["Output"])
features.inputs["Matrix"].setValue(matrix)

print features.Output.meta.axistags

t1 = time.time()
for i in range(0,mcount):
    res = numpy.ndarray((1,),numpy.uint8)
t2 = time.time()
print "\n\n"
print "PYTHON NUMPY ALLOC OVERHEAD:    %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %fus latency" % ((t2-t1)*1e6/mcount,)


t1 = time.time()
for i in range(0,mcount):
    res = arr[1:2,1:2,1:2]
t2 = time.time()
print "\n\n"
print "PYTHON NUMPY CALL OVERHEAD:     %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %fus latency" % ((t2-t1)*1e6/mcount,)


c = C((100,100,100),numpy.uint8)
t1 = time.time()
for i in range(0,mcount):
    res = c[1:2,1:2,1:2]
t2 = time.time()
print "\n\n"
print "PYTHON DYN DISPATCH OVERHEAD:   %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %fus latency" % ((t2-t1)*1e6/mcount,)

a = A()
b = B()

a.start()
time.sleep(1.0)
t1 = time.time()
b.start()
a.join()
b.join()
t2 = time.time()
print "\n\n"
print "PYTHON THREAD SWITCH OVERHEAD:  %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %fus latency" % ((t2-t1)*1e6/mcount,)


# a = ZmqA()
# b = ZmqB()
#
# a.start()
# time.sleep(0.2)
# t1 = time.time()
# b.start()
# a.join()
# b.join()
# t2 = time.time()
# print "\n\n"
# print "ZMQ    THREAD SWITCH OVERHEAD:  %f seconds for %d iterations" % (t2-t1,mcount)
# print "                                %fus latency" % ((t2-t1)*1e6/mcount,)
#
#
t1 = time.time()
for i in range(0,mcount):
    p.outputs["Output"][3,3,3,0].allocate().wait()
t2 = time.time()
print "\n\n"
print "LAZYFLOW SYNC WAIT OVERHEAD:    %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %fus latency" % ((t2-t1)*1e6/mcount,)


t1 = time.time()
requests = []
for i in range(0,mcount):
    r = p.outputs["Output"][3,3,3,0].allocate()
    requests.append(r)

for r in requests:
    r.wait()
t2 = time.time()
print "\n\n"
print "LAZYFLOW ASYNC WAIT OVERHEAD:   %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %fus latency" % ((t2-t1)*1e6/mcount,)



t1 = time.time()
for i in range(0,mcountf):
    features.outputs["Output"][:50,:50,:50,:].allocate().wait()
t2 = time.time()
print "\n\n"
print "LAZYFLOW SYNC WAIT FEATURES :    %f seconds for %d iterations" % (t2-t1,mcountf)
print "                                %0.3fms latency" % ((t2-t1)*1e3/mcountf,)


t1 = time.time()
requests = []
for i in range(0,mcountf):
    r = features.outputs["Output"][:50,:50,:50,:].allocate()
    requests.append(r)

for r in requests:
    r.submit()

for r in requests:
    r.wait()
t2 = time.time()
print "\n\n"
print "LAZYFLOW ASYNC WAIT FEATURES:   %f seconds for %d iterations" % (t2-t1,mcountf)
print "                                %0.3fms latency" % ((t2-t1)*1e3/mcountf,)

if doProfile:
    yappi.stop()
    yappi.print_stats(sort_type = yappi.SORTTYPE_TTOT)




def empty_func(b):
    a = 7 + b
    a = "lksejhkl JSFLAJSSDFJH   AKDHAJKSDH ADKJADHK AJHSKA AKJ KAJSDH AKDAJHSKAJHD KASHDAKDJH".split(" ")

t1 = time.time()

def lots_of_work():
    requests = []
    for i in range(mcount):
        req = Request(functools.partial(empty_func, b = 11))
        req.submit()

    for r in requests:
        r.wait()

# Make sure this test occurs entirely within greenlets.
req = Request( functools.partial( lots_of_work ) )
req.submit()
req.wait()

t2 = time.time()
print "\n\n"
print "LAZYFLOW REQUEST WAIT:   %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %0.3fms latency" % ((t2-t1)*1e3/mcount,)


t1 = time.time()

pool = Pool()

for i in range(50000):
    pool.request(functools.partial(empty_func, b = 11))

pool.wait()

t2 = time.time()
print "\n\n"
print "LAZYFLOW POOL WAIT:   %f seconds for %d iterations" % (t2-t1,mcount)
print "                                %0.3fms latency" % ((t2-t1)*1e3/mcount,)
