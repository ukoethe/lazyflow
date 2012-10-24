import lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot, Graph
from lazyflow.request import Request
from lazyflow.zmq_worker import ZMQWorker
import numpy as np
import numpy as numpy
import random
import time
import vigra
request_count = 10

def bribbel():
    print "jashdkjahsdkjhasdkjhaskdhaksjdhaksdjh"
    return numpy.random.randint(0,200)

def brababbel():
    bribbel()


def brubbel(a):
    l = vigra.analysis.labelVolumeWithBackground(a)
    print "labeled the cool image"
    return l

class OpA(Operator):
    out = OutputSlot()

    def setupOutputs(self):
        self.out.meta.shape = (1,)
        self.out.meta.dtype = object

    def execute(self, slot, subindex, roi, result):

        def blabb( x):
            print x


        
        def blubb( a, b, nr, rest = None):
            print type(rest)
            test = np.ndarray((100,200))
            time.sleep(random.random())
            result = a + b
            blabb("WORKER: Request %d finished" % nr)
            #bribbel()
            #brabbel("brabbel")
            return result

        a = numpy.ndarray((10,2))
        b = numpy.ndarray((10,2))
        a[:] = 1
        b[:] = 2
        
        requests = []
        for i in range(request_count):
            req = Request(blubb, a = a, b = b, nr = i)
            req.submit_cloud()
            requests.append(req)
            def callback(req):
                print "Master: Request %d finished" % req.kwargs["nr"]
            req.onFinish(callback)

        for r in requests:
            r.wait()

        result[0] = req.wait()

        return result

g = Graph()
op = OpA( graph=g )

result = op.out[0].wait()
print "Operator returned"

print result



print "testing funcion call in cloud"
req = Request(bribbel)
req.submit_cloud()
req.wait()

print "testing funcion call in cloud with vigra call"
req = Request(brubbel, a = numpy.random.rand(300,200,300).astype(numpy.float32))
req.submit_cloud()
res = req.wait()
print "finished"
print res

print "testing function call in cloud (other module)"
from zmq_test2 import brubbel
req = Request(brubbel)
req.submit_cloud()
req.wait()

print "testing function call in cloud (other module, alias import)"
from zmq_test2 import brubbel as brub
req = Request(brub)
req.submit_cloud()
req.wait()

# print "testing nested function call"
# try:
#     req = Request(brababbel)
#     req.submit_cloud()
#     req.wait()
# except Exception as e:
#     print e

