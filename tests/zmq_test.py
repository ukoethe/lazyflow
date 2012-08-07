import lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.request import Request
from lazyflow.zmq_worker import ZMQWorker
import numpy as np
import numpy as numpy
import random
request_count = 10

class OpA(Operator):
    out = OutputSlot()

    def setupOutputs(self):
        self.out.meta.shape = (1,)
        self.out.meta.dtype = object

    def execute(self, slot, roi, result):
        
        def blubb( a, b, nr):
            test = np.ndarray((100,200))
            time.sleep(random.random())
            result = a + b
            print "Worker: Request %d finished" % nr
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


op = OpA()

result = op.out[0].wait()
print "Operator returned"

print result
