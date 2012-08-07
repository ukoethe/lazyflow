import threading
import thread
import time
import zlib
import cPickle as pickle
import zmq
from zmq_worker import dump_func_to_string
from collections import deque


class ZMQMasterWork(threading.Thread):
    def __init__(self, threadpool, freeworker_timeout = 10000):
        threading.Thread.__init__(self)
        self.daemon = True
        self._threadpool = threadpool
        self._watchdog = threadpool._zmq_watch

        self._zmq_ctx = zmq.Context() # create own context
        self._running = True
        self._requests = {}
        self._out_requests = deque()
        self._wlock = threading.Lock()
        self._wlock.acquire()
        self._timeout = freeworker_timeout
    
    def putRequest(self, request):
        self._watchdog.ack(id(request)) #make sure request is not watch to prevent double entries in queue due to watchdog
        if request.finished is True:
            return
        self._out_requests.append(request)
        try:
          self._wlock.release()
        except:
          pass
    
    def _send_request(self, request):
        if not self._requests.has_key(id(request)):
            pass
        found_worker = False
        while found_worker is False:
            events = self._zmq_socket_work.poll(self._timeout) 
            if events == zmq.POLLIN:
                token = self._zmq_socket_work.recv_pyobj()
                self._requests[id(request)] = request
                print "   sending..."
                self._zmq_socket_work.send_pyobj({ 'id': id(request), 'function' : dump_func_to_string(request.function), 'kwargs': request.kwargs})
                print "   done"
                self._watchdog.watch(request) # add request to watchdog
                found_worker = True
            else:
                # if no free worker is found within timeout, reset connection
                print "No free worker - resetting"
                self._zmq_socket_work.close(-1)
                self._zmq_ctx.term()
                self._zmq_ctx = zmq.Context()
                self._zmq_socket_work = zmq.Socket(self._zmq_ctx, zmq.REP)
                self._zmq_socket_work.bind("tcp://*:6666")
                self._zmq_socket_work.setsockopt(zmq.HWM,1)


    def run(self):
        self._zmq_socket_work = zmq.Socket(self._zmq_ctx, zmq.REP)
        #self._zmq_socket_work.setsockopt(zmq.IDENTITY, "LAZYFLOW")
        self._zmq_socket_work.bind("tcp://*:6666")
        self._zmq_socket_work.setsockopt(zmq.HWM,1)
        while self._running:
            self._wlock.acquire()
            try:
              self._wlock.release()
              self._wlock.acquire()
            except thread.error:
              pass

            while len(self._out_requests) > 0:
              out_req = None
              out_req = self._out_requests.popleft()
              self._send_request(out_req)

        # notify workers of master shutdown
        while self._zmq_socket_work.poll(500) != 0:
            print("resetting worker")
            token = self._zmq_socket_work.recv_pyobj()
            self._zmq_socket_work.send_pyobj(None)

    def stop(self):
        self._running = False
        try:
          self._wlock.release()
        except thread.error:
          pass
        self.join()

class ZMQWatchDog(threading.Thread):
    def __init__(self, threadpool, ack_timout = 2000):
        threading.Thread.__init__(self)
        self.daemon = True
        self._threadpool = threadpool
        self._zmq_ctx = zmq.Context.instance()
        self._running = True
        self._unacknowledged = {}
        self._lock = threading.Lock()
        self._timout = ack_timout / 1000.0


    def watch(self, req):
        self._lock.acquire()
        self._unacknowledged[id(req)] = (req, time.time())
        self._lock.release()

    def ack(self, reqid):
        self._lock.acquire()
        if self._unacknowledged.has_key(reqid):
            del self._unacknowledged[reqid]
        self._lock.release()

    def again(self, req):
        self.ack(id(req))
        self._work.putRequest(req)

    def run(self):
        self._work = self._threadpool._zmq_work

        while self._running:
            time.sleep(0.2) # check from time to time if all unacknowledge requests have been acknowledges
            self._lock.acquire()
            items = self._unacknowledged.items()
            self._lock.release()
            t2 = time.time()
            for reqid, pair in items:
                req, t = pair
                if t2 - t > self._timout:
                    print "Request ACK Timeout for request %r --> send again" % reqid
                    self.again(req)
    
    def stop(self):
        self._running = False
        self.join()
    


class ZMQMasterResults(threading.Thread):
    def __init__(self, threadpool, request_dict):
        threading.Thread.__init__(self)
        self.daemon = True
        self._threadpool = threadpool
        self._watchdog = threadpool._zmq_watch

        self._zmq_ctx = zmq.Context.instance()
        self._running = True
        self._requests = request_dict


    def run(self):
        self._zmq_socket_results = zmq.Socket(self._zmq_ctx, zmq.PULL)
        self._zmq_socket_results.bind("tcp://*:6667")
        while self._running:
            events = self._zmq_socket_results.poll(50) # check every 50ms for thread stop
            if events != 0:
                msg = self._zmq_socket_results.recv_pyobj()
                if not self._requests.has_key(msg['id']):
                    print "Master: obtained results for unknown request %r, silently dropping..." % msg['id']
                    continue
                
                if msg["type"] == "result":
                    request = self._requests[msg['id']]
                    result = msg['result']
                    cur_req = self.current_request
                    if request.finished is True:
                        continue
                    request.after_execute(result, self, cur_req)
                    del self._requests[msg['id']] # remove request from dict to free memory
                elif msg["type"] == "ack":
                    reqid = msg["id"]
                    self._watchdog.ack(reqid)
                elif msg["type"] == "error":
                    # reraise remote error locally
                    print "Master: Request %r resulted in exception in Worker %s" % (msg['id'], msg["traceback"])
                    print "Master: Restarting request", msg['id']
                    req = self._requests[msg['id']] 
                    del self._requests[msg['id']] # remove from dict to allow resubmission
                    self._threadpool._zmq_work.putRequest(req)

    def stop(self):
        self._running = False
        self.join()
        

