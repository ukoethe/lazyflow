import threading
import thread
import time
import zlib
import cPickle as pickle
import zmq
from zmq_worker import dump_func_to_string
from collections import deque
import __builtin__


class ZMQMasterWork(threading.Thread):
    def __init__(self, threadpool, freeworker_timeout = 1000):
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
            events = self._work_socket.poll(self._timeout) 
            if events == zmq.POLLIN:
                token = self._work_socket.recv_pyobj()
                assert token == "ready", token
                self._requests[id(request)] = request
                print "found free worker -> sending request"
                self._work_socket.send_pyobj({ 'id': id(request), 'function' : dump_func_to_string(request.function), 'kwargs': request.kwargs})
                self._watchdog.watch(request) # add request to watchdog
                found_worker = True
            else:
                # if no free worker is found within timeout, reset connection
                print "No free worker - resetting"
                self._work_socket.close(0)
                self._zmq_ctx.term()
                self._zmq_ctx = zmq.Context()
                self._work_socket = zmq.Socket(self._zmq_ctx, zmq.REP)
                self._work_socket.setsockopt(zmq.HWM,1)
                self._work_socket.bind("tcp://*:6666")


    def run(self):
        self._work_socket = zmq.Socket(self._zmq_ctx, zmq.REP)
        #self._work_socket.setsockopt(zmq.IDENTITY, "LAZYFLOW")
        self._work_socket.bind("tcp://*:6666")
        self._work_socket.setsockopt(zmq.HWM,1)
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
        while self._work_socket.poll(500) != 0:
            print("resetting worker")
            token = self._work_socket.recv_pyobj()
            self._work_socket.send_pyobj(None)

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
        self._timout = 10


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
    







class ZMQMasterNotifications(threading.Thread):
    def __init__(self, threadpool, request_dict):
        threading.Thread.__init__(self)
        self.daemon = True
        self._threadpool = threadpool
        self._watchdog = threadpool._zmq_watch

        self._zmq_ctx = zmq.Context.instance()
        self._running = True
        self._requests = request_dict


    def run(self):
        self._notify_socket = zmq.Socket(self._zmq_ctx, zmq.PULL)
        self._notify_socket.bind("tcp://*:6667")
        while self._running:
            events = self._notify_socket.poll(50) # check every 50ms for thread stop
            if events != 0:
                msg = self._notify_socket.recv_pyobj()
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




class ZMQMasterRequests(threading.Thread):
    def __init__(self, threadpool, request_dict):
        threading.Thread.__init__(self)
        self.daemon = True
        self._threadpool = threadpool

        self._zmq_ctx = zmq.Context.instance()
        self._running = True
        self._requests = request_dict


    def run(self):
        self._request_socket = zmq.Socket(self._zmq_ctx, zmq.REP)
        self._request_socket.bind("tcp://*:6668")
        while self._running:
            events = self._request_socket.poll(50) # check every 50ms for thread stop
            if events != 0:
                msg = self._request_socket.recv_pyobj()
                if msg["type"] == "import":
                    print "MASTER: resolving import dependencies for %s" % msg["name"]
                    answer = self.get_import_dependencies(msg)
                    self._request_socket.send_pyobj(answer)
                
                if msg["type"] == "get_name":
                    print "MASTER: resolving name %s" % msg["name"]
                    req = self._requests[msg["id"]]
                    func = req.function
                    globs = func.func_globals
                    res = globs[msg["name"]]
                    if hasattr(res, "__module__") and res.__module__ == func.__module__:
                        if func.__module__  != "__main__":
                            answer = {
                                "type" : "something",
                                "object" : pickle.dumps(res)
                            }
                            print "   type:", type(res)
                        else:
                            raise Exception("lazyflow: transmitting dependent functions or objects from __main__: NOT YET IMPLEMENTED\n please use functions or classes defined in a DIFFERENT PYTHON file and import them for now - this is supported.")
                    else:
                        answer = {
                            "type" : "import"
                        }
                        print "   type: import"
                    self._request_socket.send_pyobj(answer)


    def _import_hook(self, name, globals=None, locals=None, fromlist=None, *args, **kwargs):
        m = self.original_import(name, globals, locals, fromlist, *args, **kwargs)
        if hasattr(m, "__file__"):
            f = open(m.__file__, "r")
            content = f.read()
            self.import_chain.append([m.__name__, m.__file__, content])

            # restore original hook after first valid file
            __builtin__.__import__ = self.original_import
        
        return m



    def get_import_dependencies(self, msg):
        self.import_chain = []

        # Save the original hooks
        self.original_import = __builtin__.__import__

        # Now install our hook
        __builtin__.__import__ = self._import_hook
        error = None

        try:
            __import__(msg["name"], msg["globals"], msg["locals"], msg["fromlist"])
        except Exception as e:
            error = e


        # restore original hook
        __builtin__.__import__ = self.original_import

        print "   IMPORT CHAIN"
        for i in self.import_chain:
            print "      ",i[0], i[1]


        return error, self.import_chain



    def stop(self):
        self._running = False
        self.join()
