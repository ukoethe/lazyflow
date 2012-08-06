import threading
import thread
import zlib
import cPickle as pickle
import zmq
from zmq_worker import dump_func_to_string
from collections import deque


class ZMQMasterWork(threading.Thread):
    def __init__(self, threadpool):
        threading.Thread.__init__(self)
        self.daemon = True
        self._threadpool = threadpool
        self._zmq_ctx = zmq.Context.instance()
        self._running = True
        self._requests = {}
        self._out_requests = deque()
        self._wlock = threading.Lock()
        self._wlock.acquire()
    
    def putRequest(self, request):
        self._out_requests.append(request)
        try:
          self._wlock.release()
        except:
          pass
    
    def _send_request(self, request):
        self._requests[id(request)] = request
        self._zmq_socket_work.send_pyobj({ 'id': id(request), 'function' : dump_func_to_string(request.function), 'kwargs': request.kwargs})

    def run(self):
        self._zmq_socket_work = zmq.Socket(self._zmq_ctx, zmq.PUSH)
        self._zmq_socket_work.bind("tcp://*:6666")
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

    def stop(self):
        self._running = False
        try:
          self._wlock.release()
        except thread.error:
          pass
        self.join()



class ZMQMasterResults(threading.Thread):
    def __init__(self, threadpool, request_dict):
        threading.Thread.__init__(self)
        self.daemon = True
        self._threadpool = threadpool
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
              if msg["finished"]:
                request = self._requests[msg['id']]
                result = msg['result']
                cur_req = self.current_request
                request.after_execute(result, self, cur_req)
              else:
                # reraise remote error locally
                print "Exception in Worker:", msg["traceback"]

    def stop(self):
        self._running = False
        self.join()
        

