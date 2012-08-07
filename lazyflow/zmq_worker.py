import os
import string
import zmq
import marshal, new
import cPickle as pickle
import threading
import traceback
import atexit

def dump_func_to_string(f):
    if f.func_closure:
        closure = tuple(c.cell_contents for c in f.func_closure)
    else:
        closure = None
    globs = f.func_globals 
    globals_mapping = {}
    for k, v in globs.items():
        name = k
        try:
            name = v.__name__
        except:
            pass
        if k != name:
            globals_mapping[k] = name
    return pickle.dumps((marshal.dumps(f.func_code), f.func_defaults, closure, globals_mapping))


def load_func_from_string(s, glob_dict):
    code, defaults, closure, globals_mapping = pickle.loads(s)
    if closure is not None:
        closure = reconstruct_closure(closure)
    code = marshal.loads(code)
    for k,v in globals_mapping.items():
        try:
            glob_dict[k] = __import__(v)
        except:
            pass
    return new.function(code, glob_dict, code.co_name, defaults, closure)


def reconstruct_closure(values):
    ns = range(len(values))
    src = ["def f(arg):"]
    src += [" _%d = arg[%d]" % (n, n) for n in ns]
    src += [" return lambda:(%s)" % ','.join("_%d"%n for n in ns), '']
    src = '\n'.join(src)
    try:
        exec src
    except:
        raise SyntaxError(src)
    return f(values).func_closure


class ZMQWorker(threading.Thread):
    def __init__(self, glob_dict = globals(), master_timeout = 5000):
        threading.Thread.__init__(self)
        self.daemon = True
        self._glob_dict = glob_dict
        
        if not os.environ.has_key("LAZYFLOW_SERVER"):
            raise RuntimeError("ZMQWorker: LAZYFLOW_SERVER environment variable not set. Example: 'tcp://127.0.0.1' ")

        self._server = os.environ["LAZYFLOW_SERVER"]
        self._ctx = zmq.Context()
        self._running = True
        self._current_req = None
        self._timeout = master_timeout
        atexit.register(self.cleanup)

    def send_ack(self, req):
        self._result_socket.send_pyobj({
            "type" : "ack",
            "id" : req["id"],
        })

    def send_result(self, req, result):
        self._result_socket.send_pyobj({
            "type" : "result",
            "id" : req["id"],
            "result" : result
        })

    def send_error(self, req, error, tb):
        self._result_socket.send_pyobj({
            "type" : "error",
            "id" : req["id"],
            "error" : error,
            "traceback" : tb
        })
    

    def send_ready(self):
        self._work_socket.send_pyobj("")

        

    def run(self):
        self._work_socket = zmq.Socket(self._ctx, zmq.REQ)
        self._work_socket.setsockopt(zmq.HWM,1)
        self._work_socket.connect(self._server + ":6666")

        self._result_socket = zmq.Socket(self._ctx, zmq.PUSH)
        self._result_socket.setsockopt(zmq.HWM,1)
        self._result_socket.connect(self._server + ":6667")
        while self._running:
            self.send_ready()
            received = False
            count = 0
            while self._running and received is False and count < self._timeout / 50:
                count += 1
                events = self._work_socket.poll(50) # check every 50ms for thread stop
                if events == zmq.POLLIN:
                    self._current_req = True
                    finished = False
                    self._current_req = req = self._work_socket.recv_pyobj()
                    received = True

                    # handle master stop
                    if req is None:
                        time.sleep(2)
                        count = self._timeout / 50
                        break

                    self.send_ack(req)

                    func = load_func_from_string(req["function"], self._glob_dict)
                    kwargs = req["kwargs"]
                    
                    tried = {}

                    try:
                        while not finished:
                            try:
                                result = func(**kwargs)
                                finished = True
                            except NameError as e:
                                parts = string.split(str(e), "'")
                                module = parts[1]
                                if tried.has_key(module) is False:
                                    tried[module] = True
                                    print "Imorting module", module
                                    globals()[parts[1]] = __import__(module)
                                else:
                                    raise NameError(module)
                    except Exception as e:
                        error = e
                        tb = traceback.format_exc()
                        
                    if finished:
                        self.send_result(req,result)
                    else:
                        self.send_error(req, error, tb)
                        traceback.print_exc()

                    self._current_req = None
                    count = 0
            
            if count >= self._timeout / 50:
                count = 0
                self._work_socket.close(0)
                self._work_socket = zmq.Socket(self._ctx, zmq.REQ)
                self._work_socket.setsockopt(zmq.HWM,1)
                self._work_socket.connect(self._server + ":6666")


    def cleanup(self):
        self._running = False
        self._work_socket.close(0)
        if self._current_req is not None:
            while self._current_req == True:
                time.sleep(0.01)
            self.send_error(self._current_req, "WORKER SHUTDOWN", "WORKER SHUTDOWN")
        self._result_socket.close(-1)

