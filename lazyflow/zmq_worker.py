import os
import string
import zmq
import marshal, new
import cPickle as pickle
import threading
import traceback

def dump_func_to_string(f):
    if f.func_closure:
        closure = tuple(c.cell_contents for c in f.func_closure)
    else:
        closure = None
    return pickle.dumps((marshal.dumps(f.func_code), f.func_defaults, closure))


def load_func_from_string(s, glob_dict):
    code, defaults, closure = pickle.loads(s)
    if closure is not None:
        closure = reconstruct_closure(closure)
    code = marshal.loads(code)
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
    def __init__(self, glob_dict = globals()):
        threading.Thread.__init__(self)
        self.daemon = True
        self._glob_dict = glob_dict

        self._server = os.environ["LAZYFLOW_SERVER"]
        self._ctx = zmq.Context()

    def run(self):
        self._work_socket = zmq.Socket(self._ctx, zmq.PULL)
        self._work_socket.connect(self._server + ":6666")

        self._result_socket = zmq.Socket(self._ctx, zmq.PUSH)
        self._result_socket.connect(self._server + ":6667")
        while True:
            req = self._work_socket.recv_pyobj()
            func = load_func_from_string(req["function"], self._glob_dict)
            kwargs = req["kwargs"]
            
            tried = {}

            finished = False
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
                self._result_socket.send_pyobj({
                    "finished": True,
                    "id" : req["id"],
                    "result" : result
                })
            else:
                self._result_socket.send_pyobj({
                    "finished" : False,
                    "id" : req["id"],
                    "error" : error,
                    "traceback" : tb
                })
                traceback.print_exc()



