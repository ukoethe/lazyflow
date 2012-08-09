import os, sys
import time
import string
import zmq
import marshal, new
import cPickle as pickle
import threading
import traceback
import atexit
import copy_reg
import threading
import __builtin__
import imp
import traceback
import types

inc_dir = os.path.expanduser("~/.lazyflow/site-packages")

if os.path.exists(inc_dir) is False:
    os.makedirs(inc_dir)

if inc_dir not in sys.path:
    sys.path.append(inc_dir)



global_import_lock = threading.Lock()

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
        elif hasattr(v, "__module__") and v.__module__ == "__main__":
            pass

    return pickle.dumps((marshal.dumps(f.func_code), f.func_defaults, closure, globals_mapping))


def load_func_from_string(s, glob_dict = globals()):
    code, defaults, closure, globals_mapping = pickle.loads(s)
    if closure is not None:
        closure = reconstruct_closure(closure)
    code = marshal.loads(code)
    for k,v in globals_mapping.items():
        if type(v) == types.StringType:
            try:
                glob_dict[k] = __import__(v)
            except:
                pass
        else:
            print "strange global", v

    return new.function(code, glob_dict, code.co_name, defaults, closure)

def func_pickler(func):
    return func_unpickler, ( dump_func_to_string(func), )

def func_unpickler(s):
    return load_func_from_string(s)

copy_reg.pickle(func_pickler.__class__, func_pickler, func_unpickler)

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


class ZMQDict(dict):
    def __init__(self, src, worker, req):
        self._worker = worker
        self._req = req

    def __getitem__(key):
        if not self.has_key(key):
            result = self._worker.resolve_name(self._req["id"], key)
            self[key] = result
        return dict.__getitem__(key)




class ZMQWorker(threading.Thread):
    def __init__(self, master_timeout = 5000):
        threading.Thread.__init__(self)
        self.daemon = True
        
        if not os.environ.has_key("LAZYFLOW_SERVER"):
            raise RuntimeError("ZMQWorker: LAZYFLOW_SERVER environment variable not set. Example: 'tcp://127.0.0.1' ")

        self._server = os.environ["LAZYFLOW_SERVER"]
        self._ctx = zmq.Context()
        self._running = True
        self._current_req = None
        self._timeout = master_timeout
        atexit.register(self.cleanup)

    def send_ack(self, req):
        self._notify_socket.send_pyobj({
            "type" : "ack",
            "id" : req["id"],
        })

    def send_result(self, req, result):
        self._notify_socket.send_pyobj({
            "type" : "result",
            "id" : req["id"],
            "result" : result
        })

    def send_error(self, req, error, tb):
        self._notify_socket.send_pyobj({
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

        self._notify_socket = zmq.Socket(self._ctx, zmq.PUSH)
        self._notify_socket.setsockopt(zmq.HWM,1)
        self._notify_socket.connect(self._server + ":6667")
        
        self._request_socket = zmq.Socket(self._ctx, zmq.REQ)
        self._request_socket.setsockopt(zmq.HWM,1)
        self._request_socket.connect(self._server + ":6668")

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
                    self._set_import_hook()
                    glob_dict = ZMQDict(globals(), self, req)

                    received = True

                    # handle master stop
                    if req is None:
                        time.sleep(2)
                        count = self._timeout / 50
                        break

                    self.send_ack(req)

                    func = load_func_from_string(req["function"], glob_dict)
                    kwargs = req["kwargs"]
                    
                    tried = {}
                    

                    try:
                        while not finished:
                            try:
                                result = func(**kwargs)
                                finished = True
                            except NameError as e:
                                parts = string.split(str(e), "'")
                                name = parts[1]
                                if tried.has_key(name) is False:
                                    tried[name] = True
                                    res = self.resolve_name(req["id"], name)
                                    glob_dict[name] = res
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
    
    
    def resolve_name(self, reqid, name):
        print "Resolving ", name
        self._request_socket.send_pyobj({
            "type" : "get_name",
            "id" : reqid,
            "name" : name
        })

        answer = self._request_socket.recv_pyobj()

        if answer["type"] == "import":
            result = __import__(name)
            print "   received import"
        elif answer["type"] == "something":
            result = pickle.loads(answer["object"])
            print "   received ",type(result)
        return result


    def get_import(self, name, globals, locals, fromlist):
        print "WORKER: fetching %s from server" % name
        self._request_socket.send_pyobj({
            "type" : "import",
            "name" : name, 
            "globals" : globals,
            "locals" : locals,
            "fromlist" : fromlist
        })
        error, files = self._request_socket.recv_pyobj()

        for mod in files:
            fname = string.split(mod[1],"/")[-1]
            if fname in ["__init__.py", "__init__.pyc"]:
                fdir = inc_dir + "/" + mod[0].replace(".","/") + "/"
            else:
                fdir = inc_dir + "/" + string.join(string.split(mod[0],".")[:-1], "/")

            print "      writing", fdir + fname
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            f = open(fdir + fname, "w")
            f.write(mod[2])
            f.close()

        return error

    def _set_import_hook(self):
        if __builtin__.__import__ != import_hook:
            __builtin__.__import__ = import_hook

    

    def cleanup(self):
        self._running = False
        self._work_socket.close(0)
        if self._current_req is not None:
            while self._current_req == True:
                time.sleep(0.01)
            self.send_error(self._current_req, "WORKER SHUTDOWN", "WORKER SHUTDOWN")
        self._notify_socket.close(-1)




# save original import
original_import = __builtin__.__import__


# Replacement for __import__()
def import_hook(name, globals=None, locals=None, fromlist=None, *args, **kwargs):
    local_globs = {}
    cur_tr = threading.current_thread()
    if isinstance(cur_tr, ZMQWorker):
        if globals is not None:
            if globals.has_key("__name__"):
                local_globs["__name__"] = globals["__name__"]
            if globals.has_key("__path__"):
                local_globs["__path__"] = globals["__path__"]
        try:
            m = original_import(name, local_globs, locals, fromlist)
        except ImportError:
            # try fetching from master
            error = cur_tr.get_import(name, local_globs, None, fromlist)
            # if import on server failed, reraise here on worker
            if error:
                raise error
            m = import_hook(name, local_globs, locals, fromlist)
    else:
        m = original_import(name, local_globs, locals, fromlist)
    
    return m
