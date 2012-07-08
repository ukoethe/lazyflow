import greenlet
from collections import deque
import multiprocessing
import threading
from functools import partial

import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(levelname)s %(name)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class RequestGreenlet(greenlet.greenlet):
    def __init__(self, owning_request, fn):
        super(RequestGreenlet, self).__init__(fn)
        self.owning_request = owning_request

class OrderedSignal(object):
    """
    A callback mechanism that ensures callbacks occur in the same order as subscription.
    """
    def __init__(self):
        self.callbacks = []

    def subscribe(self, fn, **kwargs):
        # Remove this function if we already have it
        self.unsubscribe(fn)
        # Add it to the end
        self.callbacks.append((fn, kwargs))

    def unsubscribe(self, fn):
        # Find this function and remove its entry
        for i, (f, kw) in enumerate(self.callbacks):
            if f == fn:
                self.callbacks.pop(i)
                break

    def __call__(self, *args):
        """
        Emit the signal.
        """
        for f, kw in self.callbacks:
            f(*args, **kw)

class Worker(threading.Thread):
    """
    Runs in a loop until stopped.
    The loop pops one request from the threadpool and starts (or resumes) it.
    """
    def __init__(self, thread_pool, index ):
        name = "Worker #{}".format(index)
        super(Worker, self).__init__( name=name )
        self.thread_pool = thread_pool
        self.stopped = False
        self.job_queue_condition = threading.Condition()
        self.job_queue = deque() # Threadsafe for append/pop
        
        loggerName = __name__ + '.Worker{}'.format(index)
        self.logger = logging.getLogger(loggerName)
        self.logger.debug("Created.")
    
    def run(self):
        # Try to get some work.
        current_request = self.get_next_job()

        while not self.stopped:
            # Start (or resume) the work by switching to its greenlet
            current_request.switch_to()

            # Now the request's greenlet has switched back to us.  Did it finish?
            if not current_request.finished:
                # Not finished.  Since its greenlet isn't running, the flag the request as suspended.           
                current_request.suspended.set()

            # Try to get some work.
            current_request = self.get_next_job()

    def get_next_job(self):
        """
        Get the next available job to perform.  Block if necessary until a job is available.
        """
        next_request = self.pop_job()

        # Keep trying until we get a job        
        while next_request is None and not self.stopped:
            with self.job_queue_condition:
                # Wait for work to become available
                self.job_queue_condition.wait()
                next_request = self.pop_job()

        return next_request
    
    def pop_job(self):
        """
        If possible, get a job from our own job queue.
        Otherwise, get one from the global job queue.
        Return None if neither queue has work to do.
        """
        # Try our own queue first
        if len(self.job_queue) > 0:
            return self.job_queue.pop()

        # Otherwise, try to claim a job from the global unassigned list            
        try:
            req = self.thread_pool.unassigned_requests.pop()
            req.set_assigned_worker(self)
            return req
        except IndexError:
            return None
    
    def wake_up(self, request):
        """
        Add this request to the queue of requests that are ready to be processed.
        The request may or not be started already.
        """
        assert request.assigned_worker is self
        with self.job_queue_condition:
            self.job_queue.append(request)
            self.job_queue_condition.notify()
    
    def stop(self):
        self.stopped = True
    
class Singleton(type):
    """
    simple implementation of meta class that implements the singleton pattern.
    """
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls,*args,**kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance

class ThreadPool(object):
    """
    Manages a set of worker threads and dispatches requests to them.
    """
    
    # Thread pool is unique
    __metaclass__ = Singleton

    def __init__(self):
        self.job_condition = threading.Condition()
        self.immediate_work = deque() # Threadsafe for append and pop
        
        self.unassigned_requests = deque()

        num_workers = multiprocessing.cpu_count()
        self.workers = self.start_workers( num_workers )

    def start_workers(self, num_workers):
        workers = set()
        for i in range(num_workers):
            w = Worker(self, i)
            workers.add( w )
            w.start()
            logger.debug("Started " + w.name)
        return workers

    def wake_up(self, request):
        # Once a request has been assigned, it can't eve
        if request.assigned_worker is not None:
            request.assigned_worker.wake_up( request )
        else:
            self.unassigned_requests.append(request)
            # Notify all currently waiting workers that there's new work
            self._notify_all_workers()

    def _notify_all_workers(self):
        for worker in self.workers:
            if worker.job_queue_condition.acquire(False): # Don't block for the condition if the worker is busy.
                worker.job_queue_condition.notify()
                worker.job_queue_condition.release()

    def stop(self):
        for w in self.workers:
            w.stop()
        
        self._notify_all_workers()
        
        for w in self.workers:
            w.join()
    
global_thread_pool = ThreadPool()
            

class Request( object ):
    
    logger = logging.getLogger(__name__ + '.Request')

    def __init__(self, fn):
        # Workload
        self.fn = fn
        self.result = None

        # State
        self.started = False
        self.finished = False
        self.suspended = threading.Event()
        self.suspended.set()
        self.finished_event = threading.Event()

        # Execution
        self.greenlet = None # Not created until assignment to a worker
        self.assigned_worker = None

        # Request relationships
        self.pending_requests = set()  # Requests that are waiting for this one
        self.blocking_requests = set() # Requests that this one is waiting for (currently one at most since wait() can only be called on one request at a time)

        self._lock = threading.RLock()
        self._sig_finished = OrderedSignal()
        
        self.logger.debug("Created request")

    def set_assigned_worker(self, worker):
        """
        Assign this request to the given worker thread.  (A request cannot switch between threads.)
        Must be called from the worker thread.
        """
        self.assigned_worker = worker

        # Create our greenlet now (so the greenlet has the correct parent, i.e. the worker)
        self.greenlet = RequestGreenlet(self, self.execute)

    def execute(self):
        """
        Do the real work of this request.
        """
        self.logger.debug("Executing in " + threading.current_thread().name)

        # Entering our own greenlet.  Not suspended.
        self.suspended.clear()
        
        # Do the actual work
        self.result = self.fn()

        # Notify callbacks and waiting threads
        with self._lock:
            self.finished = True
            self._sig_finished(self.result)
        self.finished_event.set()

        self.logger.debug("Finished")
    
    def submit(self):
        """
        If this request isn't started yet, schedule it to be started.
        """
        with self._lock:
            if not self.started:
                self.started = True
                self.wake_up()
    
    def wake_up(self):
        """
        Resume this request's execution (put it back on the worker's job queue).
        """
        self.logger.debug("Waking up")

        # Not valid to wake up unless we're suspended
        self.suspended.wait()
        global_thread_pool.wake_up(self)

    def switch_to(self):
        """
        Switch to this request's greenlet
        """
        # The greenlet should return to the current worker thread when it's finished,
        #  so make it's parent = the current worker
        self.greenlet.switch()
        
    def suspend(self):
        """
        Suspend this request so another one can be woken up by the worker.
        """
        # Switch back to the worker that we're currently running in.
        # The worker will flag us as suspended.
        self.greenlet.parent.switch()
        
        # Re-entering our own greenlet.  Not suspended.
        self.suspended.clear()
    
    def wait(self):
        """
        Start this request if necessary, then wait for it to complete.  Return the request's result.
        """
        # Schedule this request if it hasn't been scheduled yet.
        self.submit()

        # Identify the request that is waiting for us (the current context)
        current_request = Request.current_request()

        if current_request is None:
            # This is a non-worker thread, so just wait the old-fashioned way
            self.finished_event.wait()
        else:
            # We're running in the context of a request.
            # Instead of blocking the thread, suspend the current request and wake up a different one on this thread.
            with self._lock:
                if not self.finished:
                    current_request.blocking_requests.add(self)
                    self.pending_requests.add(current_request)
                    self.notify_finished( partial(current_request.handle_finished_request, self) )
            current_request.suspend()

        assert self.finished
        return self.result

    def handle_finished_request(self, request, result):
        """
        Called when a request that we were waiting for has completed.
        Wake ourselves up so we can resume execution.
        """
        with self._lock:
            # We're not waiting for this one any more
            self.blocking_requests.remove(request)
            if len(self.blocking_requests) == 0:
                self.wake_up()
    
    def notify_finished(self, fn):
        """
        Register a callback function to be called when this request is finished.
        If we're already finished, call it now.
        """
        # Schedule this request if it hasn't been scheduled yet.
        self.submit()

        with self._lock:            
            if self.finished:
                # Call immediately
                fn(self.result)
            else:
                # Call when we eventually finish
                self._sig_finished.subscribe(fn)
    
    @classmethod
    def current_request(cls):
        """
        Inspect the current greenlet/thread and return the request object associated with it, if any.
        """
        current_greenlet = greenlet.getcurrent()
        # Greenlets in worker threads have a monkey-patched 'owning-request' member
        if hasattr(current_greenlet, 'owning_request'):
            return current_greenlet.owning_request
        else:
            # There is no request associated with this greenlet.
            # It must be a regular (foreign) thread.
            return None
        
if __name__ == "__main__":

    import time
    import atexit
    def onExit():
        print "Exiting..."    
    atexit.register(onExit)

    def g():
        print "g() 1"
        time.sleep(1)
        return 1

    def f():
        print "f() 1"
        time.sleep(1)
        r = Request(g)
        result = r.wait()
        print "f() 2"
        
        return 2*result
    
    r = Request(f)
    res = r.wait()
    print "result =",res

    global_thread_pool.stop()















