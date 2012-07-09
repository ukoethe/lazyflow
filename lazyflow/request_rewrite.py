import greenlet
from collections import deque
import multiprocessing
import threading
from functools import partial

import sys
import logging
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#handler = logging.StreamHandler(sys.stdout)
#formatter = logging.Formatter('%(levelname)s %(name)s %(message)s')
#handler.setFormatter(formatter)
#logger.addHandler(handler)

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
        """
        Keep executing available jobs (requests) until we're stopped.
        """
        # Try to get some work.
        current_request = self._get_next_job()

        while not self.stopped:
            # Start (or resume) the work by switching to its greenlet
            current_request.switch_to()

            # Try to get some work.
            current_request = self._get_next_job()

    def stop(self):
        """
        Tell this worker to stop running.
        Does not block for thread completion.
        """
        self.stopped = True
        # Wake up the thread if it's waiting for work
        with self.job_queue_condition:
            self.job_queue_condition.notify()

    def wake_up(self, request):
        """
        Add this request to the queue of requests that are ready to be processed.
        The request may or not be started already.
        """
        assert request.assigned_worker is self
        with self.job_queue_condition:
            self.job_queue.append(request)
            self.job_queue_condition.notify()

    def _get_next_job(self):
        """
        Get the next available job to perform.  Block if necessary until a job is available.
        """
        next_request = None

        # Keep trying until we get a job        
        with self.job_queue_condition:
            next_request = self._pop_job()

            while next_request is None and not self.stopped:
                # Wait for work to become available
                self.logger.debug("Waiting for work...")
                self.job_queue_condition.wait()
                next_request = self._pop_job()

        assert next_request is not None or self.stopped
        
        if not self.stopped:
            self.logger.debug("Got work.")

        return next_request
    
    def _pop_job(self):
        """
        Non-blocking.
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
        self.workers = self._start_workers( num_workers )

    def wake_up(self, request):
        """
        Schedule the request its assigned worker.
        Assign it a worker first if necessary.
        """
        # Once a request has been assigned, it must always be processed in the same worker
        if request.assigned_worker is not None:
            request.assigned_worker.wake_up( request )
        else:
            self.unassigned_requests.append(request)
            # Notify all currently waiting workers that there's new work
            logger.debug("Notifying workers of new work")
            self._notify_all_workers()

    def stop(self):
        """
        Stop all threads in the pool, and block for them to complete.
        """
        for w in self.workers:
            w.stop()
        
        for w in self.workers:
            w.join()
    
    def _start_workers(self, num_workers):
        """
        Start a set of workers and return the set.
        """
        workers = set()
        for i in range(num_workers):
            w = Worker(self, i)
            workers.add( w )
            w.start()
            logger.debug("Started " + w.name)
        return workers

    def _notify_all_workers(self):
        """
        Wake up all worker threads that are currently waiting for work.
        """
        for worker in self.workers:
            with worker.job_queue_condition:
                worker.job_queue_condition.notify()

global_thread_pool = ThreadPool()
            

class Request( object ):
    
    logger = logging.getLogger(__name__ + '.Request')

    class CancellationException(Exception):
        """
        This is thrown when the whole request has been cancelled.
        If you catch this exception, clean up and return immediately.
        """
        pass

    class InvalidRequestException(Exception):
        """
        This is thrown when calling wait on a request that has already been cancelled,
        which can only happen if the request was spawned elsewhere 
        (i.e. you are waiting for someone else's request to avoid duplicate work).
        When this occurs, you will typically restart the request yourself.
        """
        pass

    def __init__(self, fn):
        # Workload
        self.fn = fn
        self.result = None

        # State
        self.started = False
        self.cancelled = False
        self.uncancellable = False
        self.finished = False
        self.finished_event = threading.Event()

        # Execution
        self.greenlet = None # Not created until assignment to a worker
        self.assigned_worker = None

        # Request relationships
        self.pending_requests = set()  # Requests that are waiting for this one
        self.blocking_requests = set() # Requests that this one is waiting for (currently one at most since wait() can only be called on one request at a time)
        self.child_requests = set()    # Requests that were created from within this request (NOT the same as pending_requests)
        
        current_request = Request.current_request()
        self.parent_request = current_request
        if current_request is not None:
            current_request.child_requests.add(self)

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

        # Did someone cancel us before we even started?
        if not self.cancelled:
            try:
                # Do the actual work
                self.result = self.fn()
            except Request.CancellationException:
                # Don't propagate cancellations back to the worker thread,
                # even if the user didn't catch them.
                pass

        with self._lock:
            self.finished = True

        if not self.cancelled:
            # Notify callbacks
            self._sig_finished(self.result)
        else:
            self.logger.debug("Finished a cancelled request.")

        # Notify non-request-based threads
        self.finished_event.set()

        self.logger.debug("Finished")
    
    def submit(self):
        """
        If this request isn't started yet, schedule it to be started.
        """
        with self._lock:
            if not self.started:
                self.started = True
                self._wake_up()
    
    def _wake_up(self):
        """
        Resume this request's execution (put it back on the worker's job queue).
        """
        self.logger.debug("Waking up")
        global_thread_pool.wake_up(self)
 
    def switch_to(self):
        """
        Switch to this request's greenlet
        """
        self.greenlet.switch()
        
    def _suspend(self):
        """
        Suspend this request so another one can be woken up by the worker.
        """
        # Switch back to the worker that we're currently running in.
        self.greenlet.parent.switch()
        
        # Now we're back (no longer suspended)
        # Were we cancelled in the meantime?
        if self.cancelled:
            raise Request.CancellationException()
    
    def wait(self):
        """
        Start this request if necessary, then wait for it to complete.  Return the request's result.
        """
        # Schedule this request if it hasn't been scheduled yet.
        self.submit()

        # Identify the request that is waiting for us (the current context)
        current_request = Request.current_request()

        if current_request is None:
            # Don't allow this request to be cancelled, since a real thread is waiting for it.
            self.uncancellable = True

            # This is a non-worker thread, so just block the old-fashioned way
            self.finished_event.wait()
            
            # It turns out this request was already cancelled.
            if self.cancelled:
                raise Request.InvalidRequestException()
        else:
            # We're running in the context of a request.
            # If we have to wait, suspend the current request instead of blocking the thread.
            with self._lock:
                # Before we suspend the current request, check to see if it's been cancelled since it last blocked
                if current_request.cancelled:
                    raise Request.CancellationException()
                
                # If the current request isn't cancelled but we are,
                # then the current request is trying to wait for a request (i.e. self) that was spawned elsewhere and already cancelled.
                # If they really want it, they'll have to spawn it themselves.
                if self.cancelled:
                    raise Request.InvalidRequestException()

                suspend_needed = not self.finished
                if suspend_needed:
                    current_request.blocking_requests.add(self)
                    self.pending_requests.add(current_request)
                    self.notify_finished( partial(current_request._handle_finished_request, self) )

            if suspend_needed:
                current_request._suspend()

        assert self.finished
        return self.result

    def _handle_finished_request(self, request, result):
        """
        Called when a request that we were waiting for has completed.
        Wake ourselves up so we can resume execution.
        """
        with self._lock:
            # We're not waiting for this one any more
            self.blocking_requests.remove(request)
            if len(self.blocking_requests) == 0:
                self._wake_up()
    
    def notify_finished(self, fn):
        """
        Register a callback function to be called when this request is finished.
        If we're already finished, call it now.
        """
        # Schedule this request if it hasn't been scheduled yet.
        self.submit()

        with self._lock:
            finished = self.finished
            if not finished:
                # Call when we eventually finish
                self._sig_finished.subscribe(fn)

        if finished:
            # Call immediately
            fn(self.result)
        
    def cancel(self):
        # We can only be cancelled if: 
        # (1) There are no foreign threads blocking for us (flagged via self.uncancellable) AND
        # (2) our parent request (if any) is already cancelled AND
        # (3) all requests that are pending for this one are already cancelled
        with self._lock:
            cancelled = not self.uncancellable
            cancelled &= (self.parent_request is None or self.parent_request.cancelled)
            for r in self.pending_requests:
                cancelled &= r.cancelled

            self.cancelled = cancelled

        if self.cancelled:
            # Cancel all requests that were spawned from this one.
            for child in self.child_requests:
                child.cancel()
    
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











