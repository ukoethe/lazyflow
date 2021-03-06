"""
This module implements the basic flow graph
of the lazyflow module.

Basic usage example:

---
import numpy
import lazyflow.graph
from lazyflow.operators.operators import  OpArrayPiper


g = lazyflow.graph.Graph()

operator1 = OpArrayPiper(graph=g)
operator2 = OpArrayPiper(graph=g)

operator1.inputs["Input"].setValue(numpy.zeros((10,20,30), dtype=numpy.uint8))

operator2.inputs["Input"].connect(operator1.outputs["Output"])

result = operator2.outputs["Output"][:].allocate().wait()

g.finalize()
---


"""

import lazyflow
import numpy
import sys
import copy
import psutil
import functools
import collections
import itertools

if (int(psutil.__version__.split(".")[0]) < 1 and
    int(psutil.__version__.split(".")[1]) < 3):
    print ("Lazyflow: Please install a psutil python module version"
           " of at least >= 0.3.0")
    sys.exit(1)

import threading
import logging

from request import Request
import lazyflow.utility
import rtype
from lazyflow.stype import ArrayLike
from lazyflow.utility import slicingtools

from lazyflow.utility import Tracer, OrderedSignal

class MetaDict(dict):
    """
    Helper class that manages the dirty state of the meta data of a slot.
    changing a meta dicts attributes sets it _dirty flag True.
    """
    def __init__(self, other=False):
        if(other):
            dict.__init__(self, other)
        else:
            dict.__init__(self)
            # flag that indicates whether all dependencies of the slot
            # are ready
            self._ready = False

        # flag that indicates whether any piece of meta information
        # changed since this flag was reset
        self._dirty = True

    def __setattr__(self, name, value):
        """Provide convenient acces to the metadict, allows using the
        . notation instead of [] access

        """
        if self.has_key(name):
            changed = True
            try:
                if self[name] == value:
                    changed = False
            except KeyError:
                pass
            if changed:
                self["_dirty"] = True
        self[name] = value
        return value

    def __getattr__(self, name):
        """Provide convenient acces to the metadict, allows using the
        . notation instead of [] access

        """
        return self[name]

    def __getitem__(self, name):
        """
        Does not throw KeyErrors.
        Non-existant items always return None.
        """
        if name in self.keys():
            return dict.__getitem__(self, name)
        else:
            return None

    def copy(self):
        """
        Construct a copy of the meta dict
        """
        return MetaDict(dict.copy(self))

    def assignFrom(self, other):
        """
        Copy all the elements from other into this
        """
        dirty = not (self == other)
        origdirty = self._dirty
        origready = self._ready
        if dirty:
            self.clear()
            for k, v in other.items():
                self[k] = copy.copy(v)
        self._dirty = origdirty | dirty

        # Readiness can't be assigned. It can only be assigned in
        # _setupOutputs or setValue (or copied via _changed)
        self._ready = origready

    def getTaggedShape(self):
        """Convenience function for creating an OrderedDict of axistag
        keys and shape dimensions.

        """
        assert self.axistags is not None
        assert self.shape is not None
        keys = self.getAxisKeys()
        return collections.OrderedDict(zip(keys, self.shape))

    def getAxisKeys(self):
        assert self.axistags is not None
        return [tag.key for tag in self.axistags]

    def getDtypeBytes(self):
        """
        For numpy dtypes only, return the size of the dtype in bytes.
        """
        dtype = self.dtype
        if type(dtype) is numpy.dtype:
            # Make sure we're dealing with a type (e.g. numpy.float64),
            #  not a numpy.dtype
            dtype = dtype.type        
        return dtype().nbytes


class ValueRequest(object):
    """Pseudo request that behaves like a request.Request object.

    This object is used to prevent the heavy construction of complete
    Request objects in simple cases where they are not needed.

    """
    def __init__(self, value):
        self.result = value

    def wait(self):
        return self.result

    def submit(self):
        pass

    def notify(self, callback, *args, **kwargs):
        callback(self.result, *args, **kwargs)

    def onCancel(self, callback, *args, **kwargs):
        pass

    def onFinish(self, callback, **kwargs):
        callback(self, **kwargs)

    def clean(self):
        self.result = None

    def allocate(self, priority=0):
        return self

    def writeInto(self, destination):
        destination[:] = self.result
        return self

    def getResult(self):
        return self.result


class Slot(object):
    """
    Base class for InputSlot, OutputSlot
    """

    loggerName = __name__ + '.Slot'
    logger = logging.getLogger(loggerName)
    traceLogger = logging.getLogger('TRACE.' + loggerName)

    # Allow slots to be sorted by their order of creation for debug
    # output and diagramming purposes.
    _global_counter = itertools.count()

    @property
    def graph(self):
        return self.operator.graph

    def __init__(self, name="", operator=None, stype=ArrayLike,
                 rtype=rtype.SubRegion, value=None, optional=False,
                 level=0, nonlane=False):
        """Constructor of the Slot class.

        :param name: user readable name of the slot, is normally
          assigned automatically by the Operator

        :param operator: the parent operator of a slot

        :param stype: the slot type (see stype.py)

        :param rtype: the region of interest type (see rtype.py)

        :param value: the default value of the slot

        :param optional: if True this means the slot needs a value or
          connection for its parent operator to be functional

        :param level: defines the dimensionality of the slot. 0 for
          single element (e.g. single numpy.ndarray), 1 for list of
          elements (e.g. list of strings), 2 for list of list of
          elements.

        :param nonlane: For multislot, this flag protects it from
          being considered lane-indexed

        """
        if not hasattr(self, "_type"):
            self._type = None
        if type(stype) == str:
            stype = ArrayLike
        self.partners = []
        self.name = name
        self._optional = optional
        self.operator = operator

        # in the case of an InputSlot this is the slot to which it is
        # connected
        self.partner = None
        self.level = level

        # in the case of an InputSlot one can directly assign a value
        # to a slot instead of connecting it to a partner, this
        # attribute holds the value
        self._value = None

        self._defaultValue = value

        # Causes calls to setValue to be propagated backwards to the
        # partner slot. Used by the OperatorWrapper.
        self._backpropagate_values = False

        self.rtype = rtype

        # the MetaDict that holds the slots meta information
        self.meta = MetaDict()

        # if level > 0, this holds the sub-Input/Output slots
        self._subSlots = []
        self._stypeType = stype

        # the slot type instance
        self.stype = stype(self)
        self.nonlane = nonlane

        self._sig_changed = OrderedSignal()
        self._sig_value_changed = OrderedSignal()
        self._sig_ready = OrderedSignal()
        self._sig_unready = OrderedSignal()
        self._sig_dirty = OrderedSignal()
        self._sig_connect = OrderedSignal()
        self._sig_disconnect = OrderedSignal()
        self._sig_resize = OrderedSignal()
        self._sig_resized = OrderedSignal()
        self._sig_remove = OrderedSignal()
        self._sig_removed = OrderedSignal()
        self._sig_preinsertion = OrderedSignal()
        self._sig_inserted = OrderedSignal()

        self._resizing = False

        self._executionCount = 0
        self._settingUp = False
        self._condition = threading.Condition()

        # Allow slots to be sorted by their order of creation for
        # debug output and diagramming purposes.
        self._global_slot_id = Slot._global_counter.next()

    ###########################
    #  A p i    M e t h o d s #
    ###########################
    def notifyDirty(self, function, **kwargs):
        """
        calls the corresponding function when the slot gets dirty
        first argument of the function is the slot, second argument the roi
        the keyword arguments follow
        """
        self._sig_dirty.subscribe(function, **kwargs)


    def notifyMetaChanged(self, function, **kwargs):
        """calls the corresponding function when the slot meta
        information is changed

        first argument of the function is the slot
        the keyword arguments follow

        """

        self._sig_changed.subscribe(function, **kwargs)

    def notifyValueChanged(self, function, **kwargs):
        """Used by slots with cached values to notify when the cache
        has changed, even if the output is not dirty.

        """
        self._sig_value_changed.subscribe(function, **kwargs)

    def notifyReady(self, function, **kwargs):
        """Calls the corresponding function when the slot is "ready",
        meaning it is connected and will produce data when called.
        This is implemented by manipulating and monitoring a flag in
        the slot metadata.

        first argument of the function is the slot
        the keyword arguments follow

        """
        self._sig_ready.subscribe(function, **kwargs)

    def notifyUnready(self, function, **kwargs):
        """
        Subscribe to "unready" callbacks.  See notifyReady for details.
        """
        self._sig_unready.subscribe(function, **kwargs)

    def _notifyConnect(self, function, **kwargs):
        """
        calls the corresponding function when the slot is connected
        first argument of the function is the slot
        the keyword arguments follow
        """
        self._sig_connect.subscribe(function, **kwargs)

    def notifyDisconnect(self, function, **kwargs):
        """
        calls the corresponding function when the slot is disconnected
        first argument of the function is the slot
        the keyword arguments follow
        """
        self._sig_disconnect.subscribe(function, **kwargs)

    def notifyResize(self, function, **kwargs):
        """
        calls the corresponding function before the slot is resized
        first argument of the function is the slot
        second argument is the old size and the third
        argument is the new size
        the keyword arguments follow
        """
        self._sig_resize.subscribe(function, **kwargs)

    def notifyResized(self, function, **kwargs):
        """
        calls the corresponding function after the slot is resized
        first argument of the function is the slot
        second argument is the old size and the third
        argument is the new size
        the keyword arguments follow
        """
        self._sig_resized.subscribe(function, **kwargs)

    def notifyRemove(self, function, **kwargs):
        """
        calls the corresponding function BEFORE a slot is removed
        first argument of the function is the slot
        second argument is the old size and the third
        argument is the new size
        the keyword arguments follow
        """
        self._sig_remove.subscribe(function, **kwargs)

    def notifyRemoved(self, function, **kwargs):
        """
        calls the corresponding function AFTER a slot is removed
        first argument of the function is the slot
        second argument is the old size and the third
        argument is the new size
        the keyword arguments follow
        """
        self._sig_removed.subscribe(function, **kwargs)

    def notifyPreInsertion(self, function, **kwargs):
        """
        Called immediately before a slot is going to be inserted into a multi-slot.
        Same signature as the notifyInserted signal.
        """
        self._sig_preinsertion.subscribe(function, **kwargs)

    def notifyInserted(self, function, **kwargs):
        """
        calls the corresponding function after a slot has been added
        first argument of the function is the slot
        second argument is the old size and the third
        argument is the new size
        the keyword arguments follow
        """
        self._sig_inserted.subscribe(function, **kwargs)


    def unregisterDirty(self, function):
        """
        unregister a dirty callback
        """
        self._sig_dirty.unsubscribe(function)

    def _unregisterConnect(self, function):
        """
        unregister a connect callback
        """
        self._sig_connect.unsubscribe(function)

    def unregisterDisconnect(self, function):
        """
        unregister a disconnect callback
        """
        self._sig_disconnect.unsubscribe(function)

    def unregisterMetaChanged(self, function):
        """
        unregister a changed callback
        """
        self._sig_changed.unsubscribe(function)

    def unregisterValueChanged(self, function):
        """
        unregister a value changed callback
        """
        self._sig_value_changed.unsubscribe(function)

    def unregisterReady(self, function):
        """
        unregister a ready callback
        """
        self._sig_ready.unsubscribe(function)

    def unregisterUnready(self, function):
        """
        unregister an unready callback
        """
        self._sig_unready.unsubscribe(function)

    def unregisterResize(self, function):
        """
        unregister a resize callback
        """
        self._sig_resize.unsubscribe(function)

    def unregisterResized(self, function):
        """
        unregister a resized callback
        """
        self._sig_resized.unsubscribe(function)

    def unregisterRemove(self, function):
        """
        unregister a remove callback
        """
        self._sig_remove.unsubscribe(function)

    def unregisterRemoved(self, function):
        """
        unregister a removed callback
        """
        self._sig_removed.unsubscribe(function)


    def unregisterPreInsertion(self, function):
        """
        unregister a inserted callback
        """
        self._sig_preinsertion.unsubscribe(function)

    def unregisterInserted(self, function):
        """
        unregister a inserted callback
        """
        self._sig_inserted.unsubscribe(function)

    def connect(self, partner, notify=True):
        """
        Connect a slot to another slot

        Arguments:
          partner   : the slot to which this slot is conencted
        """
        if partner is None:
            self.disconnect()
            return

        assert isinstance(partner, Slot), ("Slot.connect() can only be used to"
                                           " connect other Slots.  Did you mean"
                                           " to use Slot.setValue()?")

        if self.partner == partner and partner.level == self.level:
            return
        if self.level == 0:
            self.disconnect()

        if partner is not None:
            self._value = None
            if partner.level == self.level:
                assert isinstance(partner.stype, type(self.stype)), \
                    "Can't connect slots of non-matching stypes!" \
                    " Attempting to connect '{}' (stype: {}) to '{}' (stype: {})".format(self.name, self.stype, partner.name, partner.stype)
                self.partner = partner
                notifyReady = (self.partner.meta._ready and
                               not self.meta._ready)
                self.meta = self.partner.meta.copy()

                # the slot with more sub-slots determines
                # the number of subslots
                if len(self) < len(partner):
                    self.resize(len(partner))
                elif len(self) > len(partner):
                    partner.resize(len(self))

                partner.partners.append(self)
                for i in range(len(self.partner)):
                    p = self.partner[i]
                    self[i].connect(p)

                # call slot type connect function
                self.stype.connect(partner)

                if self.level > 0 or self.stype.isConfigured():
                    self._changed()

                # call connect callbacks
                self._sig_connect(self)

                # Notify readiness after partner is updated
                if notifyReady:
                    self._sig_ready(self)

            elif partner.level < self.level:
                self.partner = partner
                notifyReady = (self.partner.meta._ready and not
                               self.meta._ready)
                self.meta = self.partner.meta.copy()
                for i, slot in enumerate(self._subSlots):
                    slot.connect(partner)

                if notifyReady:
                    self._sig_ready(self)

                self._changed()
                # call connect callbacks
                self._sig_connect(self)

            elif partner.level > self.level:
                msg = str("Can't connect slots:"
                       " {}.{}.level={}, but"
                       " {}.{}.level={}"
                       " (Implicit OpearatorWrapper creation"
                       " is no longer supported.)").format(
                           self.getRealOperator().name,
                           self.name, self.level,
                           partner.getRealOperator().name,
                           partner.name, partner.level)
                raise RuntimeError(msg)

            # propagate value changed signals from inner to outer
            # operators.
            if self._type == partner._type == "output":
                partner.notifyValueChanged(self._sig_value_changed)

    def disconnect(self):
        """
        Disconnect a InputSlot from its partner
        """
        if self.backpropagate_values:
            if self.partner is not None:
                self.partner.disconnect()
            return

        for slot in self._subSlots:
            slot.disconnect()

        had_partner = False
        if self.partner is not None:
            had_partner = True
            try:
                self.partner.partners.remove(self)
            except ValueError:
                pass
        self.partner = None
        self._value = None
        oldReady = self.meta._ready
        self.meta = MetaDict()

        if len(self._subSlots) > 0:
            self.resize(0)

        # call callbacks
        if had_partner:
            self._sig_disconnect(self)

        # Notify our partners that we changed.
        self._changed()

        # If we were ready before, signal that we aren't any more
        if oldReady:
            self._sig_unready(self)

    def resize(self, size):
        """
        Resizes a slot to the desired length

        Arguments:
          size    : the desired number of subslots
        """
        assert isinstance(size, int) or isinstance(size, long)

        if self._resizing:
            return
        if self.level == 0:
            raise RuntimeError("Can't resize a level-0 slot!")

        oldsize = len(self)
        if size == oldsize:
            return

        self._resizing = True
        if self.operator is not None:
            self.logger.debug("Resizing slot {} of operator {} to size {}".format(
                self.name, self.operator.name, size))


        # call before resize callbacks
        self._sig_resize(self, oldsize, size)

        while size > len(self):
            self.insertSlot(len(self), len(self)+1, propagate=False)
            # connect newly added slots
            self._connectSubSlot(len(self) - 1)

        while size < len(self):
            self.removeSlot(len(self)-1, len(self)-1, propagate=False)

        # propagate size change downward
        for c in self.partners:
            if c.level == self.level:
                c.resize(size)

        # propagate size change upward
        if (self.partner and len(self.partner) < size and self.partner.level == self.level):
            self.partner.resize(size)

        # call after resize callbacks
        self._sig_resized(self, oldsize, size)

        self._resizing = False



    def insertSlot(self, position, finalsize, propagate=True):
        """
        Insert a new slot at the specififed position
        finalsize indicates the final destination size
        """
        if len(self) >= finalsize:
            return self[position]

        # pre-insert callbacks
        self._sig_preinsertion(self, position, finalsize)
        
        slot =  self._insertNew(position)
        self.logger.debug("Inserting slot {} into slot {} of operator {} to size {}".format(
            position, self.name, self.operator.name, finalsize))
        if propagate:
            if self.partner is not None and self.partner.level == self.level:
                self.partner.insertSlot(position, finalsize)
            self._connectSubSlot(position)

            for p in self.partners:
                if p.level == self.level:
                    p.insertSlot(position, finalsize)

        # call after insert callbacks
        self._sig_inserted(self, position, finalsize)
        return slot

    def removeSlot(self, position, finalsize, propagate=True):
        """
        Remove the slot at position
        finalsize indicates the final size of all subslots
        """
        if len(self) <= finalsize:
            return None
        assert position < len(self)
        if self.operator is not None:
            self.logger.debug("Removing slot {} into slot {} of operator {} to size {}".format(
                position, self.name, self.operator.name, finalsize))

        # call before-remove callbacks
        self._sig_remove(self, position, finalsize)

        slot = self._subSlots.pop(position)
        slot.operator = None
        slot.disconnect()
        if propagate:
            if self.partner is not None and self.partner.level == self.level:
                self.partner.removeSlot(position, finalsize)
            for p in self.partners:
                if p.level == self.level:
                    p.removeSlot(position, finalsize)

        # call after-remove callbacks
        self._sig_removed(self, position, finalsize)

    def get(self, roi):
        """This method is used to retrieve the actual content of a Slot.

        :param roi: the region of interest, e.g. a subregion in the
        case of an ArrayLike stype

        :param destination: this may define a destination area for the
          request, for example a ndarray into which the results should
          be written in the case of an ArrayLike stype

        Returns:
          a request.Request object.

        """
        if self._value is not None:
            # this handles the case of an inputslot
            # having a ._value
            # --> construct cheaper request object for this case
            result = self.stype.writeIntoDestination(None, self._value, roi)
            return ValueRequest(result)
        elif self.partner is not None:
            # this handles the case of an inputslot
            # --> just relay the request
            return self.partner.get(roi)
        else:
            assert self.ready(), ("Can't get data from slot {}.{} yet."
                                  " It isn't ready.".format(
                                      self.getRealOperator().__class__, self.name))

            # If someone is asking for data from an inputslot that has
            #  no value and no partner, then something is wrong.
            assert self._type != "input", "This inputSlot has no value and no partner.  You can't ask for its data yet!"
            # normal (outputslot) case
            # --> construct heavy request object..
            execWrapper = Slot.RequestExecutionWrapper(self, roi)
            request = Request(execWrapper)

            # We must decrement the execution count even if the
            # request is cancelled
            request.notify_cancelled(execWrapper.handleCancel)
            return request

    class RequestExecutionWrapper(object):
        def __init__(self, slot, roi):
            self.started = False
            self.finished = False
            self.slot = slot
            self.operator = slot.operator
            self.lock = threading.Lock()
            self.roi = roi

        def __call__(self, destination=None):
            # store whether the user wants the results in a given
            # destination area
            destination_given = destination is not None

            if destination is None:
                destination = self.slot.stype.allocateDestination(self.roi)

            # We are executing the operator. Incremement the execution
            # count to protect against simultaneous setupOutputs()
            # calls.
            self._incrementOperatorExecutionCount()

            try:
                # Execute the workload, which might not ever return
                # (if we get cancelled).
                result_op = self.operator.execute(self.slot, (), self.roi, destination)

                # copy data from result_op to destination, if
                # destination was actually given by the user, and the
                # returned result_op is different from destination.
                # (but don't copy if result_op is None, this means
                # legacy op which wrote into destination anyway)
                if destination_given and result_op is not None and id(result_op) != id(destination):
                    # check that the returned value is compatible with the requested roi
                    self.slot.stype.check_result_valid(self.roi, result_op)

                    self.slot.stype.copy_data(dst=destination, src = result_op)
                elif result_op is not None:
                    # FIXME: this should be moved to a isCompatible
                    # check in stypes.py
                    if hasattr(result_op, "shape"):
                        assert result_op.shape == destination.shape, \
                          ("ERROR: Operator {} has failed to provide a"
                           " result of correct shape. result shape is"
                           " {} vs {}.  roi was {}".format(
                               self.operator, result_op.shape,
                               destination.shape, str(self.roi)))
                    destination = result_op

                    # check that the returned value is compatible with the requested roi
                    self.slot.stype.check_result_valid(self.roi, destination)


                # Decrement the execution count
                self._decrementOperatorExecutionCount()
                return destination
            except: # except Request.CancellationException
                # Decrement the execution count
                self._decrementOperatorExecutionCount()
                raise

        def _incrementOperatorExecutionCount(self):
            self.started = True
            assert self.operator._executionCount >= 0, \
                          "BUG: How did the execution count get negative?"
            # We can't execute while the operator is in the middle of
            # setupOutputs
            with self.operator._condition:
                while self.operator._settingUp:
                    self.operator._condition.wait()
                self.operator._executionCount += 1

        def handleCancel(self, *args):
            # The new request api does clean up by handling an
            # exception, not in this callback. Only clean up if we are
            # using the old request api
            using_old_api = len(args) > 0 and not hasattr(args[0], 'notify_cancelled')
            if using_old_api:
                self._decrementOperatorExecutionCount()

        def _decrementOperatorExecutionCount(self):
            # Must lock here because cancel callbacks are
            # asynchronous. (Perhaps it would be better if they were
            # called from the worker thread instead...)
            with self.lock:
                # Only do this once per execution. If we were cancelled
                # after we finished working, don't do anything
                if self.started and not self.finished:
                    assert self.operator._executionCount > 0, \
                          "BUG: Can't decrement the execution count below zero!"
                    self.finished = True
                    with self.operator._condition:
                        self.operator._executionCount -= 1
                        self.operator._condition.notifyAll()

    def setDirty(self, *args, **kwargs):
        """This method is called by a partnering OutputSlot when its
        content changes.

        The 'key' parameter identifies the changed region
        of an numpy.ndarray

        """
        assert self.operator is not None, ("Slot '{}' cannot be set dirty,"
                                           " slot not belonging to any"
                                           " actual operator instance".format(self.name))

        if self.stype.isConfigured():
            if len(args) == 0 or not isinstance(args[0], rtype.Roi):
                roi = self.rtype(self, *args, **kwargs)
            else:
                roi = args[0]

            for c in self.partners:
                c.setDirty(roi)

            # call callbacks
            self._sig_dirty(self, roi)

            if self._type == "input" and self.operator.configured():
                self.operator.propagateDirty(self, (), roi)

    def __iter__(self):
        assert self.level >= 1
        return self._subSlots.__iter__()

    def __getitem__(self, key):
        """If level=0, emulate __call__ but with a slicing instead of
        a roi.

        If level>0, return the subslot corresponding to the key, which
        may be a tuple

                          """
        if self.level > 0:
            if isinstance(key, tuple):
                assert len(key) > 0
                assert len(key) <= self.level
                if len(key) == 1:
                    return self._subSlots[key[0]]
                else:
                    return self._subSlots[key[0]][key[1:]]
            return self._subSlots[key]
        else:
            if self.meta.shape is None:
                if not self.ready():
                    msg = "This slot ({}.{}) isn't ready yet, which means " \
                          "you can't ask for its data.  Is it connected?".format(self.getRealOperator().name, self.name)
                    self.logger.error(msg)
                    slotInfoMsg = ""
                    for slot in self.getRealOperator().inputs.values():
                        if not slot.ready() and not slot._optional:
                            slotInfoMsg += "Slot '{}' isn't ready\n".format( slot.name )
                    self.logger.error(slotInfoMsg)
                    assert False, "Slot isn't ready.  See error log."
                assert self.meta.shape is not None, \
                    ("Can't ask for slices of this slot yet:"
                     " self.meta.shape is None!"
                     " (operator {} [self={}] slot: {}, key={}".format(
                         self.operator.name, self.operator, self.name, key))
            return self(pslice=key)


    def __setitem__(self, key, value):
        """This method provied access to the subslots of a
        MultiSlot.

        """
        assert not isinstance(value, Slot), \
            "Can't use setitem to connect slots.  Use connect()"
        assert self.level == 0, \
            ("setitem can only be used with slots of level 0."
             " Did you forget to append a key?")
        assert self.operator is not None, \
            "cannot do __setitem__ on Slot '{}' -> no operator !!"
        assert slicingtools.is_bounded(key), \
            "Can't use Slot.__setitem__ with keys that include : or ..."
        roi = self.rtype(self, pslice=key)
        if self._value is not None:
            self._value[key] = value

            # only propagate the dirty key at the very beginning of
            # the chain
            self.setDirty(roi)
        if self._type == "input":
            self.operator.setInSlot(self, (), roi, value)

        # Forward to partners
        for p in self.partners:
            p[key] = value

    def index(self, slot):
        return self._subSlots.index(slot)

    def setInSlot(self, slot, subindex, roi, value):
        """For now, Slots of level > 0 pretend to be operators (as far
        as their subslots are concerned). That's why they have to have
        this setInSlot() method.

        """
        # Determine which subslot this is and prepend it to the totalIndex
        totalIndex = (self._subSlots.index(slot),) + subindex
        # Forward the call to our operator
        self.operator.setInSlot(self, totalIndex, roi, value)

    def __len__(self):
        """In the case of a MultiSlot this returns the number of
        subslots, i.e. the length of the list

        """
        return len(self._subSlots)


    @property
    def value(self):
        """This method directly returns the full content of a slot.

        Is mainly used when region of interest specification make no
        sense, e.g. in the case of slots which hold a single integer
        or float value

        """
        if self.partner is not None:
            # outputslot-inputsslot, inputslot-inputslot and outputslot-outputslot case
            temp = self[:].allocate().wait()
        elif self._value is None:
            # outputslot case
            temp =  self[:].allocate().wait()
        else:
            # _value case
            return self._value
        if isinstance(temp, numpy.ndarray) and temp.shape != (1,):
            return temp
        else:
            try:
                return temp[0]
            except IndexError:
                self.logger.warn("FIXME: Slot.value for slot {} is {},"
                                 " which should be wrapped in an ndarray.".format(
                                     self.name, temp))
                return temp

    def setValue(self, value, notify=True, check_changed=True):
        """This method can be used to directly assign a value to an
        InputSlot.

        Usually a slot is either connected to another slot from which
        it retrieves the content when it is queried, or it directly
        holds a value itself. This method can be used to set such a
        value.

        If check_changed is True, the new value is compared to the
        current one and updates are onyl triggerd if they differ. This
        check can take several seconds (for instance for large
        array-like values). In that case you should turn off the
        check.

        """
        assert isinstance(notify, bool)
        assert isinstance(check_changed, bool)

        # This assertion is here to prevent accidental use of setValue
        # when connect should be used. If your use case requires
        # passing slots as values, then this assertion can be refined.
        assert not isinstance(value, Slot), \
            "When using setValue, value cannot be a slot.  Use connect instead."

        if not self.backpropagate_values:
            assert self.partner is None, \
                ("Cannot call setValue on this slot."
                 " It is already connected to a partner."
                 " Call disconnect first if that's what you really wanted.")
        elif self.partner is not None:
            self.partner.setValue(value, notify, check_changed)
            return

        changed = True
        try:
            if check_changed and value == self._value:
                changed = False
        except:
            pass
        if changed:
            #if self.stype.isCompatible(value):
            # call disconnect callbacks
            self._sig_disconnect(self)
            self._value = value
            self.stype.setupMetaForValue(value)
            self.meta._dirty = True

            for s in self._subSlots:
                s.setValue(self._value)

            notify = (self.meta._ready == False)

            # a slot with a value is always ready
            self.meta._ready = True
            if notify:
                self._sig_ready(self)

            # call connect callbacks
            self._sig_connect(self)
            self._changed()

            # Propagate dirtyness
            if self.rtype == rtype.List:
                self.setDirty(())
            else:
                self.setDirty(slice(None))

    def setValues(self, values):
        """Set values of subslots with arraylike object. Resizes the
        multinputslot with the length of the values array

        """
        # call disconnect callbacks
        self._sig_disconnect(self)
        self.resize(len(values))
        for i, s in enumerate(self._subSlots):
            s.setValue(values[i])
        # call connect callbacks
        self._changed()
        self._sig_connect(self)

    @property
    def backpropagate_values(self):
        return self._backpropagate_values

    @backpropagate_values.setter
    def backpropagate_values(self, backprop):
        self._backpropagate_values = backprop
        for slot in self._subSlots:
            slot.backpropagate_values = backprop

    def connected(self):
        """Returns True if the slot is conencted to a partner slot or
        has a _value assigned as input

        """
        answer = True
        if self._value is None and self.partner is None:
            answer = False
        if answer is False and len(self._subSlots) > 0:
            answer = True
            for s in self._subSlots:
                if s.connected() is False:
                    answer = False
                    break
        return answer

    def configured(self):
        """Slots of level >= 1 must implement parts of the operator
        interface, including this function. This "operator" is
        considered "configured" if it is ready.

        """
        return self.ready()

    def ready(self):
        if self.level == 0:
            # If this slot is non-multi, then just check our own
            # status
            ready = self.meta._ready
        else:
            # If this slot is multi, check all of our subslots. (If we
            # have no subslots, then we are NOT ready). Operators that
            # can properly handle an empty multi-input slot should
            # mark the input as optional.
            ready = len(self._subSlots) > 0
            for p in self._subSlots:
                ready &= p.ready()
        return ready

    def _setReady(self):
        wasReady = self.ready()

        for p in self._subSlots:
            p._setReady()

        self.meta._ready = (self.level == 0) or (len(self._subSlots) > 0)

        # If we just became ready...
        if not wasReady and self.meta._ready:
            # Notify partners of changed readystatus
            self._changed()
            self._sig_ready(self)

    def __call__(self, *args, **kwargs):
        """The slot relays all arguments to the __init__ method of the
        Roi type. This allows lazyflow to support different types of
        rois without knowing anything about them.

        """
        roi = self.rtype(self, *args, **kwargs)
        return self.get(roi)

    def getRealOperator(self):
        """If a slot is owned by a higher-level slot, self.operator is
        a slot. This function keeps going up the hierarchy until it
        finds the actual operator this slot belongs to.

        """
        if isinstance(self.operator, Slot):
            return self.operator.getRealOperator()
        else:
            return self.operator

    #####################
    #  Private  Methods #
    #####################
    def _getInstance(self, operator, level=None):
        """This method constructs a copy of the slot.

        This method is used when creating an Instance of an Operator.

        All defined Input and Output slots of the Class are cloned and
        inserted into the instance of the Operator.

        """
        if level is None:
            level = self.level
        if self._type == "input":
            s = InputSlot(self.name, operator, stype=self._stypeType,
                          rtype=self.rtype, value=self._defaultValue,
                          optional=self._optional, level=level,
                          nonlane=self.nonlane)
        elif self._type == "output":
            s = OutputSlot(self.name, operator, stype=self._stypeType,
                           rtype=self.rtype, value=self._defaultValue,
                           optional=self._optional, level=level,
                           nonlane=self.nonlane)
        return s

    def _changed(self):
        oldMeta = self.meta
        if self.partner is not None and self.meta != self.partner.meta:
            self.meta = self.partner.meta.copy()

        if self._type == "output":
            for o in self._subSlots:
                o._changed()

        # Notify readiness after subslots are updated
        if self.meta._ready != oldMeta._ready:
            if self.meta._ready:
                self._sig_ready(self)
            else:
                self._sig_unready(self)

        wasdirty = self.meta._dirty
        if self.meta._dirty:
            for c in self.partners:
                c._changed()
            self.meta._dirty = False

        if self._type != "output":
            op = self.getRealOperator()
            if op is not None and not op._cleaningUp:
                self._configureOperator(self)

        if wasdirty:
            # call changed callbacks
            self._sig_changed(self)

    def _configureOperator(self, slot, oldSize=0, newSize=0, notify=True):
        """Call setupOutputs of Operator if all slots of the operator
        are connected and configured.

        """
        if self.operator is not None:
            # check whether all slots are connected and notify operator
            if self.operator.configured():
                self.operator._setupOutputs()

    def _requiredLength(self):
        """
        Returns the required number of subslots
        """
        if self.partner is not None:
            if self.partner.level == self.level:
                return len(self.partner)
            elif self.partner.level < self.level:
                return 1
        elif self._value is not None:
            return 1
        else:
            return 0

    def _setupOutputs(self):
        """
        """
        self._changed()

    def _connectSubSlot(self, slot, notify=True):
        """Connect a subslot either to the partner, or set the correct
        value in case of self._value != None

        """
        if type(slot) == int:
            index = slot
            slot = self._subSlots[slot]
        else:
            index = self._subSlots.index(slot)

        if self.partner is not None:
            if self.partner.level == self.level:
                if len(self.partner) > index:
                    slot.connect(self.partner[index])
            else:
                slot.connect(self.partner)
        if self._value is not None:
            slot.setValue(self._value, notify=notify)


    def _insertNew(self, position):
        """Construct a new subSlot of correct type and level and
        insert it to the list of subslots

        """
        assert position >= 0 and position <= len(self._subSlots)
        slot = self._getInstance(self, level=self.level - 1)
        self._subSlots.insert(position, slot)
        slot.name = self.name
        if self._value is not None:
            slot.setValue(self._value)
        return slot

    def pop(self, index=-1, event=None):
        if index < 0:
            index = len(self) + index
        self._subSlots.pop(index)

    def propagateDirty(self, slot, subindex, roi):
        """Slots with level > 0 must implement part of the operator
         interface so they look like an operator as far as their
         subslots are concerned. That's why this function is here.

        """
        totalIndex = (self._subSlots.index(slot),) + subindex
        self.operator.propagateDirty(self, totalIndex, roi)


    ######################################
    # methods aimed to enhance usability #
    ######################################

    def setShapeAtAxisTo(self, axis, size):
        tmpshape = list(self.meta.shape)
        tmpshape[self.meta.axistags.index(axis)] = size
        self.meta.shape = tuple(tmpshape)

    def __str__(self):
        if self.meta.axistags is None:
            axisStr = 'Axistags \tNone\n'
        else:
            axisStr = str(self.meta.axistags)
        return self.name + '\n' + 'Shape \t\t'+str(self.meta.shape) +'\n'\
                +axisStr +\
               'Dtype \t\t' + str(self.meta.dtype)

    def __repr__(self):
        if self.meta.axistags is None:
            axisStr = 'Axistags \tNone\n'
        else:
            axisStr = str(self.meta.axistags)
        return self.name + '\n' + 'Shape \t\t'+str(self.meta.shape) +'\n'\
                + axisStr +\
               'Dtype \t\t' + str(self.meta.dtype)

class InputSlot(Slot):
    """The base class for input slots, it provides methods to connect
    the InputSlot to an OutputSlot of another operator (i.e.
    .connect(partner) call) or allows to directly provide a value as
    input (i.e. .setValue(value) call)

    """

    def __init__(self, *args, **kwargs):
        super(InputSlot, self).__init__(*args, **kwargs)
        self._type = "input"
        # configure operator in case of slot change
        self.notifyResized(self._configureOperator)


class OutputSlot(Slot):
    """The base class for output slots, it provides methods to connect
    the OutputSlot to an InputSlot of another operator (i.e.
    .connect(partner) call).

    the content of the OutputSlot e.g. the result of the operator it
    belongs to can be requested with the usual python array slicing
    syntax, i.e.

    outputslot[3,:,14:32]

    This call returns an GetItemRequestObject.

    """


    def __init__(self, *args, **kwargs):
        super(OutputSlot, self).__init__(*args, **kwargs)
        self._type = "output"

    def execute(self, slot, subindex, roi, result):
        """For now, OutputSlots with level > 0 must pretend to be
        operators. That's why this function is here.

        """
        totalIndex = (self._subSlots.index(slot),) + subindex
        return self.operator.execute(self, totalIndex, roi, result)

class InputDict(collections.OrderedDict):

    def __init__(self, operator):
        super(InputDict, self).__init__()
        self.operator = operator


    def __setitem__(self, key, value):
        assert isinstance(value, InputSlot), \
            ("ERROR: all elements of .inputs must be of type InputSlot."
             " You provided {}!".format(value))
        return super(InputDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        if self.has_key(key):
            return super(InputDict, self).__getitem__(key)
        elif hasattr(self.operator, key):
            return getattr(self.operator, key)
        else:
            raise Exception("Operator {} (class: {}) has no input slot named '{}'."
                            " Available input slots are: {}".format(
                                self.operator.name, self.operator.__class__, key, self.keys()))



class OutputDict(collections.OrderedDict):

    def __init__(self, operator):
        super(OutputDict, self).__init__()
        self.operator = operator


    def __setitem__(self, key, value):
        assert isinstance(value, OutputSlot), \
            ("ERROR: all elements of .outputs must be of type"
             " OutputSlot. You provided {}!".format(value))
        return super(OutputDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        if self.has_key(key):
            return super(OutputDict, self).__getitem__(key)
        elif hasattr(self.operator, key):
            return getattr(self.operator, key)
        else:
            raise Exception("Operator {} (class: {}) has no output slot named '{}'."
                            " Available output slots are: {}".format(
                                self.operator.name, self.operator.__class__, key, self.keys()))


from abc import ABCMeta
class OperatorMetaClass(ABCMeta):

    def __new__(cls, name, bases, classDict):
        cls = super(OperatorMetaClass, cls).__new__(cls, name, bases, classDict)

        setattr(cls, "inputSlots", list(cls.inputSlots))
        setattr(cls, "outputSlots", list(cls.outputSlots))

        # Support fancy syntax.
        # If the user typed this in his class definition:
        #    MySlot = InputSlot()
        #    MySlot2 = InputSlot()
        #
        # Make it equivalent to this:
        #    inputSlots = [ InputSlot("MySlot"), InputSlot("MySlot2") ]

        for k, v in cls.__dict__.items():
            if isinstance(v, InputSlot):
                v.name = k
                cls.inputSlots.append(v)

            if isinstance(v, OutputSlot):
                v.name = k
                cls.outputSlots.append(v)
        return cls

    def __call__(cls, *args, **kwargs):
        # type.__call__ calls instance.__init__ internally
        try:
            instance = ABCMeta.__call__(cls, *args, **kwargs)
        except Exception as e:
            err = "Could not create instance of '{}'\n".format(cls)
            err += "args   = {}\n".format(args)
            err += "kwargs = {}\n".format(kwargs)
            err += "The exception was:\n"
            err += str(e)
            err += "\nTraceback:\n"
            import traceback
            import StringIO
            s = StringIO.StringIO()
            traceback.print_exc(file=s)
            err += s.getvalue()
            raise RuntimeError(err)
        instance._after_init()
        return instance



class Operator(object):
    """
    The base class for all Operators.

    Operators consist of a class inheriting from this class
    and need to specify their inputs and outputs via
    thei inputSlot and outputSlot class properties.

    Each instance of an operator obtains individual
    copies of the inputSlots and outputSlots, which are
    available in the self.inputs and self.outputs instance
    properties.

    these instance properties can be used to connect
    the inputs and outputs of different operators.

    Example:
        operator1.inputs["InputA"].connect(operator2.outputs["OutputC"])


    Different examples for simple operators are provided
    in an example directory. plese read through the
    examples to learn how to implement your own operators...
    """

    loggerName = __name__ + '.Operator'
    logger = logging.getLogger(loggerName)
    traceLogger = logging.getLogger('TRACE.' + loggerName)

    #definition of inputs slots
    inputSlots  = []

    #definition of output slots -> operators instances
    outputSlots = []
    name = "Operator (base class)"
    description = ""
    category = "lazyflow"

    __metaclass__ = OperatorMetaClass

    def __new__(cls, *args, **kwargs):
        ##
        # before __init__
        ##
        obj = object.__new__(cls)
        obj.inputs = InputDict(obj)
        obj.outputs = OutputDict(obj)
        return obj

    def __init__(self, parent=None, graph=None):
        if not(parent is None or isinstance(parent, Operator)):
            raise Exception("parent of operator name='{}' must be an operator,"
                            " not {} of type {}".format(self.name, parent, type(parent)))
        if graph is None:
            if parent is None:
                raise Exception("Operator.__init__() [self.name='{}']:"
                                " parent and graph can't be both None".format(self.name))
            graph = parent.graph

        self._cleaningUp = False
        self.graph = graph
        self._children = collections.OrderedDict()
        self._parent = None
        if parent is not None:
            parent._add_child(self)

        self._initialized = False

        self._condition = threading.Condition()
        self._executionCount = 0
        self._settingUp = False

        self._instantiate_slots()

        # We normally assert if an operator's upstream partners are
        # yanked away. If operator is marked as "externally_managed",
        # then we'll avoid the assert. In that case, it's assumed that
        # you know what you're doing, and you weren't planning to use
        # that operator, anyway.
        self.externally_managed = False

    @property
    def children(self):
        return list(self._children.keys())

    def _add_child(self, child):
        # We're just using an OrderedDict for O(1) lookup with
        # in-order iteration but we don't actually store any values
        assert child.parent is None
        self._children[child] = None
        child._parent = self

    # continue initialization, when user overrides __init__
    def _after_init(self):
        #provide simple default name for lazy users
        if self.name == Operator.name:
            self.name = type(self).__name__
        assert self.graph is not None, \
            ("Operator {}: self.graph is None, the parent ({})"
             " given to the operator must have a valid .graph attribute!".format(
                 self, self._parent))
        # check for slot uniqueness
        temp = {}
        for i in self.inputSlots:
            if temp.has_key(i.name):
                raise Exception("ERROR: Operator {} has multiple slots with name {},"
                                " please make sure that all input and output slot"
                                " names are unique".format(self.name, i.name))
            temp[i.name] = True

        for i in self.outputSlots:
            if temp.has_key(i.name):
                raise Exception("ERROR: Operator {} has multiple slots with name {},"
                                " please make sure that all input and output slot"
                                " names are unique".format(self.name, i.name))
            temp[i.name] = True

        self._instantiate_slots()

        self._setDefaultInputValues()

        for islot in self.inputs.values():
            islot.notifyUnready(self.handleInputBecameUnready)


        self._initialized = True
        if self.configured():
            self._setupOutputs()

    def _instantiate_slots(self):
        # replicate input slot connections
        # defined for the operator for the instance
        for i in sorted(self.inputSlots, key=lambda s: s._global_slot_id):
            if not self.inputs.has_key(i.name):
                ii = i._getInstance(self)
                ii.connect(i.partner)
                self.inputs[i.name] = ii

        for k, v in self.inputs.items():
            self.__dict__[k] = v

        # relicate output slots
        # defined for the operator for the instance
        for o in sorted(self.outputSlots, key=lambda s: s._global_slot_id):
            if not self.outputs.has_key(o.name):
                oo = o._getInstance(self)
                self.outputs[o.name] = oo

        for k, v in self.outputs.items():
            self.__dict__[k] = v

    @property
    def parent(self):
        return self._parent

    def __setattr__(self, name, value):
        """This method safeguards that operators do not overwrite slot
        names with custom instance attributes.

        """
        if self.__dict__.has_key("inputs") and self.__dict__.has_key("outputs"):
            if self.inputs.has_key(name) or self.outputs.has_key(name):
                assert isinstance(value, Slot), \
                    ("ERROR: trying to set attribute {} of operator {}"
                     " to value {}, which is not of type Slot !".format(name, self, value))
        object.__setattr__(self, name, value)

    def configured(self):
        """Returns True if all input slots that are non-optional are
        connected and configured.

        """
        allConfigured = self._initialized
        for slot in self.inputs.values():
            allConfigured &= (slot.ready() or slot._optional)
        return allConfigured

    def _setDefaultInputValues(self):
        for i in self.inputs.values():
            if (i.partner is None and
                i._value is None and
                i._defaultValue is not None):
                i.setValue(i._defaultValue)

    def _disconnect(self):
        """Disconnect our slots from their upstream partners (not
        their downstream ones) and recursively do the same to all our
        child operators.

        """
        for s in self.inputs.values() + self.outputs.values():
            s.disconnect()

        for child in self._children.keys():
            child._disconnect()

    def _initCleanup(self):
        self._cleaningUp = True
        for child in self._children.keys():
            child._initCleanup()

    def cleanUp(self):
        if not self._cleaningUp:
            self._initCleanup()
        if self._parent is not None:
            del self._parent._children[self]

        # Disconnect ourselves and all children
        self._disconnect()

        for s in self.inputs.values() + self.outputs.values():
            # See note about the externally_managed flag in Operator.__init__
            partners = list(p for p in s.partners
                            if p.getRealOperator() is not None and
                            not p.getRealOperator().externally_managed)
            if len(partners) > 0:
                msg = ("Cannot clean up this operator: Slot '{}'"
                       " is still providing data to downstream"
                       " operators!\n".format(s.name))
                for i, p in enumerate(s.partners):
                    msg += "Downstream Partner {}: {}.{}".format(
                        i, p.getRealOperator().name, p.name)
                raise RuntimeError(msg)

        # Work with a copy of the child list
        # (since it will be modified with each iteration)
        children = set(self._children.keys())
        for child in children:
            child.cleanUp()

    def propagateDirty(self, slot, subindex, roi):
        """This method is called when an output of another operator on
        which this operators depends, i.e. to which it is connected gets
        invalid. The region of interest of the inputslot which is now
        dirty is specified in the key property, the input slot which got
        dirty is specified in the inputSlot property.

        This method must calculate what output ports and which subregions
        of them are invalidated by this, and must call the .setDirty(key)
        of the corresponding outputslots.

        """
        raise NotImplementedError(".propagateDirty() of Operator {}"
                                  " is not implemented !".format(self.name))

    @staticmethod
    def forbidParallelExecute(func):
        """Use this decorator with functions that must not be run in
        parallel with the execute() function.

        - Your function won't start until no threads are in execute().

        - Calls to execute() will wait until your function is complete.

        This is better than using a simple lock in execute() because
        it allows execute() to be run in parallel with itself.

        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with self._condition:
                while self._executionCount > 0:
                    self._condition.wait()
                self._settingUp = True

                try:
                    return func(self, *args, **kwargs)
                finally:
                    self._settingUp = False
                    self._condition.notifyAll()
        wrapper.__wrapped__ = func # Emulate python 3 behavior of @wraps
        return wrapper

    def _setupOutputs(self):
        with Tracer(self.traceLogger, msg=self.name):
            # Don't setup this operator if there are currently
            # requests on it.
            with self._condition:
                while self._executionCount > 0:
                    self._condition.wait()
                self._settingUp = True

                # Outputslots may become "ready" during setupOutputs().
                # Save a copy of the ready flag for each output slot
                # so we can decide whether or not to fire the ready
                # signal.
                readyFlags = {}
                for k, oslot in self.outputs.items():
                    readyFlags[k] = oslot.meta._ready

                # Call the subclass
                self.setupOutputs()

                self._settingUp = False
                self._condition.notifyAll()

            # Determine new "ready" flags
            for k, oslot in self.outputs.items():
                if oslot.partner is None:
                    # All unconnected outputs are ready after
                    # setupOutputs
                    oslot._setReady()

            #notify outputs of probably changed meta information
            for k, v in self.outputs.items():
                v._changed()

    def handleInputBecameUnready(self, slot):
        with Tracer(self.traceLogger, msg=self.name):
            # One of our input slots was disconnected.
            # If it was optional, we don't care.
            if slot._optional:
                return

            # Keep track of the old ready statuses so we know if
            # something changed
            readyFlags = {}
            for k, oslot in self.outputs.items():
                readyFlags[k] = oslot.meta._ready

            # All unconnected outputs are no longer ready
            for oslot in self.outputs.values():
                oslot.meta._ready &= (oslot.partner is not None)

            # If the ready status changed, signal it.
            for k, oslot in self.outputs.items():
                if readyFlags[k] != oslot.meta._ready:
                    oslot._sig_unready(self)
                    oslot._changed()

    def setupOutputs(self):
        """This method is called when all input slots of an operator
        are successfully connected, a successful connection is also
        established if the input slot is not connected to another
        slot, but has a default value defined.

        In this method the operator developer should stup
        the .meta information of the outputslots.

        The default implementation emulates the old api behaviour.

        """
        pass

    def execute(self, slot, subindex, roi, result):
        """ This method of the operator is called when a connected
        operator or an outside user of the graph wants to retrieve the
        calculation results from the operator.

        The slot which is requested is specified in the slot arguemt,
        the region of interest is specified in the key property. The
        result area into which the calculation results MUST be written
        is specified in the result argument. "result" is an
        numpy.ndarray that has the same shape as the region of
        interest(key).

        The method must retrieve all required inputs that are
        neccessary to calculate the requested output area from its
        input slots, run the calculation and put the results into the
        provided result argument. """

        raise NotImplementedError("Operator {} does not implement"
                                  " execute()".format(self.name))

    def setInSlot(self, slot, subindex, key, value):
        raise NotImplementedError("Can't use __setitem__ with Operator {}"
                                  " because it doesn't implement"
                                  " setInSlot()".format(self.name))

class OperatorWrapper(Operator):
    name = "OperatorWrapper"

    loggerName = __name__ + '.OperatorWrapper'
    logger = logging.getLogger(loggerName)
    traceLogger = logging.getLogger('TRACE.' + loggerName)

    def __init__(self, operatorClass, operator_args=None,
                 operator_kwargs=None, parent=None, graph=None,
                 promotedSlotNames=None, broadcastingSlotNames=None):
        """Constructs a wrapper for the given operator. That is,
        manages a list of copies of the original operator, and
        provides access to these inner operators' slots via external
        multislots.

        :param operatorClass: An operator type that can be constructed
          with the given args and kwargs

        :param operator_args: Positional arguments for operator's
          constructor. Note: Do not include 'parent' and 'graph'
          arguments.

        :param operator_kwargs: Keyword arguments for the operator's
          constructor. Note: Do not include 'parent' and 'graph'
          arguments

        :param parent: The parent of the OperatorWrapper

        :param graph: the graph operator to init each inner operator with

        :param promotedSlotNames:

          If provided, only those slots will be promoted when
            replicated. All other slots will be replicated without
            promotion, and their input values will be broadcasted to
            all inner operators.

          If not provided (i.e. promotedSlotNames=None), the default
            behavior is to promote ALL replicated slots.

          Note: Outputslots are always promoted, regardless of whether
            or not they appear in the promotedSlotNames argument.

        """
        # Base class init
        super(OperatorWrapper, self).__init__(parent=parent, graph=graph)
        if operator_args == None:
            operator_args = ()
        if operator_kwargs == None:
            operator_kwargs = {}
        assert isinstance(operator_args, (tuple, list))
        assert isinstance(operator_kwargs, dict)
        self._createInnerOperator = functools.partial(
            operatorClass, parent=self,
            *operator_args, **operator_kwargs)

        self._initialized = False

        if operatorClass.name == "":
            self._name = "Wrapped " + operatorClass.__name__
        else:
            self._name = "Wrapped " + operatorClass.name

        self._customName = False

        if promotedSlotNames is not None:
            assert broadcastingSlotNames is None, \
                ("Please specify either the promoted slots or the"
                 " broadcasting slots, not both.")
            # 'Promoted' slots will be exposed as multi-slots
            # All others will be broadcasted
            promotedSlotNames = set(promotedSlotNames)

        elif broadcastingSlotNames is not None:
            # 'Broadcasting' slots are NOT exposed as multi-slots.
            # Each is exposed as a single slot that is shared by all
            # inner operators.
            allInputSlotNames = set(map(lambda s: s.name, operatorClass.inputSlots))

            # set difference
            promotedSlotNames = allInputSlotNames - set(broadcastingSlotNames)

        else:
            # No slots specified: All original slots are promoted by
            # default
            promotedSlotNames = set(slot.name for slot in operatorClass.inputSlots)

        # All Outputs are always promoted
        promotedSlotNames |= set(slot.name for slot in operatorClass.outputSlots)

        self.promotedSlotNames = promotedSlotNames

        self.innerOperators = []
        self.logger.log(logging.DEBUG,
                        "wrapping operator '{}'".format(
                            operatorClass.name))

        # replicate input slot definitions
        for innerSlot in sorted(operatorClass.inputSlots,
                                key=lambda s: s._global_slot_id):
            level = innerSlot.level
            if innerSlot.name in self.promotedSlotNames:
                level += 1
            outerSlot = innerSlot._getInstance(self, level=level)
            self.inputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # replicate output slot definitions
        for innerSlot in sorted(operatorClass.outputSlots,
                                key=lambda s: s._global_slot_id):
            level = innerSlot.level + 1
            outerSlot = innerSlot._getInstance(self, level=level)
            self.outputs[outerSlot.name] = outerSlot
            setattr(self, outerSlot.name, outerSlot)

        # register callbacks for inserted and removed input subslots
        for s in self.inputs.values():
            if s.name in self.promotedSlotNames:
                s.notifyInserted(self._callbackInserted)
                s.notifyRemoved(self._callbackRemove)
                s._notifyConnect(self._callbackConnect)

        # register callbacks for inserted and removed output subslots
        for s in self.outputs.values():
            s.notifyInserted(self._callbackInserted)
            s.notifyRemoved(self._callbackRemove)

        for s in self.inputs.values():
            assert len(s) == 0
        for s in self.outputs.values():
            assert len(s) == 0

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        self._customName = True

    def __getitem__(self, key):
        return self.innerOperators[key]

    def __len__(self):
        return len(self.innerOperators)

    def _callbackInserted(self, slot, index, size):
        with Tracer(self.traceLogger, msg=self.name):
            self._insertInnerOperator(index, size)

    def _callbackRemove(self, slot, index, size):
        with Tracer(self.traceLogger, msg=self.name):
            self._removeInnerOperator(index, size)

    def _callbackConnect(self, slot):
        with Tracer(self.traceLogger, msg=self.name):
            slot.resize(len(self.innerOperators))
            for index, innerOp in enumerate(self.innerOperators):
                innerOp.inputs[slot.name].connect(slot[index])

    def propagateDirty(self, slot, subindex, roi):
        # Nothing to do: All inputs are directly connected to internal
        # operators.
        pass

    def _insertInnerOperator(self, index, length):
        with Tracer(self.traceLogger, msg=self.name):
            if len(self.innerOperators) >= length:
                return self.innerOperators[index]
            op = self._createInnerOperator()

            # Update our name (if the client didn't already give us a
            # special one)
            if self._customName is False:
                self._name = "Wrapped " + op.name

            # If anyone calls setValue() on one of these slots,
            # forward the setValue call to the slot's partner (the
            # outer slot on the operator wrapper)
            for slot in op.inputs.values():
                slot.backpropagate_values = True
                slot.notifyDisconnect(self.handleEarlyDisconnect)

            self.innerOperators.insert(index, op)

            # Connect the inner operator's inputs to our outer input
            # slots
            for key, outerSlot in self.inputs.items():
                # Only connect to a subslot if it was promoted during
                # wrapping
                if outerSlot.name in self.promotedSlotNames:
                    outerSlot.insertSlot(index, length)
                    partner = outerSlot[index]
                else:
                    partner = outerSlot
                if op.inputs[key].partner is not None:
                    msg = ("Can't set up OperatorWrapper connections."
                           " Input slot {} is already connected to a"
                           " partner (must have happened in {}'s"
                           " constructor".format(key, op.name))
                    raise RuntimeError(msg)
                op.inputs[key].connect(partner)

            # Connect our outer output slots to the inner operator's output slots.
            for key, mslot in self.outputs.items():
                mslot.insertSlot(index, length)
                mslot[index].backpropagate_values = True
                mslot[index].connect(op.outputs[key])
                mslot[index].notifyDisconnect(self.handleEarlyDisconnect)
                #mslot[index]._changed()
            return op

    def handleEarlyDisconnect(self, slot):
        assert False, \
            ("You aren't allowed to disconnect the internal"
             " connections of an operator wrapper.")

    def _removeInnerOperator(self, index, length):
        with Tracer(self.traceLogger, msg=self.name):
            if len(self.innerOperators) <= length:
                return
            assert index < len(self.innerOperators)

            for key, mslot in self.outputs.items():
                if len(mslot) > length:
                    mslot[index].backpropagate_values = False
                    mslot[index].unregisterDisconnect(self.handleEarlyDisconnect)

            op = self.innerOperators.pop(index)
            for slot in op.inputs.values():
                slot.backpropagate_values = False
                slot.unregisterDisconnect(self.handleEarlyDisconnect)

            for oslot in self.outputs.values():
                oslot.removeSlot(index, length)

            for islot in self.inputs.values():
                islot.removeSlot(index, length)

            op.cleanUp()
            length = len(self.innerOperators)

    def _setupOutputs(self):
        with Tracer(self.traceLogger, msg=self.name):
            for oslot in self.outputs.values():
                oslot._changed()

    def execute(self, slot, subindex, roi, result):
        #this should never be called !!!
        assert False, \
            "OperatorWrapper execute() function should never be called.  "\
            "You can only ask for data from SUBslots, not the outer multi-slots themselves."

    def setInSlot(self, slot, subindex, key, value):
        # Nothing to do here. Calls to Slot.setitem are already
        # forwarded to all slot partners.
        pass

class Graph(object):
    def stopGraph(self):
        pass

    def finalize(self):
        pass

    def resumeGraph(self):
        pass

    def _registerCache(self, cache):
        pass

    def _notifyMemoryHit(self, *args, **kwargs):
        pass

    def _notifyMemoryAllocation(self, *args, **kwargs):
        pass

    def _notifyFreeMemory(self, *args, **kwargs):
        pass

# singleton graph class, that
# serves as parent graph for all operators
# wich are created without parent
class GlobalGraph(Graph):
    __metaclass__ = lazyflow.utility.Singleton
