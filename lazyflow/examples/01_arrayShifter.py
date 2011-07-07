"""

"""
import vigra
import threading
from lazyflow.graph import *
import copy

from lazyflow.operators.operators import OpArrayPiper 
from lazyflow.operators.vigraOperators import *
from lazyflow.operators.valueProviders import *
from lazyflow.operators.classifierOperators import *
from lazyflow.operators.generic import *




class OpArrayShifter(Operator):
    name = "ArrayPiper"
    description = "simple shifting operator"
    shift = 50   
    inputSlots = [InputSlot("Input")]
    outputSlots = [OutputSlot("Output")]    
    
    def notifyConnectAll(self):
        inputSlot = self.inputs["Input"]
        self.outputs["Output"]._dtype = inputSlot.dtype
        self.outputs["Output"]._shape = inputSlot.shape
        self.outputs["Output"]._axistags = copy.copy(inputSlot.axistags)

    def getOutSlot(self, slot, key, result):
        
        shape =  self.inputs["Input"].shape     
        
        #get N-D coordinate out of slice
        ostart, ostop = sliceToRoi(key, shape)    
        #copy original array        
        rstart, rstop = ostart.copy(), ostop.copy()
      
        #shift the reading scope 
        rstart[-2] +=  self.shift
        rstop[-2]  +=  self.shift
         
        #shifted rstart/rstop has to be in the original range for shifts in both directions
        rstart = numpy.minimum(rstart,numpy.array(shape))
        rstart = numpy.maximum(rstart, rstart - rstart)
        rstop  = numpy.minimum(rstop,numpy.array(shape))
        rstop = numpy.maximum(rstop, rstop-rstop)
        
        #create slice out of the reading start and stop coordinates                
        rkey = roiToSlice(rstart,rstop)       
        
        #calculate writing coordinates
        wstart =  rstart - ostart
        wstop  =  rstop - ostart
        
        #
        #wstart = numpy.minimum(wstart,numpy.array(result.shape))
        #wstart = numpy.maximum(wstart, wstart - wstart)
        #wstop  = numpy.minimum(wstop,numpy.array(result.shape))
        #wstop = numpy.maximum(wstop, wstop-wstop)       
        
        #create slice out of the reading start and stop coordinates                
        wkey = roiToSlice(wstart,wstop)
        
        #prefill result array with 0's
        result[:] = 0
        
        req = self.inputs["Input"][rkey].writeInto(result[wkey])
        res = req()
        return res

    def notifyDirty(self,slot,key):
        self.outputs["Output"].setDirty(key)

    @property
    def shape(self):
        return self.outputs["Output"]._shape
    
    @property
    def dtype(self):
        return self.outputs["Output"]._dtype







g = Graph(numThreads = 1, softMaxMem = 2000*1024**2)

        
vimageReader = OpImageReader(g)
vimageReader.inputs["Filename"].setValue("/net/gorgonzola/storage/cripp/lazyflow/tests/ostrich.jpg")

shifter = OpArrayShifter(g)
shifter.inputs["Input"].connect(vimageReader.outputs["Image"])

shifter.outputs["Output"][:].allocate().wait()

vimageWriter = OpImageWriter(g)
vimageWriter.inputs["Filename"].setValue("/net/gorgonzola/storage/cripp/lazyflow/lazyflow/examples/01-shift_resultpp50.jpg")
vimageWriter.inputs["Image"].connect(shifter.outputs["Output"])


g.finalize()