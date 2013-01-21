from lazyflow.graph import Operator,InputSlot,OutputSlot
from lazyflow.utility.helpers import newIterator
from lazyflow.request import Pool,Request
from lazyflow.rtype import SubRegion
import numpy
import vigra
from math import sqrt
from functools import partial
from lazyflow.roi import roiToSlice,sliceToRoi
import collections
import warnings

class OpBaseVigraFilter(Operator):
    
    inputSlots = []
    outputSlots = [OutputSlot("Output")]
    
    name = 'OpBaseVigraFilter'
    
    vigraFilter = None
    windowSize = 4
    
    def __init__(self, *args, **kwargs):
        super(OpBaseVigraFilter, self).__init__(*args, **kwargs)
        self.iterator = None
    
    def getChannelResolution(self):
        """
        returns the number of source channels which get mapped on one result
        channel
        """
        return 1
     
    def resultingChannels(self):
        """
        returns the resulting channels
        """
        pass
    
    def setupFilter(self):
        """
        setups the vigra filter and returns the maximum sigma
        """
        pass
    
    def calculateHalo(self,sigma):
        """
        calculates halo, depends on the filter
        """
        if isinstance(sigma, collections.Iterable):
            halos = [2*numpy.ceil(s*self.windowSize)+1 for s in sigma]
            return tuple(halos)
        else:
            return 2*numpy.ceil(sigma*self.windowSize)+1
    
    def propagateDirty(self,slot,subindex,roi):
        if slot == self.Input:
            cIndex = self.Input.meta.axistags.channelIndex
            retRoi = roi.copy()
            retRoi.start[cIndex] *= self.channelsPerChannel()
            retRoi.stop[cIndex] *= self.channelsPerChannel()
            self.Output.setDirty(retRoi)
    
    def setupIterator(self,source,result):
        self.iterator = AxisIterator(source,'spatialc',result,'spatialc',[(),(1,1,1,1,self.resultingChannels())])
    
    def setupOutputs(self):
        inputSlot = self.Input
        outputSlot = self.Output
        channelNum = self.resultingChannels()
        outputSlot.meta.assignFrom(inputSlot.meta)
        outputSlot.setShapeAtAxisTo('c', channelNum)
        
    def execute(self, slot, subindex, roi, result):
        #request,set or compute the necessary parameters
        axistags = self.Input.meta.axistags
        inputShape  = self.Input.meta.shape
        channelIndex = axistags.index('c')
        channelsPerC = self.channelsPerChannel()
        channelRes = self.getChannelResolution()
        timeIndex = axistags.index('t')
        if timeIndex >= roi.dim:
            timeIndex = None
        roi.setInputShape(inputShape)
        origRoi = roi.copy()
        sigma = self.setupFilter()
        halo = self.calculateHalo(sigma)
        
        #set up the roi to get the necessary source
        roi.expandByShape(halo,channelIndex,timeIndex).adjustChannel(channelsPerC,channelIndex,channelRes)
        print roi,self.name
        source = self.Input(roi.start,roi.stop).wait()
        source = vigra.VigraArray(source,axistags=axistags)
        
        #set up the grid for the iterator, and the iterator
        srcGrid = [source.shape[i] if i!= channelIndex else channelRes for i in range(len(source.shape))]
        trgtGrid = [inputShape[i]  if i != channelIndex else self.channelsPerChannel() for i in range(len(source.shape))]
        if timeIndex is not None:
            srcGrid[timeIndex] = 1
            trgtGrid[timeIndex] = 1
        nIt = newIterator(origRoi,srcGrid,trgtGrid,timeIndex=timeIndex,channelIndex = channelIndex)
        
        #set up roi to work with vigra filters
        if timeIndex > channelIndex and timeIndex is not None:
            origRoi.popDim(timeIndex)
            origRoi.popDim(channelIndex)
        elif timeIndex < channelIndex and timeIndex is not None:
            origRoi.popDim(channelIndex)
            origRoi.popDim(timeIndex)
        else:
            origRoi.popDim(channelIndex)
        origRoi.adjustRoi(halo)
        
        #iterate over the requested volumes
        pool = Pool()
        for src,trgt,mask in nIt:
            req = Request(partial(result.__setitem__,trgt,self.vigraFilter(source = source[src],window_size=self.windowSize,roi=origRoi)[mask]))
            pool.add(req)
        pool.wait()
        return result
    
class OpGaussianSmoothing(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"),InputSlot("Sigma")]
    name = "GaussianSmoothing"
    
    def __init__(self, *args, **kwargs):
        super(OpGaussianSmoothing, self).__init__(*args, **kwargs)
        
    def setupIterator(self,source,result):
        self.iterator = AxisIterator(source,'spatialc',result,'spatialc',[(),({'c':self.channelsPerChannel()})])   
    
    def setupFilter(self):
        sigma = self.inputs["Sigma"].value
        def tmpFilter(source,sigma,window_size,roi):
            tmpfilter = vigra.filters.gaussianSmoothing
            return tmpfilter(array=source,sigma=sigma,window_size=window_size,roi=(roi.start,roi.stop))
    
        self.vigraFilter = partial(tmpFilter,sigma=sigma,window_size=self.windowSize)
        return sigma
        
    def resultingChannels(self):
        return self.Input.meta.shape[self.Input.meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1
    
class OpDifferenceOfGaussians(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float"), InputSlot("Sigma2", stype = "float")]
    name = "DifferenceOfGaussians"
    
    def __init__(self, *args, **kwargs):
        super(OpDifferenceOfGaussians, self).__init__(*args, **kwargs)
        
    def setupFilter(self):
        sigma0 = self.inputs["Sigma"].value
        sigma1 = self.inputs["Sigma2"].value
        
        def tmpFilter(s0,s1,source,window_size,roi):
            tmpfilter = vigra.filters.gaussianSmoothing
            return tmpfilter(source,s0,window_size=window_size,roi=(roi.start,roi.stop))-tmpfilter(source,s1,window_size=window_size,roi=(roi.start,roi.stop))
        
        self.vigraFilter = partial(tmpFilter,s0=sigma0,s1=sigma1,window_size=self.windowSize)
        return max(sigma0,sigma1)
    
    def resultingChannels(self):
        return self.Input.meta.shape[self.Input.meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1
        
class OpHessianOfGaussian(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"),InputSlot("Sigma")]
    name = "OpHessianOfGaussian"
    
    def __init__(self, *args, **kwargs):
        super(OpHessianOfGaussian, self).__init__(*args, **kwargs)
        
    def setupIterator(self,source,result):
        self.iterator = AxisIterator(source,'spatial',result,'spatial',[(),({'c':self.resultingChannels()})])   
    
    def setupFilter(self):
        sigma = self.inputs["Sigma"].value
        
        def tmpFilter(source,sigma,window_size,roi):
            tmpfilter = vigra.filters.hessianOfGaussian
            return tmpfilter(source,sigma=sigma,window_size=window_size,roi=(roi.start,roi.stop))
            
        self.vigraFilter = partial(tmpFilter,sigma=sigma,window_size=self.windowSize)
        return sigma
        
    def resultingChannels(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)*(self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
    
    def channelsPerChannel(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)*(self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
    
class OpLaplacianOfGaussian(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float")]
    name = "LaplacianOfGaussian"
    
    def __init__(self, *args, **kwargs):
        super(OpLaplacianOfGaussian, self).__init__(*args, **kwargs)
        
    def setupFilter(self):
        scale = self.inputs["Sigma"].value
        
        def tmpFilter(source,scale,window_size,roi):
            tmpfilter = vigra.filters.laplacianOfGaussian
            return tmpfilter(array=source,scale=scale,window_size=window_size,roi=(roi.start,roi.stop))

        self.vigraFilter = partial(tmpFilter,scale=scale,window_size=self.windowSize)
        return scale
    
    def resultingChannels(self):
        return self.Input.meta.shape[self.Input.meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1

class OpStructureTensorEigenvaluesSummedChannels(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float"),InputSlot("Sigma2", stype = "float")]
    name = "StructureTensorEigenvalues"
    
    def __init__(self, *args, **kwargs):
        super(OpStructureTensorEigenvaluesSummedChannels, self).__init__(*args, **kwargs)
    
    def getChannelResolution(self):
        return self.Input.meta.shape[self.Input.meta.axistags.channelIndex]
    
    def calculateHalo(self, sigma):
        sigma1 = self.Sigma.value
        sigma2 = self.Sigma2.value
        return int(numpy.ceil(sigma1*self.windowSize))+int(numpy.ceil(sigma2*self.windowSize))
        
    def setupFilter(self):
        innerScale = self.Sigma.value
        outerScale = self.inputs["Sigma2"].value
        
        def tmpFilter(source,innerScale,outerScale,window_size,roi):
            tmpfilter = vigra.filters.structureTensorEigenvalues
            return tmpfilter(image=source,innerScale=innerScale,outerScale=outerScale,window_size=window_size,roi=(roi.start,roi.stop))

        self.vigraFilter = partial(tmpFilter,innerScale=innerScale,outerScale=outerScale,window_size=self.windowSize)

        return max(innerScale,outerScale)
    
    def setupIterator(self, source, result):
        self.iterator = AxisIterator(source,'spatial',result,'spatial',[(),({'c':self.channelsPerChannel()})])   
        
    def resultingChannels(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)
    
    def channelsPerChannel(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)
    
class OpStructureTensorEigenvalues(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float"),InputSlot("Sigma2", stype = "float")]
    name = "StructureTensorEigenvalues"
    
    def __init__(self, *args, **kwargs):
        super(OpStructureTensorEigenvalues, self).__init__(*args, **kwargs)
    
    def calculateHalo(self, sigma):
        sigma1 = self.Sigma.value
        sigma2 = self.Sigma2.value
        return int(numpy.ceil(sigma1*self.windowSize))+int(numpy.ceil(sigma2*self.windowSize))
        
    def setupFilter(self):
        innerScale = self.Sigma.value
        outerScale = self.inputs["Sigma2"].value
        
        def tmpFilter(source,innerScale,outerScale,window_size,roi):
            tmpfilter = vigra.filters.structureTensorEigenvalues
            return tmpfilter(image=source,innerScale=innerScale,outerScale=outerScale,window_size=window_size,roi=(roi.start,roi.stop))

        self.vigraFilter = partial(tmpFilter,innerScale=innerScale,outerScale=outerScale,window_size=self.windowSize)

        return max(innerScale,outerScale)
    
    def setupIterator(self, source, result):
        self.iterator = AxisIterator(source,'spatial',result,'spatial',[(),({'c':self.channelsPerChannel()})])   
        
    def resultingChannels(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)*self.Input.meta.shape[self.Input.meta.axistags.channelIndex]
    
    def channelsPerChannel(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)


class OpHessianOfGaussianEigenvalues(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float")]
    name = "HessianOfGaussianEigenvalues"
    
    def __init__(self, *args, **kwargs):
        super(OpHessianOfGaussianEigenvalues, self).__init__(*args, **kwargs)
        
    def setupFilter(self):
        scale = self.inputs["Sigma"].value
        
        def tmpFilter(source,scale,window_size,roi):
            tmpfilter = vigra.filters.hessianOfGaussianEigenvalues
            return tmpfilter(source,scale=scale,window_size=window_size,roi=(roi.start,roi.stop))

        self.vigraFilter = partial(tmpFilter,scale=scale)
        
        return scale
    
    def setupIterator(self, source, result):
        self.iterator = AxisIterator(source,'spatial',result,'spatial',[(),({'c':self.channelsPerChannel()})])   
  
    def resultingChannels(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)*self.Input.meta.shape[self.Input.meta.axistags.channelIndex]
    
    def channelsPerChannel(self):
        return self.Input.meta.axistags.axisTypeCount(vigra.AxisType.Space)
    
class OpGaussianGradientMagnitude(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float")]
    name = "GaussianGradientMagnitude"
    
    def __init__(self, *args, **kwargs):
        super(OpGaussianGradientMagnitude, self).__init__(*args, **kwargs)
        
    def setupFilter(self):
        sigma = self.inputs["Sigma"].value
                
        def tmpFilter(source,sigma,window_size,roi):
            tmpfilter = vigra.filters.gaussianGradientMagnitude
            return tmpfilter(source,sigma=sigma,window_size=window_size,roi=(roi.start,roi.stop),accumulate=False)

        self.vigraFilter = partial(tmpFilter,sigma=sigma,window_size=self.windowSize)
        return sigma

    def resultingChannels(self):
        return self.Input.meta.shape[self.Input.meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1
    


class OpPixelFeaturesPresmoothed(Operator):
    name="OpPixelFeaturesPresmoothed"
    category = "Vigra filter"

    inputSlots = [InputSlot("Input"),
                  InputSlot("Matrix"),
                  InputSlot("Scales"),
                  InputSlot("FeatureIds")] # The selection of features to compute

    outputSlots = [OutputSlot("Output"), # The entire block of features as a single image (many channels)
                   OutputSlot("Features", level=1)] # Each feature image listed separately, with feature name provided in metadata

    # Specify a default set & order for the features we compute
    DefaultFeatureIds = [ 'GaussianSmoothing',
                          'LaplacianOfGaussian',
                          'StructureTensorEigenvalues',
                          'HessianOfGaussianEigenvalues',
                          'GaussianGradientMagnitude',
                          'DifferenceOfGaussians' ]
    #set Operators
    FeatureInfos = collections.OrderedDict(
                    # FeatureId : (Operator class, sigma2, name format)
                    [ ('GaussianSmoothing' , (OpGaussianSmoothing, False, "Gaussian Smoothing (s={})")),
                      ('LaplacianOfGaussian' , (OpLaplacianOfGaussian, False, "Laplacian of Gaussian (s={})")),
                      ('StructureTensorEigenvalues' , (OpStructureTensorEigenvalues, 0.5, "Structure Tensor Eigenvalues (s={})")),
                      ('HessianOfGaussianEigenvalues' , (OpHessianOfGaussianEigenvalues, False, "Hessian of Gaussian Eigenvalues (s={})")),
                      ('GaussianGradientMagnitude' , (OpGaussianGradientMagnitude, False, "Gaussian Gradient Magnitude (s={})")),
                      ('DifferenceOfGaussians' , (OpDifferenceOfGaussians, 0.66, "Difference of Gaussians (s={})")) ] )
    
    def __init__(self, *args, **kwargs):
        super(OpPixelFeaturesPresmoothed, self).__init__(*args, **kwargs)
        #Defaults
        self.FeatureIds.setValue( self.DefaultFeatureIds )
        self.destSigma = 1.0
        self.windowSize = 4.0
        
    def setupOutputs(self):
        
        self.features = self.FeatureIds.value
        self.inScales = self.Scales.value
        self.inMatrix = self.Matrix.value
        
        #####
        #    check the input
        #####
        
        #Check for the correct type of the input
        if not isinstance(self.inMatrix, numpy.ndarray):
            raise RuntimeError("OpPixelFeatures: Please input a numpy.ndarray as 'Matrix'")
        if not isinstance(self.inScales, list):
            raise RuntimeError("OpPixelFeatures: Please input a list as 'Scales'")
        if not isinstance(self.features, list):
            raise RuntimeError("OpPixelFeatures: Please input a list as 'FeatureIds'")
        
        #Check for the correct form of the input
        if not self.inMatrix.shape == (len(self.features),len(self.inScales)):
            raise RuntimeError("OpPixelFeatures: Please input numpy.ndarray as 'Matrix', which has the form %sx%s."%(len(self.inScales),len(self.features)))
        
        #Check for the correct content of the input
        if not reduce(lambda x,y: True if x and y else False,[True if fId in self.DefaultFeatureIds else False for fId in self.features],True):
            raise RuntimeError("OpPixelFeatures: Please use only one of the following strings as a featureId:\n%s)"%(self.DefaultFeatureIds))
        if not reduce(lambda x,y: True if x and y else False,[True if type(s) in [float,int] else False for s in self.inScales],True):
            raise RuntimeError("OpPixelFeatures: Please use only one of the following types as a scale:\n int,float")
        if not reduce(lambda x,y: True if x and y else False,[True if m in [1.0,0.0] else False for m in self.inMatrix.flatten()],True):
            raise RuntimeError("OpPixelFeatures: Please use only one of the following values as a matrix entry:\n 0,1")
        
        
        #####
        #    preparations for the operator chain configuration
        #####
        
        #cast the scales to float, create modSigma list
        self.inScales = [float(s) for s in self.inScales]
        self.modSigmas = [0]*len(self.inScales)
        for i in xrange(len(self.inScales)):
            if self.inScales[i] > self.destSigma:
                self.modSigmas[i]=(sqrt(self.inScales[i]**2-self.destSigma**2))
            else:
                self.modSigmas[i]=self.inScales[i]
        
        #####
        #    build the operator chain
        #####
        
        #this list will contain the smoothing operators instances
        self.smoothers = [None]*len(self.inScales)
        
        #this matrix will contain the instances of the operators
        self.operatorMatrix = [[None]*len(self.inMatrix[0]) for i in xrange(len(self.inMatrix))]
        #this matrix will contain the start and stop position in the channel dimension of the ouput
        #for each sig/feat combination
        self.positionMatrix = [[None]*len(self.inMatrix[0]) for i in xrange(len(self.inMatrix))]
        #same information, different form
        self.featureOutputChannels = []
        self.Features.resize( (self.inMatrix == True).sum())
       
        #transpose operatorMatrix and positionMatrix for better handling
        self.inMatrix = zip(*self.inMatrix)
        self.inMatrix = [list(t) for t in self.inMatrix]
        
        #check which smoothers are actually needed,connect them and set the 
        #increment sigmas accordingly
        first = True
        proximus = None
        for sig in range(len(self.inMatrix)): #loop through sigmas
            if reduce(lambda x,y: x or y,self.inMatrix[sig]):
                self.smoothers[sig] = OpGaussianSmoothing(graph=self.graph)
                if first:
                    self.smoothers[sig].Input.connect(self.Input)
                    self.smoothers[sig].Sigma.setValue(self.modSigmas[sig])
                    first = False
                else:
                    self.smoothers[sig].Input.connect(self.smoothers[proximus].Output)
                    self.smoothers[sig].Sigma.setValue(sqrt(self.modSigmas[sig]**2-self.modSigmas[proximus]**2))
                proximus = sig
        
        #now append the filters to the smoothers and while at it, calculate the
        #outputchannels subdivision
        c = 0
        f = 0
        for sig in range(len(self.inMatrix)): #loop through sigmas
                if self.smoothers[sig] is not None: #needed at all?
                    for feat in range(len(self.inMatrix[sig])):#loop through features
                        if self.inMatrix[sig][feat]: #if true: instantiate, configure  and connect operator
                            self.operatorMatrix[sig][feat] = self.FeatureInfos[self.features[feat]][0](graph=self.graph)
                            self.operatorMatrix[sig][feat].Input.connect(self.smoothers[sig].Output)
                            self.operatorMatrix[sig][feat].Sigma.setValue(self.destSigma)
                            if self.FeatureInfos[self.features[feat]][1]:#is there a second sigma needed?
                                self.operatorMatrix[sig][feat].Sigma2.setValue(self.destSigma*self.FeatureInfos[self.features[feat]][1])
                            outChannels = self.operatorMatrix[sig][feat].Output.meta.shape[self.operatorMatrix[sig][feat].Output.meta.axistags.channelIndex]
                            self.positionMatrix[sig][feat] = [c,c + outChannels]
                            c += outChannels
                            #now the freaking features
                            self.Features[f].meta.assignFrom(self.operatorMatrix[sig][feat].Output.meta)
                            self.Features[f].meta.description = self.FeatureInfos[self.features[feat]][2].format(self.inScales[sig])
                            self.featureOutputChannels.append(self.positionMatrix[sig][feat])
                            f+=1
                            
        for index, slot in enumerate(self.Features):
            assert slot.meta.description is not None, "Feature {} has no description!".format(index)
        
        self.Output.meta.assignFrom(self.Input.meta)
        newShape = list(self.Input.meta.shape)
        newShape[self.Input.meta.axistags.channelIndex] = c
        self.Output.meta.shape = tuple(newShape) 
        
    def execute(self,slot,subindex,roi,result):
        #####
        #    OutputSlot: "Output"
        #####
        if slot == self.Output:
            #get the channel information
            cIndex = self.Output.meta.axistags.channelIndex
            cstart,cstop = roi.start[cIndex],roi.stop[cIndex]
            #open a pool. find some hookers, put them in
            pool = Pool()
            resultC = 0 #channel variable for result
            for sig in range(len(self.positionMatrix)):#loop sigma
                for feat in range(len(self.positionMatrix[sig])):#loop features
                    if self.positionMatrix[sig][feat]:
                        start,stop = self.positionMatrix[sig][feat] #start,stop of the individual channeldims
                        if cstop-start > 0 and stop-cstart > 0: #check if its requested
                           rstart = max(0,cstart-start) #calculate request start
                           rstop = min(cstop-start,stop-start) #calculate request stop
                           reqroi = roi.copy() #roi for the request
                           resroi = roi.copy() #roi for the result array
                           reqroi.setDim(cIndex,rstart,rstop) #adjust roi for the individual requests
                           resroi.setDim(cIndex,resultC,resultC+rstop-rstart) #set result roi channelposition
                           resultC += rstop-rstart #carry on.
                           req = self.operatorMatrix[sig][feat].Output(reqroi.start,reqroi.stop).writeInto(result[resroi.toSlice()])
                           pool.add(req)
            pool.wait()
            return result
        #####
        #    OutputSlot: "Features"
        #####
        elif slot == self.Features:
            index = subindex[0]
            cIndex = self.Output.meta.axistags.channelIndex
            roi.setDim(cIndex,self.featureOutputChannels[index][0] + roi.start[cIndex],\
                              self.featureOutputChannels[index][0] + roi.stop[cIndex])
            return self.execute(self.Output,(), roi, result)

    def propagateDirty(self,slot,subindex,roi):
        if slot == self.Input:
            cIndex = self.Input.meta.axistags.channelIndex
            cSizeIn = self.Input.meta.shape[cIndex]
            cSizeOut = self.Output.meta.shaoe[cIndex]
            dirtyC = roi.stop[cIndex] - roi.start[cIndex]
            # If all the input channels were dirty, the dirty output region is a contiguous block
            if dirtyC == cSize:
                droi = roi.copy()
                droi.setDim(cIndex,0,cSizeOut)
                self.Output.setDirty(dirtyRoi.start, dirtyRoi.stop)
            else:
                # Only some input channels were dirty,
                # so we must mark each dirty output region separately.
                for f in range(len(self.features)):
                    cPerc = self.FeatureInfos[f][0].channelsPerChannel()
                    fStart = self.featureOutputChannels[f][0] 
                    start = fStart+cPerc*roi.start[cIndex]
                    stop = fStart+cPerc*roi.stop[cIndex]
                    droi = roi.copy()
                    droi.setDim(cIndex,start,stop)
                    self.Output.setDirty(droi.start,droi.stop)
        elif (slot == self.Matrix
              or slot == self.Scales
              or slot == self.FeatureIds):
            self.Output.setDirty(slice(None))
        else:
            assert False, "Unknown dirty input slot."
        pass

if __name__ == "__main__":
    from lazyflow.graph import Graph
    from volumina.viewer import Viewer
    from PyQt4.QtGui import QMainWindow, QApplication
    import sys,vigra
    g = Graph()
    v = vigra.VigraArray(1000*numpy.random.rand(100,100,3),axistags = vigra.defaultAxistags('xyc'))
    op = OpPixelFeaturesPresmoothed(graph = g)
    op.FeatureIds.setValue(["StructureTensorEigenvalues","HessianOfGaussianEigenvalues"])
    op.Scales.setValue([1.5,2.0])
    op.Input.setValue(v)
    n = numpy.ndarray((2,2))
    n[:] = [[1,1],[1,1]] 
    op.Matrix.setValue(n)
    w = op.Output().wait()
    for i in range(w.shape[2]):
        vigra.impex.writeImage(w[:,:,i],"%02d.jpg"%(i))
    print w.shape