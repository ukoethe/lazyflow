from lazyflow.graph import Operator,InputSlot,OutputSlot
from lazyflow.helpers import newIterator
from lazyflow.operators.obsolete.generic import OpMultiArrayStacker
from lazyflow.operators.obsolete.vigraOperators import Op50ToMulti,\
    OpGaussianGradientMagnitude
from lazyflow.rtype import SubRegion
import numpy
import vigra
from math import sqrt
from functools import partial
from lazyflow.roi import roiToSlice
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
        returns how the number of source channels which get mapped on one unit
        of result channels
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
        return self.windowSize*sigma
    
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
        inputSlot = self.inputs["Input"]
        outputSlot = self.outputs["Output"]
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
        source = self.inputs["Input"](roi.start,roi.stop).wait()
        source = vigra.VigraArray(source,axistags=axistags)
        
        #set up the grid for the iterator, and the iterator
        srcGrid = list(source.shape)
        trgtGrid = list(source.shape)
        srcGrid[channelIndex] = channelRes
        trgtGrid[channelIndex] = self.channelsPerChannel()
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
        warnings.warn("Parallelize me!")
        for src,trgt,mask in nIt:
            result[trgt] = self.vigraFilter(source = source[src],window_size=self.windowSize,roi=origRoi)[mask]
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
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]

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
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]
    
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
        return self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)*(self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
    
    def channelsPerChannel(self):
        return self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)*(self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space) + 1) / 2
    
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
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1

class OpStructureTensorEigenvalues(OpBaseVigraFilter):
    inputSlots = [InputSlot("Input"), InputSlot("Sigma", stype = "float"),InputSlot("Sigma2", stype = "float")]
    name = "StructureTensorEigenvalues"
    
    def __init__(self, *args, **kwargs):
        super(OpStructureTensorEigenvalues, self).__init__(*args, **kwargs)
    
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
        return self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)
    
    def channelsPerChannel(self):
        return self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)
    
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
        return self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)*self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.channelIndex]
    
    def channelsPerChannel(self):
        return self.inputs["Input"].meta.axistags.axisTypeCount(vigra.AxisType.Space)
    
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
        return self.inputs["Input"].meta.shape[self.inputs["Input"].meta.axistags.index('c')]
    
    def channelsPerChannel(self):
        return 1
    

class OpPixelFeaturesPresmoothed(Operator):
    
    name="OpPixelFeaturesPresmoothed"
    Input = InputSlot()
    Matrix = InputSlot()
    Scales = InputSlot()
    FeatureIds = InputSlot()
    
    Output = OutputSlot()
    Features = OutputSlot(level=1)
    
    #set Operators
    FeatureInfos = collections.OrderedDict(
                    # FeatureId : (Operator class, sigma2, name format)
                    [ ('GaussianSmoothing'            , (OpGaussianSmoothing, False, "Gaussian Smoothing (s={})")),
                      ('LaplacianOfGaussian'          , (OpLaplacianOfGaussian, False, "Laplacian of Gaussian (s={})")),
                      ('StructureTensorEigenvalues'   , (OpStructureTensorEigenvalues, 0.5, "Structure Tensor Eigenvalues (s={})")),
                      ('HessianOfGaussianEigenvalues' , (OpHessianOfGaussianEigenvalues, False, "Hessian of Gaussian Eigenvalues (s={})")),
                      ('GaussianGradientMagnitude'    , (OpGaussianGradientMagnitude, False, "Gaussian Gradient Magnitude (s={})")),
                      ('DifferenceOfGaussians'        , (OpDifferenceOfGaussians, 0.66, "Difference of Gaussians (s={})")) ] )

    def __init__(self, *args, **kwargs):
        super(OpPixelFeaturesPresmoothed, self).__init__(*args, **kwargs)
        
        self.multi = Op50ToMulti(graph=self.graph)
        self.stacker = OpMultiArrayStacker(graph=self.graph)
        self.smoother = OpGaussianSmoothing(graph=self.graph)
        self.destSigma = 1.0
        self.windowSize = 4
        self.operatorList = [OpGaussianSmoothing,OpLaplacianOfGaussian,\
                        OpStructureTensorEigenvalues,OpHessianOfGaussianEigenvalues,\
                        OpGaussianGradientMagnitude,OpDifferenceOfGaussians]
        self.opInstances = []
    
    def propagateDirty(self,slot,subindex,roi):
        if self.Input is not None and self.Matrix is not None:
            opInstances = []
            for i in range(len(self.operatorList)):
                op = self.operatorList[i](graph=self.graph)
                op.inputs["Input"].connect(self.inputs["Input"])
                op.inputs["Sigma"].setValue(1.0)#dummy
                if "Sigma2" in op.inputs.keys():
                    op.inputs["Sigma2"].setValue(2.0)#dummy
                opInstances.append(op)
            self.opInstances = opInstances

            if slot == self.inputs["Input"]:
                cIndex = self.inputs["Input"].axistags.channelIndex
                inMatrix = self.inputs["Matrix"].value
                usedOperators = [reduce(lambda x,y: x or y,operator,False) for operator in inMatrix]
                cCount = 0
                for i in range(len(usedOperators)):
                    if usedOperators[i]:
                        roiCopy = roi.copy()
                        roiCopy.start[cIndex] = cCount + roi.start[cIndex]*opInstances[i].channelsPerChannel()
                        roiCopy.stop[cIndex] = cCount + roi.stop[cIndex]*opInstances[i].channelsPerChannel()
                        self.outputs["Output"].setDirty(roiCopy)
                    cCount += (roi.stop[cIndex]-roi.start[cIndex])*opInstances[i].channelsPerChannel()
            
    def setupOutputs(self):
        
        #TODO: Different assertions and stuff.
        self.inMatrix = self.inputs["Matrix"].value
        self.inScales = self.inputs["Scales"].value
        self.modSigmas = [0]*len(self.inScales)
        self.maxSigma = numpy.max(self.inScales)
        self.incrSigmas = [0]*len(self.inScales)
        
        #set modified sigmas
        for i in xrange(len(self.inScales)):
            if self.inScales[i] > self.destSigma:
                self.modSigmas[i]=(sqrt(self.inScales[i]**2-self.destSigma**2))
            else:
                self.modSigmas[i]=self.inScales[i]
                
        self.modSigmas.insert(0,0)
        for i in xrange(len(self.modSigmas)-1):
            self.incrSigmas[i]=sqrt(self.modSigmas[i+1]**2-self.modSigmas[i]**2)
        self.modSigmas.remove(0)
            
        operatorList = []
        scaleMultiplyList = []
        nameFormatList = []
        for featureId in self.FeatureIds.value:
            op, sigma, nameFormat = self.FeatureInfos[featureId]
            operatorList.append(op)
            scaleMultiplyList.append(sigma)
            nameFormatList.append(nameFormat)
        self.operatorList = operatorList 
        
        self.operatorMatrix = [[None]*len(self.inMatrix[0]) for i in xrange(len(self.inMatrix))]

        channelCount = 0
        featureCount = 0
        self.featureOutputChannels = []
        k=0
        for i in xrange(len(self.inMatrix)): #Cycle through operators == i
            for j in xrange(len(self.inMatrix[i])): #Cycle through sigmas == j
                if self.inMatrix[i][j]:
                    self.operatorMatrix[i][j] = operatorList[i](graph=self.graph)
                    self.operatorMatrix[i][j].inputs["Input"].connect(self.inputs["Input"])
                    self.operatorMatrix[i][j].inputs["Sigma"].setValue(self.destSigma)
                    self.operatorMatrix[i][j].outputs["Output"].meta.description = nameFormatList[i].format( self.inScales[j] )
                    if scaleMultiplyList[i]:
                        self.operatorMatrix[i][j].inputs["Sigma2"].setValue(self.destSigma*scaleMultiplyList[i])
                    self.multi.inputs["Input%02d"%(k)].connect(self.operatorMatrix[i][j].outputs["Output"])
                    k += 1
                    
                    # Prepare the individual features
                    featureCount += 1
                    self.Features.resize( featureCount )

                    featureMeta = self.operatorMatrix[i][j].outputs["Output"].meta
                    featureChannels = featureMeta.shape[ featureMeta.axistags.index('c') ]
                    self.Features[featureCount-1].meta.assignFrom( featureMeta )
                    self.featureOutputChannels.append( (channelCount, channelCount + featureChannels) )
                    channelCount += featureChannels

        self.stacker.inputs["AxisFlag"].setValue('c')
        self.stacker.inputs["AxisIndex"].setValue(self.inputs["Input"].meta.axistags.index('c'))
        self.stacker.inputs["Images"].connect(self.multi.outputs["Outputs"])
        
        self.outputs["Output"].meta.axistags = self.stacker.outputs["Output"].meta.axistags
        self.outputs["Output"].meta.shape = self.stacker.outputs["Output"].meta.shape
        self.outputs["Output"].meta.dtype = numpy.float32 
        
        #transpose operatorMatrix for better handling
        opMatrix = self.operatorMatrix
        newOpMatrix = [[None]*len(opMatrix)] * len(opMatrix[0])
        opList = []
        for l in opMatrix:
            opList += l
        for i in xrange(len(opList)):
            newOpMatrix[i/len(opMatrix)][i%len(opMatrix)] = opList[i]
        self.operatorMatrix = newOpMatrix

    def execute(self, slot, subindex, roi, result):
        if slot == self.Features:
            return self.executeFeatures(slot, subindex, roi, result)
        elif slot == self.Output:
            return self.executeOutput(slot, subindex, roi, result)
        else:
            assert False, "Unknown output slot"

    def executeFeatures(self, slot, subindex, roi, result):
        key = roiToSlice(roi.start, roi.stop)
        index = subindex[0]
        subslot = self.Features[index]
        key = list(key)
        channelIndex = self.Input.meta.axistags.index('c')
        
        # Translate channel slice to the correct location for the output slot.
        key[channelIndex] = slice(self.featureOutputChannels[index][0] + key[channelIndex].start,
                                  self.featureOutputChannels[index][0] + key[channelIndex].stop)
        roi = SubRegion(subslot, pslice=key)

        # Get output slot region for this channel
        return self.execute(self.Output, (), roi, result)

    def executeOutput(self, slot, subindex, roi, result):
        #Get axistags and inputShape
        axistags = self.Input.meta.axistags
        inputShape  = self.Input.meta.shape
        resultCIndex = self.Output.meta.axistags.channelIndex
        tIndex = self.Output.meta.axistags.index('t')
        
        #Set up roi 
        roi.setInputShape(inputShape)

        #Request Required Region
        srcRoi = roi.expandByShape(self.maxSigma*self.windowSize,resultCIndex, tIndex)
        roi.setDim( resultCIndex, 0, inputShape[resultCIndex] )
        source = self.inputs["Input"](srcRoi.start,srcRoi.stop).wait()
        source = source.view(vigra.VigraArray)
        source.axistags = self.Input.meta.axistags
        
        #disconnect all operators
        opM = self.operatorMatrix
        cIter = 0
        for sig in xrange(len(opM)):#for each sigma
            self.smoother.inputs["Sigma"].setValue(self.incrSigmas[sig])
            self.smoother.inputs["Input"].setValue( source )
            source = self.smoother.outputs["Output"]().wait() # FIXME: 
            source = source.view(vigra.VigraArray)
            source.axistags = self.Input.meta.axistags
            for op in xrange(len(opM[sig])):#for each operator with this sigma
                if opM[sig][op] is not None:
                    opM[sig][op].inputs["Input"].disconnect()
                    opM[sig][op].inputs["Input"].setValue(source)
                    cIndex = opM[sig][op].outputs["Output"].meta.axistags.channelIndex
                    cSize  = opM[sig][op].outputs["Output"].meta.shape[cIndex]
                    slicing = [slice(0,result.shape[i],None) if i != cIndex \
                               else slice(cIter,cIter+cSize,None) for i in \
                               range(len(result.shape))]
                    result[slicing] = opM[sig][op].outputs["Output"]().wait()
                    cIter += cSize
        return result     
    
if __name__ == "__main__":
    pass