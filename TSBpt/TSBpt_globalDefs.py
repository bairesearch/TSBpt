"""TSBpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBpt_globalDefs.py

# Usage:
see TSBpt_globalDefs.py

# Description:
TSBpt globalDefs

"""

#recursive algorithm selection:
useAlgorithmTransformer = True
useAlgorithmRNN = False
useAlgorithmSANI = False

userName = 'user'	#default: user

useSmallDatasetDebug = False
useSingleHiddenLayerDebug = False
usePretainedModelDebug = False	#executes stateTestDataset only
useSmallBatchSizeDebug = False

useSmallTokenizerTrainNumberOfFiles = True	#used during rapid testing only (FUTURE: assign est 80 hours to perform full tokenisation train)

statePreprocessDataset = False	#only required once
stateTrainTokenizer = False	#only required once
stateTrainDataset = True
stateTestDataset = False	#requires reserveValidationSet


trainStartEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train x epochs at a time)
trainStartDataFile = 0	#default: 0	#start data file to train (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfDataFiles = 100	#2	#100	#default: -1 (all)	#number of data files to train (for production typically train x dataFiles at a time)	#< numberOfDataFiles (30424) * trainSplitFraction
testNumberOfDataFiles = 10	#2	#10	#default: -1 (all)

		
reserveValidationSet = True	#reserves a fraction of the data for validation
trainSplitFraction = 0.9	#90% train data, 10% test data

if(useAlgorithmTransformer):
	batchSize = 8	#default: 8	#8 and 16 train at approx same rate (16 uses more GPU ram)	#depends on GPU RAM	#with 12GB GPU RAM, batchSize max = 16
	learningRate = 1e-4
elif(useAlgorithmRNN):
	batchSize = 8  
	learningRate = 1e-4
elif(useAlgorithmSANI):
	batchSize = 8  
	learningRate = 1e-4

if(useSmallBatchSizeDebug):
	batchSize = 1	#use small batch size to enable simultaneous execution (GPU ram limited) 
	
numberOfSamplesPerDataFile = 10000
numberOfSamplesPerDataFileLast = 423
dataFileLastSampleIndex = 30423

#storage location vars (requires 4TB harddrive);
downloadCacheFolder = '/media/' + userName + '/datasets/cache'
dataFolder = '/media/' + userName + '/datasets/data'
modelFolderName = '/media/' + userName + '/large/source/ANNpython/TSBpt/model'

modelSaveNumberOfBatches = 1000	#resave model after x training batches


sequenceMaxNumTokens = 512	#window length (transformer/RNN/SANI)

#transformer only;
customMaskTokenID = 4	#3
fractionOfMaskedTokens = 0.15

vocabularySize = 30522	#default: 30522
