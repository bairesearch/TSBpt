"""TSBpt_SANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBpt_main.py

# Usage:
see TSBpt_main.py

# Description:
TSBpt SANI

Similar to WaveNet 

"""

import torch as pt

from TSBpt_globalDefs import *
import TSBpt_data

from TSBpt_SANImodel_recursiveLayers import recursiveLayers, SANIrecursiveLayersModel, SANIrecursiveLayersConfig, calculateVocabPredictionHeadLoss

embeddingLayerSize = 768
hiddenLayerSize = 1024	#1024	#depends on GPU memory	#2^16 = 65536 - large hidden size is required for recursive SANI as parameters are shared across a) sequence length and b) number of layers
#numberOfHiddenLayers = 6

modelPathName = modelFolderName + '/modelSANI.pt'

#useBidirectionalSANI = False	#not currently supported
#if(useBidirectionalSANI):
#	bidirectional = 2
#else:
#	bidirectional = 1

def createModel():
	print("creating new model")
	config = SANIrecursiveLayersConfig(
		vocabularySize=vocabularySize,
		#numberOfHiddenLayers=numberOfHiddenLayers,
		batchSize=batchSize,
		sequenceLength=sequenceMaxNumTokens,
		#bidirectional=bidirectional,
		hiddenLayerSize=hiddenLayerSize,
		embeddingLayerSize=embeddingLayerSize
	)
	model = SANIrecursiveLayersModel(config)
	return model

def loadModel():
	print("loading existing model")
	model = pt.load(modelPathName)
	return model
	
def saveModel(model):
	pt.save(model, modelPathName)

def propagate(device, model, tokenizer, batch):
	
	labels = batch['labels'].to(device)
	loss, predictionScores = model(labels, device)
	if(calculateVocabPredictionHeadLoss):
		accuracy = getAccuracy(tokenizer, labels, predictionScores)
	else:
		accuracy = 0.0
	
	return loss, accuracy

def getAccuracy(tokenizer, labels, outputs):
	tokenizerNumberTokens = TSBpt_data.getTokenizerLength(tokenizer)
	
	tokenLogits = outputs.detach()

	tokenLogitsTopIndex = pt.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = pt.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	

		comparison = (tokenLogitsTopIndex == labels).float()
		accuracy = (pt.sum(comparison)/comparison.nelement()).cpu().numpy()
	else:
		labelsExpanded = pt.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		
		accuracy = (pt.sum(comparison)/comparison.nelement()).cpu().numpy()
		
	return accuracy
