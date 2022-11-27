"""TSBpt_RNN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBpt_main.py

# Usage:
see TSBpt_main.py

# Description:
TSBpt RNN

"""

import torch as pt

from TSBpt_globalDefs import *
import TSBpt_data

from TSBpt_RNNmodel_recursiveLayers import recursiveLayers, RNNrecursiveLayersModel, RNNrecursiveLayersConfig, calculateVocabPredictionHeadLoss

embeddingLayerSize = 768
hiddenLayerSize = 1024	#65536	#2^16 - large hidden size is required for recursive RNN as parameters are shared across a) sequence length and b) number of layers
numberOfHiddenLayers = 6

modelPathName = modelFolderName + '/modelRNN.pt'

useBidirectionalRNN = False
if(useBidirectionalRNN):
	bidirectional = 2
else:
	bidirectional = 1

def createModel():
	print("creating new model")
	config = RNNrecursiveLayersConfig(
		vocabularySize=vocabularySize,
		numberOfHiddenLayers=numberOfHiddenLayers,
		batchSize=batchSize,
		sequenceLength=sequenceMaxNumTokens,
		bidirectional=bidirectional,
		hiddenLayerSize=hiddenLayerSize,
		embeddingLayerSize=embeddingLayerSize
	)
	model = RNNrecursiveLayersModel(config)
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
