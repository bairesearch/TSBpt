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

import torch

from TSBpt_globalDefs import *
import TSBpt_data

from TSBpt_RNNmodel_recursiveLayers import recursiveLayers, RNNrecursiveLayersModel, RNNrecursiveLayersConfig

embeddingLayerSize = 768
hiddenLayerSize = 768	#65536	#2^16 - large hidden size is required for recursive RNN as parameters are shared across a) sequence length and b) number of layers
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
	model = torch.load(modelPathName)
	return model
	
def saveModel(model):
	torch.save(model, modelPathName)

def propagate(device, model, tokenizer, batch):
	
	labels = batch['labels'].to(device)
	loss = model(labels, device)
	accuracy = 0.0	#TODO: requires implementation	#getAccuracy(tokenizer, labels, outputs)
	
	return loss, accuracy

#def getAccuracy():
