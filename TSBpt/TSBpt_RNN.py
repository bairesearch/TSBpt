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

from TSBpt_RNNmodel import recursiveLayers, RNNrecursiveLayersModel, RNNrecursiveLayersConfig, calculateVocabPredictionHeadLoss, applyIOconversionLayers

hiddenLayerSize = 1024	#65536	#2^16 - large hidden size is required for recursive RNN as parameters are shared across a) sequence length and b) number of layers
if(applyIOconversionLayers):
	embeddingLayerSize = 768
else:
	embeddingLayerSize = hiddenLayerSize

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
	input_ids = batch['input_ids'].to(device)
	attention_mask = batch['attention_mask'].to(device)
	labels = batch['labels'].to(device)
	
	loss, outputs = model(labels, device)
	
	if(calculateVocabPredictionHeadLoss):
		accuracy = TSBpt_data.getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs)
	else:
		accuracy = 0.0
	
	return loss, accuracy

