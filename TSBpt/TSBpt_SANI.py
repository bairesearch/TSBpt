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

import torch

from TSBpt_globalDefs import *
import TSBpt_data

from TSBpt_SANImodel_recursiveLayers import recursiveLayers, SANIrecursiveLayersModel, SANIrecursiveLayersConfig

embeddingLayerSize = 768
hiddenLayerSize = 768	#65536	#2^16 - large hidden size is required for recursive SANI as parameters are shared across a) sequence length and b) number of layers
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
