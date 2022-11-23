"""TSBpt_SANImodel_recursiveLayers.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBpt_main.py

# Usage:
see TSBpt_main.py

# Description:
TSBpt SANI model recursiveLayers

"""

import torch as pt
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

recursiveLayers = True

class SANIrecursiveLayersConfig():
	def __init__(self, vocabularySize, batchSize, sequenceLength, hiddenLayerSize, embeddingLayerSize):
		self.vocab_size = vocabularySize
		self.num_layers = sequenceLength
		self.N = batchSize
		self.L = sequenceLength
		#self.D = bidirectional	#default: 1 #2 if bidirectional=True otherwise 1
		self.hiddenLayerSize = hiddenLayerSize
		self.embeddingLayerSize = embeddingLayerSize	#input token embedding size (equivalent to roberta hidden_size)
		self.pad_token_id = 1	#default=1 #https://huggingface.co/transformers/v2.11.0/model_doc/roberta.html 
		self.applyIOconversionLayers = False
		if(embeddingLayerSize != hiddenLayerSize):
			self.applyIOconversionLayers = True
		
class SANIrecursiveLayersModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.word_embeddings = nn.Embedding(config.vocab_size, config.embeddingLayerSize, padding_idx=config.pad_token_id)
		self.outputStates = [None]*config.num_layers
		if(recursiveLayers):
			self.sani =	nn.Linear(config.hiddenLayerSize*2, config.hiddenLayerSize)	#CHECKTHIS
		else:
			self.SANIlayers = []
			for layerIndex in range(config.num_layers):
				sani = nn.Linear(config.hiddenLayerSize*2, config.hiddenLayerSize)
				self.SANIlayers.append(sani)
		if(config.applyIOconversionLayers):
			self.inputLayer = nn.Linear(config.embeddingLayerSize, config.hiddenLayerSize)
			self.outputLayer = nn.Linear(config.hiddenLayerSize, config.embeddingLayerSize)
		self.lossFunction = CrossEntropyLoss()	#CHECKTHIS
		self.hiddenStateLast = [None]*config.num_layers	#last activated hidden state of each layer
		for layerIndex in range(config.num_layers):
			self.hiddenStateLast[layerIndex] = pt.zeros([config.N, config.hiddenLayerSize])
				
	def forward(self, labels, device):
		
		config = self.config
		
		inputsEmbeddings = self.word_embeddings(labels)
		if(config.applyIOconversionLayers):
			inputsEmbeddingsReshaped = pt.reshape(inputsEmbeddings, (config.N*config.L, config.embeddingLayerSize))
			inputState = self.inputLayer(inputsEmbeddingsReshaped)
			inputState = pt.reshape(inputState, (config.N, config.L, config.hiddenLayerSize))
		else:
			inputState = inputsEmbeddings
		
		for sequenceIndex in range(config.L):
			hiddenState = inputState[:, sequenceIndex, :]
			sequenceIndexMaxLayers = sequenceIndex
			for layerIndex in range(sequenceIndexMaxLayers+1):
				if(layerIndex == sequenceIndexMaxLayers):
					self.hiddenStateLast[layerIndex] = hiddenState
					self.outputStates[sequenceIndex] = hiddenState	#or self.outputStates[layerIndex]
				else:
					previousInput = self.hiddenStateLast[layerIndex]
					currentInput = hiddenState
					#print("previousInput.shape = ", previousInput.shape)
					#print("currentInput.shape = ", currentInput.shape)
					combinedInput = pt.concat((previousInput, currentInput), dim=1)	#CHECKTHIS
					if(recursiveLayers):
						sani = self.sani
					else:
						sani = self.SANIlayer[layerIndex]
					hiddenState = sani(combinedInput)
					self.hiddenStateLast[layerIndex] = currentInput
			
		if(config.applyIOconversionLayers):
			outputState = pt.concat(self.outputStates, dim=0)
			y = self.outputLayer(outputState)
			y = pt.reshape(y, (config.N, config.L, config.embeddingLayerSize))
		else:
			y = pt.stack(self.outputStates, dim=1)
		yHat = inputsEmbeddings

		loss = self.lossFunction(y, yHat)
		
		return loss
	
