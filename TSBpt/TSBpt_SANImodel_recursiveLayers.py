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
skipLayers = True
if(skipLayers):
	skipLayersDominance = 0.9	#degree of sequentialInputState preservation (direct recursive/loop connection) as signal is propagated to higher layers
	skipLayersNorm = True
retainHiddenEmbeddingStructure = True	#experimental	#do not mix hidden embeddings (for every unit/neuron in hidden layer, calculate new value based on current and previous value)

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
			self.saniLayer = self.generateSANIlayer(config.hiddenLayerSize)
		else:
			self.SANIlayers = []
			for layerIndex in range(config.num_layers):
				saniLayer = generateSANIlayer(self, config.hiddenLayerSize)
				self.SANIlayers.append(saniLayer)
		if(config.applyIOconversionLayers):
			self.inputLayer = nn.Linear(config.embeddingLayerSize, config.hiddenLayerSize)
			self.outputLayer = nn.Linear(config.hiddenLayerSize, config.embeddingLayerSize)
		self.lossFunction = CrossEntropyLoss()	#CHECKTHIS
		self.hiddenStateLast = [None]*config.num_layers	#last activated hidden state of each layer
		for layerIndex in range(config.num_layers):
			self.hiddenStateLast[layerIndex] = pt.zeros([config.N, config.hiddenLayerSize])
		self.activationFunction = pt.nn.ReLU()
		if(skipLayers):
			if(skipLayersNorm):
				self.layerNorm = pt.nn.LayerNorm(config.hiddenLayerSize)

	def generateSANIlayer(self, hiddenLayerSize):
		#https://stackoverflow.com/questions/58374980/run-multiple-models-of-an-ensemble-in-parallel-with-pytorch/58389075#58389075
		if(retainHiddenEmbeddingStructure):
			self.numberOfHeads = hiddenLayerSize
			self.numberOfInputChannels = 2
			#input shape = B x (2 * numberOfHeads) x 1
			#output shape = B x (1 x numberOfHeads) x 1
			saniLayer = pt.nn.Conv1d(2*self.numberOfHeads, 1*self.numberOfHeads, kernel_size=1, groups=self.numberOfHeads)
		else:
			saniLayer =	pt.nn.Linear(hiddenLayerSize*2, hiddenLayerSize)	#CHECKTHIS
		return saniLayer
				
	def forward(self, labels, device):
		
		config = self.config
		
		inputsEmbeddings = self.word_embeddings(labels)
		if(config.applyIOconversionLayers):
			inputState = pt.reshape(inputsEmbeddings, (config.N*config.L, config.embeddingLayerSize))
			inputState = self.inputLayer(inputState)
			inputState = self.activationFunction(inputState)
			inputState = pt.reshape(inputState, (config.N, config.L, config.hiddenLayerSize))
			#print("inputState.shape = ", inputState.shape)
		else:
			inputState = inputsEmbeddings
			#print("inputState.shape = ", inputState.shape)
		
		for sequenceIndex in range(config.L):
			sequentialInputState = inputState[:, sequenceIndex, :]
			hiddenState = sequentialInputState
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
					if(recursiveLayers):
						saniLayer = self.saniLayer
					else:
						saniLayer = self.SANIlayers[layerIndex]
					if(retainHiddenEmbeddingStructure):
						combinedInput = pt.stack([previousInput, currentInput], dim=1)	#shape = [batchSize, numberOfHeads, numberOfInputChannels, ..]
						combinedInput = pt.reshape(combinedInput, (config.N, self.numberOfHeads*self.numberOfInputChannels, 1))
						currentOutput = self.saniLayer(combinedInput)
						currentOutput = pt.reshape(currentOutput, (config.N, self.numberOfHeads))
					else:
						combinedInput = pt.concat((previousInput, currentInput), dim=1)	#CHECKTHIS
						currentOutput = saniLayer(combinedInput)
					currentOutput = self.activationFunction(currentOutput)
					if(skipLayers):
						currentOutput = pt.add(currentOutput*(1.0-skipLayersDominance), sequentialInputState)
						#print("currentOutput.shape = ", currentOutput.shape)
						if(skipLayersNorm):
							currentOutput = self.layerNorm(currentOutput)
					hiddenState = currentOutput
					self.hiddenStateLast[layerIndex] = currentInput
			
		if(config.applyIOconversionLayers):
			outputState = pt.concat(self.outputStates, dim=0)
			outputState = self.outputLayer(outputState)
			outputState = self.activationFunction(outputState)
			y = pt.reshape(outputState, (config.N, config.L, config.embeddingLayerSize))
			#print("y.shape = ", y.shape)
		else:
			y = pt.stack(self.outputStates, dim=1)
			#print("y.shape = ", y.shape)
		yHat = inputsEmbeddings

		loss = self.lossFunction(y, yHat)
		
		return loss
	
