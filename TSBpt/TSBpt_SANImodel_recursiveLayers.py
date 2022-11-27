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
from transformers.activations import gelu

recursiveLayers = True
skipLayers = True
if(skipLayers):
	skipLayersDominance = 0.9	#degree of sequentialInputState preservation (direct recursive/loop connection) as signal is propagated to higher layers
	skipLayersNorm = True
retainHiddenEmbeddingStructure = False	#experimental	#do not mix hidden embeddings (for every unit/neuron in hidden layer, calculate new value based on current and previous value)
parallelProcessLayers = True
calculateVocabPredictionHeadLoss = True	#apply loss to vocubulary predictions (rather than embedding predictions)

class SANIrecursiveLayersConfig():
	def __init__(self, vocabularySize, batchSize, sequenceLength, hiddenLayerSize, embeddingLayerSize):
		self.vocab_size = vocabularySize
		self.num_layers = sequenceLength
		self.batchSize = batchSize
		self.sequenceLength = sequenceLength
		#self.bidirectional = bidirectional	#default: 1 #2 if bidirectional=True otherwise 1
		self.hiddenLayerSize = hiddenLayerSize
		self.embeddingLayerSize = embeddingLayerSize	#input token embedding size (equivalent to roberta hidden_size)
		self.pad_token_id = 1	#default=1 #https://huggingface.co/transformers/v2.11.0/model_doc/roberta.html 
		self.applyIOconversionLayers = False
		if(embeddingLayerSize != hiddenLayerSize):
			self.applyIOconversionLayers = True
		self.layer_norm_eps = 1e-12	#https://huggingface.co/transformers/v4.2.2/_modules/transformers/models/bert/configuration_bert.html#BertConfig

#based on RobertaLMHead
class ModelVocabPredictionHead(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.hiddenLayerSize, config.hiddenLayerSize)
		self.layer_norm = nn.LayerNorm(config.hiddenLayerSize, eps=config.layer_norm_eps)
		self.decoder = nn.Linear(config.hiddenLayerSize, config.vocab_size)
		self.bias = nn.Parameter(pt.zeros(config.vocab_size))
		self.decoder.bias = self.bias

	def forward(self, features, **kwargs):
		x = self.dense(features)
		x = gelu(x)
		x = self.layer_norm(x)
		x = self.decoder(x)
		return x

	def _tie_weights(self):
		self.bias = self.decoder.bias
						
class SANIrecursiveLayersModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.word_embeddings = nn.Embedding(config.vocab_size, config.embeddingLayerSize, padding_idx=config.pad_token_id)
		self.outputStateList = [None]*config.num_layers
		if(not parallelProcessLayers):
			self.hiddenStateLastList = [None]*config.num_layers	#last activated hidden state of each layer
			for layerIndex in range(config.num_layers):
				self.hiddenStateLastList[layerIndex] = pt.zeros([config.batchSize, config.hiddenLayerSize])
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
		self.activationFunction = pt.nn.ReLU()
		if(calculateVocabPredictionHeadLoss):
			self.lossFunction = CrossEntropyLoss()
		else:
			self.lossFunction = MSELoss()
		if(skipLayers):
			if(skipLayersNorm):
				self.layerNorm = pt.nn.LayerNorm(config.hiddenLayerSize)
		self.predictionHead = ModelVocabPredictionHead(config)

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
			inputState = pt.reshape(inputsEmbeddings, (config.batchSize*config.sequenceLength, config.embeddingLayerSize))
			inputState = self.inputLayer(inputState)
			inputState = self.activationFunction(inputState)
			inputState = pt.reshape(inputState, (config.batchSize, config.sequenceLength, config.hiddenLayerSize))
		else:
			inputState = inputsEmbeddings
			
		if(parallelProcessLayers):
			hiddenState = inputState
			for layerIndex in range(0, config.sequenceLength):
				if(layerIndex < config.sequenceLength-2):
					#print("layerIndex = ", layerIndex)
					#print("hiddenState.shape = ", hiddenState.shape)
					minSequenceIndex = layerIndex
					blankSequentialHiddenStates = pt.zeros(hiddenState.shape[0], minSequenceIndex+1, hiddenState.shape[2]).to(device)
					sequentialInputState = pt.reshape(hiddenState, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
					previousInput = pt.cat((blankSequentialHiddenStates, hiddenState[:, minSequenceIndex:-1]), dim=1)
					currentInput = pt.cat((blankSequentialHiddenStates, hiddenState[:, minSequenceIndex+1:]), dim=1)
					previousInput = pt.reshape(previousInput, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
					currentInput = pt.reshape(currentInput, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
					currentOutput = self.processLayer(sequentialInputState, layerIndex, currentInput, previousInput)
					currentOutput = pt.reshape(hiddenState, (config.batchSize, config.sequenceLength, config.hiddenLayerSize))
					hiddenState = currentOutput
					currentOutputSequentialIndex = currentOutput[:, minSequenceIndex+1]
				else:
					#last two tokens in window provide no prediction (pad hidden states with zeros). final prediction uses index(N-2) and index(N-1) to predict next token
					currentOutputSequentialIndex = pt.zeros(config.batchSize, config.hiddenLayerSize).to(device)	#padding
				self.outputStateList[layerIndex] = currentOutputSequentialIndex
			outputState = pt.stack(self.outputStateList, dim=1)
		else:
			for sequenceIndex in range(config.sequenceLength):
				sequentialInputState = inputState[:, sequenceIndex, :]
				hiddenState = sequentialInputState
				sequenceIndexMaxLayers = sequenceIndex
				for layerIndex in range(sequenceIndexMaxLayers+1):
					if(layerIndex == sequenceIndexMaxLayers):
						self.hiddenStateLastList[layerIndex] = hiddenState
						self.outputStateList[sequenceIndex] = hiddenState	#or self.outputStateList[layerIndex]
					else:
						previousInput = self.hiddenStateLastList[layerIndex]
						currentInput = hiddenState
						currentOutput = self.processLayer(sequentialInputState, layerIndex, currentInput, previousInput)
						hiddenState = currentOutput
						self.hiddenStateLastList[layerIndex] = currentInput
			outputState = pt.stack(self.outputStateList, dim=1)
			
		if(calculateVocabPredictionHeadLoss):
			predictionScores = self.predictionHead(outputState)
			#used last layer hidden emeddings to predict next word
			#print("predictionScores.shape = ", predictionScores.shape)
			loss = self.lossFunction(predictionScores.view(-1, config.vocab_size), labels.view(-1))
		else:
			if(config.applyIOconversionLayers):
				outputState = pt.reshape(outputState, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
				outputState = self.outputLayer(outputState)
				#outputState = self.activationFunction(outputState)
				outputState = pt.reshape(outputState, (config.batchSize, config.sequenceLength, config.embeddingLayerSize))
				#print("outputState.shape = ", outputState.shape)
			y = outputState
			yHat = inputsEmbeddings
			loss = self.lossFunction(y, yHat)
			predictionScores = None
				
		return loss, predictionScores
	
	def processLayer(self, sequentialInputState, layerIndex, currentInput, previousInput):
		config = self.config
		batchSize = currentInput.shape[0]
		if(recursiveLayers):
			saniLayer = self.saniLayer
		else:
			saniLayer = self.SANIlayers[layerIndex]
		if(retainHiddenEmbeddingStructure):
			combinedInput = pt.stack([previousInput, currentInput], dim=1)	#shape = [batchSize, numberOfHeads, numberOfInputChannels, ..]
			combinedInput = pt.reshape(combinedInput, (batchSize, self.numberOfHeads*self.numberOfInputChannels, 1))
			currentOutput = self.saniLayer(combinedInput)
			currentOutput = pt.reshape(currentOutput, (batchSize, self.numberOfHeads))
		else:
			combinedInput = pt.concat((previousInput, currentInput), dim=1)	#CHECKTHIS
			currentOutput = saniLayer(combinedInput)
		currentOutput = self.activationFunction(currentOutput)
		if(skipLayers):
			currentOutput = pt.add(currentOutput*(1.0-skipLayersDominance), sequentialInputState)
			if(skipLayersNorm):
				currentOutput = self.layerNorm(currentOutput)
		return currentOutput
