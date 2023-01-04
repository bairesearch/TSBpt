"""SBNLPpt_SANImodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see SBNLPpt_main.py

# Usage:
see SBNLPpt_main.py

# Description:
SBNLPpt SANI model recursiveLayers

"""

import torch as pt
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import gelu
import numpy as np

parallelProcessLayers = True
generatePredictionsForEverySubsequence = False	#uses high amount of GPU ram

recursiveLayers = True
skipLayers = True
if(skipLayers):
	skipLayersDominance = 0.0	#0.9	#sequentialInputState preservation bias (direct recursive/loop connection) as signal is propagated to higher layers
	skipLayersNorm = True

processLayerRetainHiddenEmbeddingStructure = False	#experimental	#do not mix hidden embeddings (for every unit/neuron in hidden layer, calculate new value based on current and previous value)
processLayerSubtractEmbeddings = False #experimental

biologicalSimulationNoMultilayerBackprop = False
if(biologicalSimulationNoMultilayerBackprop):
	calculateVocabPredictionHeadLoss = False	#calculate loss for embedding predictions
	applyIOconversionLayers = True	#CHECKTHIS - gradient will currently backpropagate through io conversion layers (for strict single layer backprop set to False or prevent gradient flow)
else:
	calculateVocabPredictionHeadLoss = True	#calculate loss for vocubulary predictions
	applyIOconversionLayers = True
	
if(applyIOconversionLayers):
	applyIOconversionLayersInput = True	#ensure input embeddings are positive
	applyIOconversionLayersOutput = True
else:
	applyIOconversionLayersInput = False
	applyIOconversionLayersOutput = False
		
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

		if(recursiveLayers):
			self.saniLayer = self.generateSANIlayer(config.hiddenLayerSize)
		else:
			self.SANIlayers = []
			for layerIndex in range(config.num_layers):
				saniLayer = generateSANIlayer(self, config.hiddenLayerSize)
				self.SANIlayers.append(saniLayer)
		if(applyIOconversionLayers):
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
		self.vocabPredictionHead = ModelVocabPredictionHead(config)
		self.predictiveTokenOffset = 1	#no prediction is generated for first and last token in sentence (requires at least 2 tokens to generate a prediction)
		
	def generateSANIlayer(self, hiddenLayerSize):
		if(processLayerSubtractEmbeddings):
			self.numberOfInputChannels = 1
		else:
			self.numberOfInputChannels = 2
		#https://stackoverflow.com/questions/58374980/run-multiple-models-of-an-ensemble-in-parallel-with-pytorch/58389075#58389075
		if(processLayerRetainHiddenEmbeddingStructure):
			self.numberOfHeads = hiddenLayerSize
			#input shape = B x (numberOfInputChannels * numberOfHeads) x 1
			#output shape = B x (1 x numberOfHeads) x 1
			saniLayer = pt.nn.Conv1d(numberOfSequentialInputStates*self.numberOfHeads, 1*self.numberOfHeads, kernel_size=1, groups=self.numberOfHeads)
		else:
			saniLayer =	pt.nn.Linear(hiddenLayerSize*self.numberOfInputChannels, hiddenLayerSize)	#CHECKTHIS
		return saniLayer
				
	def forward(self, labels, attentionMask, device):
		
		config = self.config
		
		predictionLabels = labels
		inputEmbeddings = self.word_embeddings(labels)
		if(applyIOconversionLayersInput):
			inputState = pt.reshape(inputEmbeddings, (config.batchSize*config.sequenceLength, config.embeddingLayerSize))
			inputState = self.inputLayer(inputState)
			inputState = self.activationFunction(inputState)
			inputState = pt.reshape(inputState, (config.batchSize, config.sequenceLength, config.hiddenLayerSize))
		else:
			inputState = inputEmbeddings
		
		if(parallelProcessLayers):	
			self.processLayersVectorised(inputState, predictionLabels, device)
		else:
			self.processLayersStandard(inputState, predictionLabels, device)
		
		outputState = pt.stack(list(self.outputStateArray), dim=1)
		predictionLabels = pt.stack(list(self.predictionLabelsArray), dim=1)
		predictionLabels = predictionLabels.type(pt.LongTensor).to(device)	#object should already be on gpu
		predictionMask = pt.stack(list(self.predictionMaskArray), dim=1)
		predictionMask = predictionMask.type(pt.LongTensor).to(device)	#object should already be on gpu
		if(generatePredictionsForEverySubsequence):
			if(parallelProcessLayers):
				outputState = pt.reshape(outputState, (outputState.shape[0], outputState.shape[1]*outputState.shape[2], outputState.shape[3]))	#pt.flatten(outputState, start_dim=1, end_dim=2)
				predictionLabels = pt.reshape(predictionLabels, (predictionLabels.shape[0], predictionLabels.shape[1]*predictionLabels.shape[2]))	#pt.flatten(predictionLabels, start_dim=1, end_dim=2)
				predictionMask = pt.reshape(predictionMask, (predictionMask.shape[0], predictionMask.shape[1]*predictionMask.shape[2]))
				
		if(applyIOconversionLayersOutput):
			if(calculateVocabPredictionHeadLoss):
				predictionScores = self.vocabPredictionHead(outputState)
				#used last layer hidden emeddings to predict next word
				loss = self.lossFunction(predictionScores.view(-1, config.vocab_size), predictionLabels.view(-1))
			else:
				outputState = pt.reshape(outputState, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
				outputState = self.outputLayer(outputState)
				#outputState = self.activationFunction(outputState)	#do not add activationFunction as outputs should be positive/negative (to match predictionEmbeddings)
				outputState = pt.reshape(outputState, (config.batchSize, config.sequenceLength, config.embeddingLayerSize))
		if(not calculateVocabPredictionHeadLoss):
			predictionEmbeddings = self.word_embeddings(predictionLabels)
			y = outputState
			yHat = predictionEmbeddings
			loss = self.lossFunction(y, yHat)
			predictionScores = None

		predictionMask = predictionMask*attentionMask
	
		return loss, predictionScores, predictionMask

	def processLayersVectorised(self, inputState, predictionLabels, device):
		predictiveTokenOffset = self.predictiveTokenOffset
		config = self.config

		self.outputStateArray = np.empty([config.num_layers], dtype=object)
		self.predictionLabelsArray = np.empty([config.num_layers], dtype=object)
		self.predictionMaskArray = np.empty([config.num_layers], dtype=object)
						
		#first two tokens in window generate a hidden representation for the second token (pad output states of 1st token with zeros), and are used to predict third token
		for sequenceIndex in range(predictiveTokenOffset):
			if(generatePredictionsForEverySubsequence):
				self.outputStateArray[sequenceIndex] = pt.zeros(inputState.shape).to(device)
				self.predictionLabelsArray[sequenceIndex] = pt.zeros(predictionLabels.shape).to(device)
				self.predictionMaskArray[sequenceIndex] = pt.zeros(predictionLabels.shape).to(device)
				self.outputStateArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(inputState.shape).to(device)
				self.predictionLabelsArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(predictionLabels.shape).to(device)
				self.predictionMaskArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(predictionLabels.shape).to(device)
			else:
				self.outputStateArray[sequenceIndex] = pt.zeros(inputState.shape[0], inputState.shape[2]).to(device)
				self.predictionLabelsArray[sequenceIndex] = pt.zeros(predictionLabels.shape[0]).to(device)
				self.predictionMaskArray[sequenceIndex] = pt.zeros(predictionLabels.shape[0]).to(device)
				self.outputStateArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(inputState.shape[0], inputState.shape[2]).to(device)
				self.predictionLabelsArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(predictionLabels.shape[0]).to(device)
				self.predictionMaskArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(predictionLabels.shape[0]).to(device)

		hiddenState = inputState
		for layerIndex in range(0, config.sequenceLength):
			if((layerIndex >= predictiveTokenOffset) and (layerIndex < config.sequenceLength-predictiveTokenOffset)):
				sequenceIndex = layerIndex
				blankSequentialHiddenStates = pt.zeros(hiddenState.shape[0], sequenceIndex, hiddenState.shape[2]).to(device)
				sequentialInputState = pt.reshape(hiddenState, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
				previousInput = pt.cat((blankSequentialHiddenStates, hiddenState[:, sequenceIndex-predictiveTokenOffset:-predictiveTokenOffset]), dim=1)
				currentInput = pt.cat((blankSequentialHiddenStates, hiddenState[:, sequenceIndex:]), dim=1)
				previousInput = pt.reshape(previousInput, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
				currentInput = pt.reshape(currentInput, (config.batchSize*config.sequenceLength, config.hiddenLayerSize))
				currentOutput = self.processLayer(sequentialInputState, layerIndex, currentInput, previousInput)
				currentOutput = pt.reshape(hiddenState, (config.batchSize, config.sequenceLength, config.hiddenLayerSize))
				hiddenState = currentOutput
				if(generatePredictionsForEverySubsequence):
					#last output does not provide a prediction
					currentOutputLayerIndex1 = currentOutput[:, :-predictiveTokenOffset]
					currentOutputLayerIndex2 = pt.zeros([currentOutput.shape[0], predictiveTokenOffset, currentOutput.shape[2]]).to(device)
					currentOutputLayerIndex = pt.cat((currentOutputLayerIndex1, currentOutputLayerIndex2), dim=1)	
					predictionLabelsLayerIndex1 = pt.zeros([predictionLabels.shape[0], sequenceIndex]).to(device)
					predictionLabelsLayerIndex2 = predictionLabels[:, sequenceIndex+predictiveTokenOffset:config.sequenceLength]
					predictionLabelsLayerIndex3 = pt.zeros([predictionLabels.shape[0], predictiveTokenOffset]).to(device)
					predictionLabelsLayerIndex = pt.cat((predictionLabelsLayerIndex1, predictionLabelsLayerIndex2, predictionLabelsLayerIndex3), dim=1)	
					predictionMaskLayerIndex1 = pt.zeros([predictionLabels.shape[0], sequenceIndex]).to(device)
					predictionMaskLayerIndex2 = pt.ones(predictionLabelsLayerIndex2.shape).to(device)
					predictionMaskLayerIndex3 = pt.zeros([predictionLabels.shape[0], predictiveTokenOffset]).to(device)
					predictionMaskLayerIndex = pt.cat((predictionMaskLayerIndex1, predictionMaskLayerIndex2, predictionMaskLayerIndex3), dim=1)
				else:
					currentOutputLayerIndex = currentOutput[:, sequenceIndex]	#aka currentOutputSequentialIndex
					predictionLabelsLayerIndex = predictionLabels[:, sequenceIndex+predictiveTokenOffset]	#aka predictionLabelsSequentialIndex
					predictionMaskLayerIndex = pt.ones(predictionLabelsLayerIndex.shape).to(device)
				self.outputStateArray[layerIndex] = currentOutputLayerIndex
				self.predictionLabelsArray[layerIndex] = predictionLabelsLayerIndex
				self.predictionMaskArray[layerIndex] = predictionMaskLayerIndex
	
	def processLayersStandard(self, inputState, predictionLabels, device):
		predictiveTokenOffset = self.predictiveTokenOffset
		config = self.config

		if(generatePredictionsForEverySubsequence):
			self.outputStateArray = np.empty([config.num_layers, config.num_layers], dtype=object)
			self.predictionLabelsArray = np.empty([config.num_layers, config.num_layers], dtype=object)
			self.predictionMaskArray = np.empty([config.num_layers, config.num_layers], dtype=object)
		else:
			self.outputStateArray = np.empty([config.num_layers], dtype=object)
			self.predictionLabelsArray = np.empty([config.num_layers], dtype=object)
			self.predictionMaskArray = np.empty([config.num_layers], dtype=object)
		self.hiddenStateLastList = [None]*config.num_layers	#last activated hidden state of each layer
		for layerIndex in range(config.num_layers):
			self.hiddenStateLastList[layerIndex] = pt.zeros([config.batchSize, config.hiddenLayerSize])
				
		#first two tokens in window generate a hidden representation for the second token (pad output states of 1st token with zeros), and are used to predict third token
		for sequenceIndex in range(config.sequenceLength):
			sequentialInputState = inputState[:, sequenceIndex, :]
			hiddenState = sequentialInputState
			for layerIndex in range(config.sequenceLength):
				if(generatePredictionsForEverySubsequence):
					if(sequenceIndex < predictiveTokenOffset):
						self.outputStateArray[sequenceIndex][layerIndex] = pt.zeros(hiddenState.shape).to(device)
						self.predictionLabelsArray[sequenceIndex][layerIndex] = pt.zeros(hiddenState.shape[0]).to(device)	
						self.predictionMaskArray[sequenceIndex][layerIndex] = pt.zeros(hiddenState.shape[0]).to(device)	
						self.outputStateArray[config.sequenceLength-sequenceIndex-1][layerIndex] = pt.zeros(hiddenState.shape).to(device)
						self.predictionLabelsArray[config.sequenceLength-sequenceIndex-1][layerIndex] = pt.zeros(hiddenState.shape[0]).to(device)
						self.predictionMaskArray[config.sequenceLength-sequenceIndex-1][layerIndex] = pt.zeros(hiddenState.shape[0]).to(device)
					if(layerIndex < predictiveTokenOffset):
						self.outputStateArray[sequenceIndex][layerIndex] = pt.zeros(hiddenState.shape).to(device)
						self.predictionLabelsArray[sequenceIndex][layerIndex] = pt.zeros(hiddenState.shape[0]).to(device)
						self.predictionMaskArray[sequenceIndex][layerIndex] = pt.zeros(hiddenState.shape[0]).to(device)	
						self.outputStateArray[sequenceIndex][config.sequenceLength-layerIndex-1] = pt.zeros(hiddenState.shape).to(device)
						self.predictionLabelsArray[sequenceIndex][config.sequenceLength-layerIndex-1] = pt.zeros(hiddenState.shape[0]).to(device)
						self.predictionMaskArray[sequenceIndex][config.sequenceLength-layerIndex-1] = pt.zeros(hiddenState.shape[0]).to(device)
				else:
					if(sequenceIndex < predictiveTokenOffset):
						self.outputStateArray[sequenceIndex] = pt.zeros(hiddenState.shape).to(device)
						self.predictionLabelsArray[sequenceIndex] = pt.zeros(hiddenState.shape[0]).to(device)
						self.predictionMaskArray[sequenceIndex] = pt.zeros(hiddenState.shape[0]).to(device)
						self.outputStateArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(hiddenState.shape).to(device)
						self.predictionLabelsArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(hiddenState.shape[0]).to(device)
						self.predictionMaskArray[config.sequenceLength-sequenceIndex-1] = pt.zeros(hiddenState.shape[0]).to(device)

		for sequenceIndex in range(config.sequenceLength-predictiveTokenOffset):
			sequentialInputState = inputState[:, sequenceIndex, :]
			hiddenState = sequentialInputState
			sequenceIndexMaxLayers = sequenceIndex	#number of hidden layers used to encode the subsequence
			for layerIndex in range(config.sequenceLength):	
				if(layerIndex < sequenceIndexMaxLayers+1):
					if(generatePredictionsForEverySubsequence):
						if(layerIndex >= predictiveTokenOffset):
							self.outputStateArray[sequenceIndex][layerIndex] = hiddenState
							self.predictionLabelsArray[sequenceIndex][layerIndex] = predictionLabels[:, sequenceIndex+predictiveTokenOffset]
							self.predictionMaskArray[sequenceIndex][layerIndex] = pt.ones([config.batchSize]).to(device)
					else:
						if(layerIndex == sequenceIndexMaxLayers):
							self.outputStateArray[sequenceIndex] = hiddenState
							self.predictionLabelsArray[sequenceIndex] = predictionLabels[:, sequenceIndex+predictiveTokenOffset]
							self.predictionMaskArray[sequenceIndex] = pt.ones([config.batchSize]).to(device)

					if(layerIndex == sequenceIndexMaxLayers):
						self.hiddenStateLastList[layerIndex] = hiddenState
					else:
						previousInput = self.hiddenStateLastList[layerIndex]
						currentInput = hiddenState
						currentOutput = self.processLayer(sequentialInputState, layerIndex, currentInput, previousInput)
						hiddenState = currentOutput
						self.hiddenStateLastList[layerIndex] = currentInput
				else:
					if(generatePredictionsForEverySubsequence):
						self.outputStateArray[sequenceIndex][layerIndex] = pt.zeros(hiddenState.shape).to(device)
						self.predictionLabelsArray[sequenceIndex][layerIndex] = pt.zeros([hiddenState.shape[0]]).to(device)
						self.predictionMaskArray[sequenceIndex][layerIndex] = pt.zeros([hiddenState.shape[0]]).to(device)

		if(generatePredictionsForEverySubsequence):
			self.outputStateArray = np.reshape(self.outputStateArray, (self.outputStateArray.shape[0]*self.outputStateArray.shape[1]))
			self.predictionLabelsArray = np.reshape(self.predictionLabelsArray, (self.predictionLabelsArray.shape[0]*self.predictionLabelsArray.shape[1]))
			self.predictionMaskArray = np.reshape(self.predictionMaskArray, (self.predictionMaskArray.shape[0]*self.predictionMaskArray.shape[1]))
	
	def processLayer(self, sequentialInputState, layerIndex, currentInput, previousInput):
		config = self.config
		batchSize = currentInput.shape[0]
		if(recursiveLayers):
			saniLayer = self.saniLayer
		else:
			saniLayer = self.SANIlayers[layerIndex]
		if(processLayerSubtractEmbeddings):
			combinedInput = pt.abs(pt.subtract(previousInput, currentInput))
			currentOutput = self.saniLayer(combinedInput)
		else:
			if(processLayerRetainHiddenEmbeddingStructure):
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
