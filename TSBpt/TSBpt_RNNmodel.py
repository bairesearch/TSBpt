"""TSBpt_RNNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBpt_main.py

# Usage:
see TSBpt_main.py

# Description:
TSBpt RNN model recursiveLayers

"""

import torch as pt
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import gelu

recursiveLayers = True
calculateVocabPredictionHeadLoss = True	#apply loss to vocubulary predictions (rather than embedding predictions)
applyIOconversionLayers = True	#ensure input embeddings are positive

class RNNrecursiveLayersConfig():
	def __init__(self, vocabularySize, numberOfHiddenLayers, batchSize, sequenceLength, bidirectional, hiddenLayerSize, embeddingLayerSize):
		#https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
		self.vocab_size = vocabularySize
		if(recursiveLayers):
			self.num_layers = 1
			self.numberOfRecursiveLayers = numberOfHiddenLayers
		else:
			self.num_layers = numberOfHiddenLayers	#Default: 1	#Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results
			if(num_layers == 1):
				print("RNNrecursiveLayersConfig warning: !recursiveLayers && (num_layers == 1)")
		self.batchSize = batchSize
		self.sequenceLength = sequenceLength
		self.bidirectional = bidirectional	#default: 1 #2 if bidirectional=True otherwise 1
		self.hiddenLayerSize = hiddenLayerSize
		self.embeddingLayerSize = embeddingLayerSize	#input token embedding size (equivalent to roberta hidden_size)
		self.pad_token_id = 1	#default=1 #https://huggingface.co/transformers/v2.11.0/model_doc/roberta.html 
		self.applyIOconversionLayers = False
		if(applyIOconversionLayers):
			self.applyIOconversionLayers = True
		else:
			if(embeddingLayerSize != hiddenLayerSize):
				print("error: !applyIOconversionLayers and (embeddingLayerSize != hiddenLayerSize)")
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
						
class RNNrecursiveLayersModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.word_embeddings = nn.Embedding(config.vocab_size, config.embeddingLayerSize, padding_idx=config.pad_token_id)
		self.rnnLayer = nn.RNN(input_size=config.hiddenLayerSize, hidden_size=config.hiddenLayerSize, num_layers=config.num_layers, batch_first=True)
		if(config.applyIOconversionLayers):
			self.inputLayer = nn.Linear(config.embeddingLayerSize, config.hiddenLayerSize)
			self.outputLayer = nn.Linear(config.hiddenLayerSize, config.embeddingLayerSize)
		self.activationFunction = pt.nn.ReLU()
		if(calculateVocabPredictionHeadLoss):
			self.lossFunction = CrossEntropyLoss()
		else:
			self.lossFunction = MSELoss()
		self.predictionHead = ModelVocabPredictionHead(config)
		
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
		
		hn = pt.zeros(config.num_layers*config.bidirectional, config.batchSize, config.hiddenLayerSize).to(device)	#randn	#https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
		
		hiddenState = inputState
		if(recursiveLayers):
			for layerIndex in range(config.numberOfRecursiveLayers):
				hiddenState, hn = self.rnnLayer(hiddenState, hn)
				#print("hiddenState = ", hiddenState)
		else:
			hiddenState, hn = rnnLayer(hiddenState, hn)
		outputState = hiddenState
		
		if(calculateVocabPredictionHeadLoss):
			predictionScores = self.predictionHead(outputState)
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
	
