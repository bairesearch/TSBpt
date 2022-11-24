"""TSBpt_RNNmodel_recursiveLayers.py

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

recursiveLayers = True

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
		self.N = batchSize
		self.L = sequenceLength
		self.D = bidirectional	#default: 1 #2 if bidirectional=True otherwise 1
		self.hiddenLayerSize = hiddenLayerSize
		self.embeddingLayerSize = embeddingLayerSize	#input token embedding size (equivalent to roberta hidden_size)
		self.pad_token_id = 1	#default=1 #https://huggingface.co/transformers/v2.11.0/model_doc/roberta.html 
		self.applyIOconversionLayers = False
		if(embeddingLayerSize != hiddenLayerSize):
			self.applyIOconversionLayers = True
		
class RNNrecursiveLayersModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.word_embeddings = nn.Embedding(config.vocab_size, config.embeddingLayerSize, padding_idx=config.pad_token_id)
		self.rnnLayer = nn.RNN(input_size=config.hiddenLayerSize, hidden_size=config.hiddenLayerSize, num_layers=config.num_layers, batch_first=True)
		if(config.applyIOconversionLayers):
			self.inputLayer = nn.Linear(config.embeddingLayerSize, config.hiddenLayerSize)
			self.outputLayer = nn.Linear(config.hiddenLayerSize, config.embeddingLayerSize)
		self.lossFunction = CrossEntropyLoss()	#CHECKTHIS
		
	def forward(self, labels, device):
		
		config = self.config
		
		inputsEmbeddings = self.word_embeddings(labels)
		if(config.applyIOconversionLayers):
			inputsEmbeddingsReshaped = pt.reshape(inputsEmbeddings, (config.N*config.L, config.embeddingLayerSize))
			inputState = self.inputLayer(inputsEmbeddingsReshaped)
			inputState = pt.reshape(inputState, (config.N, config.L, config.hiddenLayerSize))
		else:
			inputState = inputsEmbeddings
		
		hn = pt.zeros(config.num_layers*config.D, config.N, config.hiddenLayerSize).to(device)	#randn
		
		hiddenState = inputState
		if(recursiveLayers):
			for layerIndex in range(config.numberOfRecursiveLayers):
				hiddenState, hn = self.rnnLayer(hiddenState, hn)
				#print("hiddenState = ", hiddenState)
		else:
			hiddenState, hn = rnnLayer(hiddenState, hn)
		outputState = hiddenState
		
		if(config.applyIOconversionLayers):
			outputState = pt.reshape(outputState, (config.N*config.L, config.hiddenLayerSize))
			y = self.outputLayer(outputState)
			y = pt.reshape(y, (config.N, config.L, config.embeddingLayerSize))
		else:
			y = outputState
		yHat = inputsEmbeddings

		loss = self.lossFunction(y, yHat)
		
		return loss
	
