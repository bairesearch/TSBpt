"""TSBpt_transformer.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBpt_main.py

# Usage:
see TSBpt_main.py

# Description:
TSBpt transformer

- recursiveLayers:numberOfHiddenLayers=6 supports approximately 2^6 tokens per sentence (contextual window = 512 tokens)

See RobertaForMaskedLM tutorial; 
	https://huggingface.co/blog/how-to-train
	https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6
	
"""

from TSBpt_globalDefs import *
import TSBpt_data

from TSBpt_transformerModel import recursiveLayers

if(recursiveLayers):
	from TSBpt_transformerModel import sharedLayerWeights
	from TSBpt_transformerModel import sharedLayerWeightsOutput
	recursiveLayersNormaliseNumParameters = False	#default: True	#optional	#if use recursiveLayers normalise/equalise num of parameters with respect to !recursiveLayers
	if(recursiveLayersNormaliseNumParameters):
		recursiveLayersNormaliseNumParametersIntermediate = True	#normalise intermediateSize parameters also
else:
	recursiveLayersNormaliseNumParameters = False	#mandatory
	
if(not usePretainedModelDebug):

	if(useSingleHiddenLayerDebug):
		numberOfHiddenLayers = 1
	else:
		numberOfHiddenLayers = 6	#default: 6

	hiddenLayerSize = 768	#default: 768
	numberOfAttentionHeads = 12	#default: 12
	intermediateSize = 3072	#default: 3072

	if(recursiveLayers):
		#same model size irrespective of useSingleHiddenLayerDebug
		if(recursiveLayersNormaliseNumParameters):
			if(sharedLayerWeights):
				if(sharedLayerWeightsOutput):
					if(recursiveLayersNormaliseNumParametersIntermediate):
						hiddenLayerSizeMultiplier = (7/4)	#model size = 249MB	
						#hiddenLayerSizeMultiplier = (5/3)	#~230MB	
					else:
						hiddenLayerSizeMultiplier = 2	#model size = ~255MB
				else:
					if(recursiveLayersNormaliseNumParametersIntermediate):
						hiddenLayerSizeMultiplier = (4/3)	#model size = 273MB
					else:
						hiddenLayerSizeMultiplier = 1.5	#model size = ~255MB
			else:
				hiddenLayerSizeMultiplier = (7/4)	#model size = ~250MB	#optimisation failure observed
				#hiddenLayerSizeMultiplier = (11/6)	#model size = ~265MB	#optimisation failure observed
				#hiddenLayerSizeMultiplier = 2.0	#model size = ~280MB	#optimisation failure observed
					
			hiddenLayerSize = round(hiddenLayerSize*hiddenLayerSizeMultiplier)
			numberOfAttentionHeads = round(numberOfAttentionHeads*hiddenLayerSizeMultiplier)	#or: round(numberOfAttentionHeads)
			if(recursiveLayersNormaliseNumParametersIntermediate):
				intermediateSize = round(intermediateSize*hiddenLayerSizeMultiplier)
			print("hiddenLayerSize = ", hiddenLayerSize)
			print("numberOfAttentionHeads = ", numberOfAttentionHeads)
			print("intermediateSize = ", intermediateSize)
		else:
			if(sharedLayerWeights):
				if(sharedLayerWeightsOutput):
					pass	#model size = ~120MB
				else:
					pass	#model size = 176.7MB
			else:
				pass	#model size = 120.4MB
	else:
		if(useSingleHiddenLayerDebug):
			pass	#model size = 120.4MB
		else:
			pass	#model size = 255.6MB
		

import torch
from transformers import RobertaConfig
if(recursiveLayers):
	from TSBpt_transformerModel import RobertaForMaskedLM
else:
	from transformers import RobertaForMaskedLM
	


def createModel():
	print("creating new model")	
	config = RobertaConfig(
		vocab_size=vocabularySize,  #sync with tokenizer vocab_size
		max_position_embeddings=(sequenceMaxNumTokens+2),
		hidden_size=hiddenLayerSize,
		num_attention_heads=numberOfAttentionHeads,
		num_hidden_layers=numberOfHiddenLayers,
		intermediate_size=intermediateSize,
		type_vocab_size=1
	)
	print("config.pad_token_id = ", config.pad_token_id)
	model = RobertaForMaskedLM(config)
	return model

def loadModel():
	print("loading existing model")
	model = RobertaForMaskedLM.from_pretrained(modelFolderName, local_files_only=True)
	return model
	
def saveModel(model):
	model.save_pretrained(modelFolderName)

def propagate(device, model, tokenizer, batch):
	inputIDs = batch['inputIDs'].to(device)
	attentionMask = batch['attentionMask'].to(device)
	labels = batch['labels'].to(device)
	
	outputs = model(inputIDs, attention_mask=attentionMask, labels=labels)

	predictionMask = torch.where(inputIDs==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()	
	accuracy = TSBpt_data.getAccuracy(tokenizer, inputIDs, predictionMask, labels, outputs.logits)
	loss = outputs.loss
	
	return loss, accuracy

