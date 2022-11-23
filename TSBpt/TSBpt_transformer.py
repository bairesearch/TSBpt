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

"""

from TSBpt_globalDefs import *
import TSBpt_data

from TSBpt_modeling_roberta_recursiveLayers import recursiveLayers

if(recursiveLayers):
	from TSBpt_modeling_roberta_recursiveLayers import sharedLayerWeights
	from TSBpt_modeling_roberta_recursiveLayers import sharedLayerWeightsOutput
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
		
accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions


import torch
from transformers import RobertaConfig
if(recursiveLayers):
	from TSBpt_modeling_roberta_recursiveLayers import RobertaForMaskedLM
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
	input_ids = batch['input_ids'].to(device)
	attention_mask = batch['attention_mask'].to(device)
	labels = batch['labels'].to(device)

	outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

	accuracy = getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs)
	loss = outputs.loss
	
	return loss, accuracy


def getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs):
	tokenizerNumberTokens = TSBpt_data.getTokenizerLength(tokenizer)
	
	tokenLogits = (outputs.logits).detach()

	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	
	maskTokenIndex = torch.where(input_ids==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()	

	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	

		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, maskTokenIndex)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(maskTokenIndex)).cpu().numpy() 
	else:
		labelsExpanded = torch.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		maskTokenIndexExpanded = torch.unsqueeze(maskTokenIndex, dim=2)
		maskTokenIndexExpanded = maskTokenIndexExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#maskTokenIndex broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, maskTokenIndexExpanded)	#maskTokenIndex broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(maskTokenIndex)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(maskTokenIndexExpanded)/accuracyTopN)
	
	#accuracy2 = (torch.mean(comparisonMasked)).cpu().numpy()
	
	return accuracy
