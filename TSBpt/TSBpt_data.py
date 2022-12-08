"""TSBpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see TSBpt_main.py

# Usage:
see TSBpt_main.py

# Description:
TSBpt data

"""

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer
import os
from TSBpt_globalDefs import *

if(not useLovelyTensors):
	torch.set_printoptions(profile="full")

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space


def downloadDataset():
	if(useSmallDatasetDebug):
		dataset = load_dataset('nthngdy/oscar-small', 'unshuffled_original_en', cache_dir=downloadCacheFolder)	#unshuffled_deduplicated_en
	else:
		dataset = load_dataset('oscar', 'unshuffled_deduplicated_en', cache_dir=downloadCacheFolder)
	
	return dataset

def preprocessDataset(dataset):
	textData = []
	fileCount = 0
	for sample in tqdm(dataset['train']):
		sample = sample['text'].replace('\n', '')
		textData.append(sample)
		if(len(textData) == numberOfSamplesPerDataFile):
			writeDataFile(fileCount, textData)
			textData = []
			fileCount += 1
	writeDataFile(fileCount, textData)	#remaining data file will be < numberOfSamplesPerDataFile
	
def writeDataFile(fileCount, textData):
	fileName = dataFolder + "/text_" + str(fileCount) + ".txt"
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write('\n'.join(textData))

def trainTokenizer(paths):
	if(useSmallTokenizerTrainNumberOfFiles):
		trainTokenizerNumberOfFilesToUse = 1000	#default 1000	#100: 15 min, 1000: 3.75 hours
	else:
		trainTokenizerNumberOfFilesToUse = len(paths)

	tokenizer = ByteLevelBPETokenizer()

	tokenizer.train(files=paths[:trainTokenizerNumberOfFilesToUse], vocab_size=vocabularySize, min_frequency=2, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

	#os.mkdir(modelFolderName)

	tokenizer.save_model(modelFolderName)
		
	return tokenizer

def loadTokenizer():	
	tokenizer = RobertaTokenizer.from_pretrained(modelFolderName, max_len=sequenceMaxNumTokens)
	return tokenizer

def addMaskTokens(useMLM, inputIDs):
	if(useMLM):
		rand = torch.rand(inputIDs.shape)
		mask_arr = (rand < fractionOfMaskedTokens) * (inputIDs > 2)	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)
		for i in range(inputIDs.shape[0]):
			selection = torch.flatten(mask_arr[i].nonzero()).tolist()
			inputIDs[i, selection] = customMaskTokenID
	else:	
		mask_arr = (inputIDs > 2)	#or * (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)
		for i in range(inputIDs.shape[0]):
			selection = torch.flatten(mask_arr[i].nonzero()).tolist()
			inputIDs[i, selection] = customMaskTokenID
	return inputIDs

def dataFileIndexListContainsLastFile(dataFileIndexList, paths):
	containsDataFileLastSample = False
	for dataFileIndex in dataFileIndexList:
		path = paths[dataFileIndex]
		#print("path = ", path)
		if(str(dataFileLastSampleIndex) in path):
			containsDataFileLastSample = True
	return containsDataFileLastSample
	
class DatasetHDD(torch.utils.data.Dataset):
	def __init__(self, useMLM, dataFileIndexList, paths, tokenizer):
		self.useMLM = useMLM
		self.dataFileIndexList = dataFileIndexList
		self.paths = paths
		self.encodings = None
		self.containsDataFileLastSample = dataFileIndexListContainsLastFile(dataFileIndexList, paths)
		self.tokenizer = tokenizer

	def __len__(self):
		numberOfSamples = len(self.dataFileIndexList)*numberOfSamplesPerDataFile
		if(self.containsDataFileLastSample):
			numberOfSamples = numberOfSamples-numberOfSamplesPerDataFile + numberOfSamplesPerDataFileLast
		return numberOfSamples

	def __getitem__(self, i):
	
		loadNextDataFile = False
		sampleIndex = i // numberOfSamplesPerDataFile
		itemIndexInSample = i % numberOfSamplesPerDataFile
		if(itemIndexInSample == 0):
			loadNextDataFile = True	
		dataFileIndex = self.dataFileIndexList[sampleIndex]
					
		if(loadNextDataFile):
			
			path = self.paths[dataFileIndex]

			with open(path, 'r', encoding='utf-8') as fp:
				lines = fp.read().split('\n')

			sample = self.tokenizer(lines, max_length=sequenceMaxNumTokens, padding='max_length', truncation=True, return_tensors='pt')
			inputIDs = []
			mask = []
			labels = []
			labels.append(sample.input_ids)
			mask.append(sample.attention_mask)
			sampleInputIDs = (sample.input_ids).detach().clone()
			inputIDs.append(addMaskTokens(self.useMLM, sampleInputIDs))
			inputIDs = torch.cat(inputIDs)
			mask = torch.cat(mask)
			labels = torch.cat(labels)
			
			self.encodings = {'inputIDs': inputIDs, 'attentionMask': mask, 'labels': labels}
		
		return {key: tensor[itemIndexInSample] for key, tensor in self.encodings.items()}

def createDataLoader(useMLM, tokenizer, paths, pathIndexMin, pathIndexMax):

	dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
	print("dataFileIndexList = ", dataFileIndexList)
	
	dataset = DatasetHDD(useMLM, dataFileIndexList, paths, tokenizer)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)	#shuffle not supported by DatasetHDD

	return loader

def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py


def getAccuracy(tokenizer, inputIDs, predictionMask, labels, outputs):
	tokenizerNumberTokens = getTokenizerLength(tokenizer)
	
	tokenLogits = outputs.detach()

	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, sequenceMaxNumTokens, accuracyTopN
	
	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	

		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, predictionMask)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 
	else:
		labelsExpanded = torch.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		maskTokenIndexExpanded = torch.unsqueeze(predictionMask, dim=2)
		maskTokenIndexExpanded = maskTokenIndexExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, maskTokenIndexExpanded)	#predictionMask broadcasted to [batchSize, sequenceMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(predictionMask)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(maskTokenIndexExpanded)/accuracyTopN)
	
	#accuracy2 = (torch.mean(comparisonMasked)).cpu().numpy()
	
	return accuracy

