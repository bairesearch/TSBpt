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

#torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full")

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space

from TSBpt_globalDefs import *

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

def addMaskTokens(input_ids):
	rand = torch.rand(input_ids.shape)
	mask_arr = (rand < fractionOfMaskedTokens) * (input_ids > 2)	#or * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
	for i in range(input_ids.shape[0]):
		selection = torch.flatten(mask_arr[i].nonzero()).tolist()
		input_ids[i, selection] = customMaskTokenID
	return input_ids

def dataFileIndexListContainsLastFile(dataFileIndexList, paths):
	containsDataFileLastSample = False
	for dataFileIndex in dataFileIndexList:
		path = paths[dataFileIndex]
		#print("path = ", path)
		if(str(dataFileLastSampleIndex) in path):
			containsDataFileLastSample = True
	return containsDataFileLastSample
	
class DatasetHDD(torch.utils.data.Dataset):
	def __init__(self, dataFileIndexList, paths, tokenizer):
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

			#sample = tokenizer(lines, max_length=sequenceMaxNumTokens, padding='max_length', truncation=True)
			#labels = torch.tensor([x for x in sample['input_ids']])
			#mask = torch.tensor([x for x in sample['attention_mask']])
			#input_ids = labels.detach().clone() 
			#input_ids = addMaskTokens(input_ids)
			
			sample = self.tokenizer(lines, max_length=sequenceMaxNumTokens, padding='max_length', truncation=True, return_tensors='pt')
			input_ids = []
			mask = []
			labels = []
			labels.append(sample.input_ids)
			mask.append(sample.attention_mask)
			sample_input_ids = (sample.input_ids).detach().clone()
			input_ids.append(addMaskTokens(sample_input_ids))
			input_ids = torch.cat(input_ids)
			mask = torch.cat(mask)
			labels = torch.cat(labels)
			
			self.encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
		
		return {key: tensor[itemIndexInSample] for key, tensor in self.encodings.items()}

def createDataLoader(tokenizer, paths, pathIndexMin, pathIndexMax):

	dataFileIndexList = list(range(pathIndexMin, pathIndexMax))
	print("dataFileIndexList = ", dataFileIndexList)
	
	dataset = DatasetHDD(dataFileIndexList, paths, tokenizer)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)	#shuffle not supported by DatasetHDD

	return loader

def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py



