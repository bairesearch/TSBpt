"""TSBpt-main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n transformersenv
source activate transformersenv
conda install python=3.7	[DOESNTWORK: conda install python (python-3.10.6) because transformers not found]
pip install datasets
pip install transfomers==4.23.1
pip install torch
pip install torchsummary

# Usage:
source activate transformersenv
python TSBpt-main.py

# Description:
TSBpt main - Transformer Syntactic Bias (TSB): trains a RoBERTa transformer with a number of syntactic inductive biases:
- recursiveLayers
	- Roberta number of layers = 6 supports approximately 2^6 words per sentence (contextual window = 512 tokens)

See RobertaForMaskedLM tutorial; 
	https://huggingface.co/blog/how-to-train
	https://towardsdatascience.com/how-to-train-a-bert-model-from-scratch-72cfce554fc6

"""

from modeling_roberta_recursiveLayers import recursiveLayers

#user config vars:

useSmallDatasetDebug = False
useSingleHiddenLayerDebug = False
usePretainedModelDebug = False	#executes stateTestDataset only
useSmallBatchSizeDebug = False

useSmallTokenizerTrainNumberOfFiles = True	#used during rapid testing only (FUTURE: assign est 80 hours to perform full tokenisation train)

statePreprocessDataset = False	#only required once
stateTrainTokenizer = False	#only required once
stateTrainDataset = False
stateTestDataset = True	#requires reserveValidationSet

if(recursiveLayers):
	from modeling_roberta_recursiveLayers import sharedLayerWeights
	from modeling_roberta_recursiveLayers import sharedLayerWeightsOutput
	recursiveLayersNormaliseNumParameters = False	#default: True	#optional	#if use recursiveLayers normalise/equalise num of parameters with respect to !recursiveLayers
	if(recursiveLayersNormaliseNumParameters):
		recursiveLayersNormaliseNumParametersIntermediate = True	#normalise intermediateSize parameters also
		recursiveLayersNormaliseNumParametersDebug = False	#normalise hiddenLayerSize/numberOfAttentionHeads with respect to orig numberOfHiddenLayers instead of number of parameters (model size)
else:
	recursiveLayersNormaliseNumParameters = False	#mandatory
	

trainStartEpoch = 0	#start epoch of training (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfEpochs = 1	#default: 10	#number of epochs to train (for production typically train x epochs at a time)
trainStartDataFile = 0	#default: 0	#start data file to train (if continuing a training regime set accordingly >0)	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
trainNumberOfDataFiles = 100	#default: -1 (all)	#number of data files to train (for production typically train x dataFiles at a time)	#< numberOfDataFiles (30424) * trainSplitFraction
testNumberOfDataFiles = 10	#default: -1 (all)

if(not usePretainedModelDebug):

	if(useSingleHiddenLayerDebug):
		numberOfHiddenLayers = 1
	else:
		numberOfHiddenLayers = 6	#default: 6

	vocabularySize = 30522	#default: 30522
	hiddenLayerSize = 768	#default: 768
	numberOfAttentionHeads = 12	#default: 12
	intermediateSize = 3072	#default: 3072

	if(recursiveLayers):
		#same model size irrespective of useSingleHiddenLayerDebug
		if(recursiveLayersNormaliseNumParameters):
			if(recursiveLayersNormaliseNumParametersDebug):
				hiddenLayerSizeMultiplier = numberOfHiddenLayers	#model size = 1.7GB
			else:
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
		
reserveValidationSet = True	#reserves a fraction of the data for validation
trainSplitFraction = 0.9	#90% train data, 10% test data

if(recursiveLayersNormaliseNumParameters):
	if(recursiveLayersNormaliseNumParametersDebug):
		batchSize = 1
		learningRate = 1.25e5 #1e-4/8=0.0000125
	else:
		batchSize = 8	#recursiveLayersNormaliseNumParameters uses ~16x more GPU RAM than !recursiveLayersNormaliseNumParameters, and ~2x more GPU RAM than !recursiveLayers
		learningRate = 1e-4
else:
	batchSize = 8  #default: 16	#8 and 16 train at approx same rate (16 uses more GPU ram)	#depends on GPU RAM	#with 12GB GPU RAM, batchSize max = 16
	learningRate = 1e-4
fractionOfMaskedTokens = 0.15

if(useSmallBatchSizeDebug):
	batchSize = 1	#use small batch size to enable simultaneous execution (GPU ram limited) 
	
numberOfSamplesPerDataFile = 10000
numberOfSamplesPerDataFileLast = 423
dataFileLastSampleIndex = 30423

#storage location vars (requires 4TB harddrive);
downloadCacheFolder = '/media/user/datasets/cache'
dataFolder = '/media/user/datasets/data'
modelFolderName = 'model'

modelSaveNumberOfBatches = 1000	#resave model after x training batches

accuracyTopN = 1	#default: 1	#>= 1	#calculates batch accuracy based on top n dictionary predictions

from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import os
from transformers import RobertaTokenizer
import torch
from transformers import RobertaConfig
if(recursiveLayers):
	from modeling_roberta_recursiveLayers import RobertaForMaskedLM
else:
	from transformers import RobertaForMaskedLM
from transformers import AdamW
from transformers import pipeline
from torchsummary import summary
import math 

#torch.set_printoptions(threshold=10_000)
torch.set_printoptions(profile="full")

#store models to large datasets partition cache folder (not required)
#os.environ['TRANSFORMERS_CACHE'] = '/media/user/datasets/models/'	#select partition with 3TB+ disk space

transformerMaxNumTokens = 512
customMaskTokenID = 4	#3


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

	os.mkdir(modelFolderName)

	tokenizer.save_model(modelFolderName)
		
	return tokenizer

def loadTokenizer():	
	tokenizer = RobertaTokenizer.from_pretrained(modelFolderName, max_len=transformerMaxNumTokens)
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
	def __init__(self, dataFileIndexList, paths):
		self.dataFileIndexList = dataFileIndexList
		self.paths = paths
		self.encodings = None
		self.containsDataFileLastSample = dataFileIndexListContainsLastFile(dataFileIndexList, paths)

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

			#sample = tokenizer(lines, max_length=transformerMaxNumTokens, padding='max_length', truncation=True)
			#labels = torch.tensor([x for x in sample['input_ids']])
			#mask = torch.tensor([x for x in sample['attention_mask']])
			#input_ids = labels.detach().clone() 
			#input_ids = addMaskTokens(input_ids)
			
			sample = tokenizer(lines, max_length=transformerMaxNumTokens, padding='max_length', truncation=True, return_tensors='pt')
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
	
	dataset = DatasetHDD(dataFileIndexList, paths)

	loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)	#shuffle not supported by DatasetHDD

	return loader

def continueTrainingModel():
	continueTrain = False
	if((trainStartEpoch > 0) or (trainStartDataFile > 0)):
		continueTrain = True	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
	return continueTrain	

def trainDataset(tokenizer, paths):

	if(continueTrainingModel()):
		print("loading existing model")
		model = RobertaForMaskedLM.from_pretrained(modelFolderName, local_files_only=True)
	else:
		print("creating new model")
		config = RobertaConfig(
			vocab_size=vocabularySize,  #sync with tokenizer vocab_size
			max_position_embeddings=(transformerMaxNumTokens+2),
			hidden_size=hiddenLayerSize,
			num_attention_heads=numberOfAttentionHeads,
			num_hidden_layers=numberOfHiddenLayers,
			intermediate_size=intermediateSize,
			type_vocab_size=1
		)
		model = RobertaForMaskedLM(config)

		#inputShape = (batchSize, transformerMaxNumTokens)
		#summary(model, inputShape)
		#print(sum(p.numel() for p in model.parameters()))
	#inputShape = (batchSize, transformerMaxNumTokens)
	#print("inputShape = ", inputShape)
	#summary(model, input_size=inputShape) 	#, dtypes=['torch.IntTensor']
	#print(model)
	
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	model.train()
	optim = AdamW(model.parameters(), lr=learningRate)
	
	numberOfDataFiles = len(paths)

	pathIndexMin = trainStartDataFile
	if(reserveValidationSet and trainNumberOfDataFiles==-1):
		pathIndexMax = int(numberOfDataFiles*trainSplitFraction)
	else:
		pathIndexMax = pathIndexMin+trainNumberOfDataFiles
	loader = createDataLoader(tokenizer, paths, pathIndexMin, pathIndexMax)
	
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		for batchIndex, batch in enumerate(loop):
			optim.zero_grad()
			
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
						
			accuracy = getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs)
			loss = outputs.loss
			
			loss.backward()
			optim.step()

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss.item(), accuracy=accuracy)
		
			if(batchIndex % modelSaveNumberOfBatches == 0):
				model.save_pretrained(modelFolderName)
		model.save_pretrained(modelFolderName)

def testDataset(tokenizer, paths):

	if(usePretainedModelDebug):
		model = RobertaForMaskedLM.from_pretrained("roberta-base")
	else:
		model = RobertaForMaskedLM.from_pretrained(modelFolderName, local_files_only=True)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	model.eval()
	
	numberOfDataFiles = len(paths)

	pathIndexMin = int(numberOfDataFiles*trainSplitFraction)
	pathIndexMax = pathIndexMin+testNumberOfDataFiles		
	loader = createDataLoader(tokenizer, paths, pathIndexMin, pathIndexMax)
		
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		
		averageAccuracy = 0.0
		averageLoss = 0.0
		batchCount = 0
		
		for batchIndex, batch in enumerate(loop):
			
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
						
			outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
			
			accuracy = getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs)
			loss = outputs.loss
			loss = loss.detach().cpu().numpy()
			
			if(not math.isnan(accuracy)):	#required for usePretainedModelDebug only
				averageAccuracy = averageAccuracy + accuracy
				averageLoss = averageLoss + loss
				batchCount = batchCount + 1

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		averageAccuracy = averageAccuracy/batchCount
		averageLoss = averageLoss/batchCount
		print("averageAccuracy = ", averageAccuracy)
		print("averageLoss = ", averageLoss)
		
def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py

def getAccuracy(tokenizer, input_ids, attention_mask, labels, outputs):
	tokenizerNumberTokens = getTokenizerLength(tokenizer)
	
	tokenLogits = (outputs.logits).detach()

	tokenLogitsTopIndex = torch.topk(tokenLogits, accuracyTopN).indices	#get highest n scored entries from dictionary	#tokenLogitsTopIndex.shape = batchSize, transformerMaxNumTokens, accuracyTopN
	
	maskTokenIndex = torch.where(input_ids==customMaskTokenID, 1.0, 0.0)	#maskTokenIndexFloat = maskTokenIndex.float()	

	if(accuracyTopN == 1):
		tokenLogitsTopIndex = torch.squeeze(tokenLogitsTopIndex)	#tokenLogitsTopIndex[:, :, 1] -> #tokenLogitsTopIndex[:, :] 	

		comparison = (tokenLogitsTopIndex == labels).float()
		comparisonMasked = torch.multiply(comparison, maskTokenIndex)
		accuracy = (torch.sum(comparisonMasked)/torch.sum(maskTokenIndex)).cpu().numpy() 
	else:
		labelsExpanded = torch.unsqueeze(labels, dim=2)
		labelsExpanded = labelsExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#labels broadcasted to [batchSize, transformerMaxNumTokens, accuracyTopN]
		comparison = (tokenLogitsTopIndex == labelsExpanded).float()
		maskTokenIndexExpanded = torch.unsqueeze(maskTokenIndex, dim=2)
		maskTokenIndexExpanded = maskTokenIndexExpanded.expand(-1, -1, tokenLogitsTopIndex.shape[2])	#maskTokenIndex broadcasted to [batchSize, transformerMaxNumTokens, accuracyTopN]
		comparisonMasked = torch.multiply(comparison, maskTokenIndexExpanded)	#maskTokenIndex broadcasted to [batchSize, transformerMaxNumTokens, accuracyTopN]
		accuracy = (torch.sum(comparisonMasked)/torch.sum(maskTokenIndex)).cpu().numpy() 	#or torch.sum(comparisonMasked)/(torch.sum(maskTokenIndexExpanded)/accuracyTopN)
	
	#accuracy2 = (torch.mean(comparisonMasked)).cpu().numpy()
	
	return accuracy

if(__name__ == '__main__'):
	if(statePreprocessDataset):
		dataset = downloadDataset()
		preprocessDataset(dataset)
	paths = [str(x) for x in Path(dataFolder).glob('**/*.txt')]
	if(usePretainedModelDebug):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		testDataset(tokenizer, paths)
	else:
		if(stateTrainTokenizer):
			trainTokenizer(paths)
		if(stateTrainDataset or stateTestDataset):
			tokenizer = loadTokenizer()
		if(stateTrainDataset):
			trainDataset(tokenizer, paths)
		if(stateTestDataset):
			testDataset(tokenizer, paths)


