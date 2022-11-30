"""TSBpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n transformersenv
source activate transformersenv
conda install python=3.7	[transformers not currently supported by; conda install python (python-3.10.6)]
pip install datasets
pip install transfomers==4.23.1
pip install torch
pip install lovely-tensors

# Usage:
source activate transformersenv
python TSBpt_main.py

# Description:
TSBpt main - Transformer Syntactic Bias (TSB): various neural architectures with syntactic inductive biases (recursiveLayers)

"""


import torch
from tqdm.auto import tqdm
from pathlib import Path

from transformers import AdamW
import math 

from TSBpt_globalDefs import *
import TSBpt_data
if(useAlgorithmTransformer):
	from TSBpt_transformer import createModel, loadModel, saveModel, propagate
elif(useAlgorithmRNN):
	from TSBpt_RNN import createModel, loadModel, saveModel, propagate
elif(useAlgorithmSANI):
	from TSBpt_SANI import createModel, loadModel, saveModel, propagate

def main():
	if(statePreprocessDataset):
		dataset = TSBpt_data.downloadDataset()
		TSBpt_data.preprocessDataset(dataset)
	paths = [str(x) for x in Path(dataFolder).glob('**/*.txt')]
	
	if(usePretainedModelDebug):
		tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		testDataset(tokenizer, paths)
	else:
		if(stateTrainTokenizer):
			TSBpt_data.trainTokenizer(paths)
		if(stateTrainDataset or stateTestDataset):
			tokenizer = TSBpt_data.loadTokenizer()
		if(stateTrainDataset):
			trainDataset(tokenizer, paths)
		if(stateTestDataset):
			testDataset(tokenizer, paths)
			
def continueTrainingModel():
	continueTrain = False
	if((trainStartEpoch > 0) or (trainStartDataFile > 0)):
		continueTrain = True	#if trainStartEpoch=0 and trainStartDataFile=0 will recreate model, if trainStartEpoch>0 or trainStartDataFile>0 will load existing model
	return continueTrain	
	
def trainDataset(tokenizer, paths):

	if(continueTrainingModel()):
		model = loadModel()
	else:
		model = createModel()
	
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
	if(useAlgorithmTransformer):
		useMLM = True
	else:
		useMLM = False
	loader = TSBpt_data.createDataLoader(useMLM, tokenizer, paths, pathIndexMin, pathIndexMax)
	
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		for batchIndex, batch in enumerate(loop):
			optim.zero_grad()
			
			loss, accuracy = propagate(device, model, tokenizer, batch)

			loss.backward()
			optim.step()

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss.item(), accuracy=accuracy)
		
			if(batchIndex % modelSaveNumberOfBatches == 0):
				saveModel(model)
		saveModel(model)

def testDataset(tokenizer, paths):

	if(usePretainedModelDebug):
		if(useAlgorithmTransformer):
			model = RobertaForMaskedLM.from_pretrained("roberta-base")
		else:
			print("testDataset error: usePretainedModelDebug requires useAlgorithmTransformer")
			exit()
	else:
		model = loadModel()

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)

	model.eval()
	
	numberOfDataFiles = len(paths)

	pathIndexMin = int(numberOfDataFiles*trainSplitFraction)
	pathIndexMax = pathIndexMin+testNumberOfDataFiles
	loader = TSBpt_data.createDataLoader(tokenizer, paths, pathIndexMin, pathIndexMax)
		
	for epoch in range(trainStartEpoch, trainStartEpoch+trainNumberOfEpochs):
		loop = tqdm(loader, leave=True)
		
		averageAccuracy = 0.0
		averageLoss = 0.0
		batchCount = 0
		
		for batchIndex, batch in enumerate(loop):
			
			loss, accuracy = propagate(device, model, tokenizer, batch)

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

if(__name__ == '__main__'):
	main()

