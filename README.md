# SigProp analysis

Signal propagation (SigProp) is a framework used to train a model in alternative to backpropagation. 
This is the code used to compare ConvSpikeNN results with the ones of original paper and differences in metrics between backpropagation and SigProp.

## Prerequisite
Install on local machine:
- pip
- python

To install the needed dependencies run the command:
```
pip install -r requirements.txt
```

## Run the analysis
First analysis uses the ConvSpikeNN and could be run launching the command:
```
python main.py SpikeNN
```
The second analysis is done using VGG model. To run it the command is:
```
python main.py VGG
```
Results of analysis are collected on json files inside the directory `results`.