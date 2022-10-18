# Word Sense Disambiguation using PyTorch
Project for Natural Language Processing Course at Stony Brook University in March 2021 featuring two models, a word sense identifer and a language generator. Though both models learned on the same dataset, through changes in training parameters and architecture, these models are able to serve drastically different functions.

## Word Sense Identifier ##
Without addiional tweaking, the initial model parameters where able to achieve baseline 70% accuracy in indentifying the correct word sense after 500 epochs of logistic regression training.

## Language Generator ##
Final generated language provided interesting results, while most generated sentences make little sense as a whole, the grammar between each word does follow expectations. Results could be greatly improved by adding LSTMs or GRUs to the model architecture, so that context beyond immediately neighboring words can be considered in the lanaguage generation.
