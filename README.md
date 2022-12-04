# Word Sense Disambiguation using PyTorch
Project for Natural Language Processing (CSE 354) at Stony Brook University in March 2021 consisting of two models, a word sense identifer and a language generator. Both models were trained on the same dataset, a subset of OneSeC Small (Scarlini et al., 2019), a dataset consisting of automatically-generated corpora in multiple languages with sense annotations for nouns using WordNet for English and BabelNet for all other languages as inventories of senses. The subset used in this project was narrowed down to only senses of the words *Process*, *Machine*, and *Language*.

The dataset can be found here: 
[OneSeC Small](https://metatext.io/datasets/onesec-small)

## Word Sense Identifier ##
Without addiional tweaking, the initial model parameters were able to achieve baseline 70% accuracy in indentifying the correct word sense after 500 epochs of logistic regression training.

## Language Generator ##
Final generated language provided interesting results, while most generated sentences make little sense as a whole, the grammar of each sentence does meet expectations. Results could be greatly improved by adding LSTMs or GRUs to the model architecture, so that context beyond immediately neighboring words can be considered in the lanaguage generation.
