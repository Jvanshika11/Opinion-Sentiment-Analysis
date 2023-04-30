# Opinion-Sentiment-Analysis

## Environment Setup:
1. Install python==3.9.0
2. Install the following libraries using pip

  a. pytorch = 1.13.1
  
  b. pytorch-lightning = 1.8.1
  
  c. transformers = 4.22.2
  
  d. datasets = 2.9.0 (just the library ‘datasets’, no labelled data)
  
  e. sentencepiece = 0.1.97
  
  f. scikit-learn = 1.2.0
  
  g. numpy = 1.23.5
  
  h. pandas = 1.5.3
  
  i. nltk = 3.8.1
  
  j. stanza = 1.4.2

## Execution:
You can run the code using this command:
python tester.py --n_runs 5 

### You can use gpu with specifying gpu-id using the --gpu flag. for example:
python tester.py --n_runs 5 --gpu 0

## Overview:
Using the aspect_category, aspect_term, and  sentence triplet, we aim with developing a classifier to predict the positive, negative, or neutral polarities of opinions in sentences. 
The input to the classifier consists of a sentence, an aspect term occurring in the sentence, and its aspect category. The dataset is in TSV format, with 1503 lines in the training set and 376 lines in the development set. The classifier should be trained only on the training set and evaluated on a test dataset that is not distributed.
We have three types of polarities and a class imbalance problem: only a small portion(only 4%) of the data have 'neutral' labels. 
Moreover, the dataset provided for testing the performance(devdata) is 70% positive labeled and hence biased and weak baseline.


## Implementing the Classifier: 
The type of classification model we used is a multi-class text classification mode. This classifier uses the AutoModel class from the transformers package to load a pre-trained BERT model and applies a sequence of fully connected layers to the output of the model to predict the sentiment of a given input text. 
We used the following steps to implement our classifier:

1. Firstly we used a pre-trained Transformer model ‘bert-base-uncased’. 
2. We did this by creating a Tokenizer and Model using the AutoTokenizer and AutoModel from transformers library with pre-trained weights of ‘bert-base-uncased’. 
3. Then, we created a sequential model with three layers (dropout, linear and followed by a softmax function) to classify the feature representation extracted from the bert-base-uncased model into one of the three labels.
4. Since, this is multi class classification problem, we use the softmax function to predict the labels using the probability distribution for each label and selecting the label with max probability for an instance.
5. Finally. we used the CrossEntropyLoss function to compute the loss. We also pass a tensor of weights with 1 for positive, 3 for negative and 20 for neutral to cater the class imbalance.


## Model Training:
The optimizer used for training the classifier is AdamW with a learning rate of 5e-5. The number of epochs is set to 3. 
We load the data using the _load_data function using the pandas library into a dataframe for initial manipulation. We drop the term and offset columns and map labels to integer values as follows:
"positive": 0, "negative": 1, "neutral": 2
Next, we create a dataset using the Dataset library from this dataframe and using the _tokenize function we concatenate each text with the aspect category with a separator token "[SEP]" between them, and generate the tokens using our pre-trained model tokenizer. 
The tokens are then encoded using the bert-base-uncased model to generate embeddings, which are representations of the input texts in a high-dimensional space that captures semantic information.
We have set the padding equal to true to ensure that all the inputs are of the same length. Finally, after this being passed to a DataLoader object for efficient batch processing during training or evaluation, they are fed to the classifier layer.


The Classifier class is used to train and make predictions using the SentimentClassifier model. We also define the train() and predict() methods that take in the file paths of the train and test data, respectively, and the device to run the training and inference on. The train() method loads the train and dev data with the _load_data method and trains the model on the train data for a fixed number of epochs (3) using the optimizer (AdamW) and the learning rate scheduler. We compute the loss and using loss.backward() function, optimise the weights of our network. 


## Prediction:
The predict() method loads the dev_data with the _load_data method, performing the same manipulations and makes predictions on the test data using the trained model.
We use the argmax function to select the label with maximum probability for an instance in a batch and finally create a list of all the predicted labels from the Tensors.
Lastly, we map each integer value back to ‘positive’, ‘negative’ or ‘neutral’ using the _get_polarity_label() function.


## Evaluation and Results:
Finally we use and run the tester.py with 5 runs and gpu device to generate results. The accuracy that we get on the dev dataset: 80.69%, and the execution time was found to be approximately 9 minutes.

