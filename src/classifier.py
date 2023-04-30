from typing import List
import torch
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModel, pipeline, Trainer, TrainingArguments, DataCollatorWithPadding, get_scheduler
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

class TransformerBinaryClassifier(torch.nn.Module):

    def __init__(self, plm_name: str):
        super(TransformerBinaryClassifier, self).__init__()
        self.lmconfig = AutoConfig.from_pretrained(plm_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(plm_name)
        self.lm = AutoModel.from_pretrained(plm_name, output_attentions=False)
        self.emb_dim = self.lmconfig.hidden_size
        self.num_labels = 3
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.emb_dim, self.num_labels),
            torch.nn.Softmax(dim=-1)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 3, 20], dtype=torch.float32)) # set class weights to balance loss function


    def forward(self, x):
        x : torch.Tensor = self.lm(x['input_ids'], x['attention_mask']).last_hidden_state
        global_vects = x.mean(dim=1)
        x = self.classifier(global_vects)
        return x.squeeze(-1)

    def compute_loss(self, predictions, target):
        return self.loss_fn(predictions, target)

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        #self.lmtokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.lm = AutoModel.from_pretrained(self.model_name, output_attentions=False)
        #self.lmconfig = AutoConfig.from_pretrained(self.model_name)
        self.model = TransformerBinaryClassifier(self.model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.num_epochs = 3



    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_data = self._load_data(train_filename)
        dev_data = self._load_data(dev_filename)

        num_training_steps = self.num_epochs * len(train_data)
        lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.model.to(device)
        progress_bar = tqdm(range(num_training_steps))
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in train_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = self.model(batch)
                #predicted_labels = torch.argmax(predictions, dim=-1).tolist()
                loss = self.model.compute_loss(predictions, batch['labels'])
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
        #trainer.save_model("model")


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        test_data = self._load_data(filename = data_filename, train = False)
        model = self.model
        model.to(device)
        model.eval()
        prediction = []
        with torch.no_grad():
            for batch in test_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = model(batch)
                predicted_labels = torch.argmax(predictions, dim=-1).tolist()
                prediction.append(predicted_labels)

        final_predictions = [item for sublist in prediction for item in sublist]
        return [self._get_polarity_label(x) for x in final_predictions]

    '''
    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained("model")
    '''
    # concatenate aspect_category
    def _tokenize_function(self, examples):
        text = examples["text"]
        aspect = examples["aspect_category"]
        concatenated = [a + ' [SEP] ' + t for a, t in zip(aspect,text)]
        return self.model.lmtokenizer(concatenated, truncation=True, padding=True)

    def _load_data(self, filename: str, train = True):
        df = pd.read_csv(filename, sep="\t", header=None, names=["labels", "aspect_category", "term", "offsets", "text"])
        # Remove the term and offsets column
        df = df.drop("term", axis=1)
        df = df.drop("offsets", axis=1)
        # Convert labels to integers
        label2id = {"positive": 0, "negative": 1, "neutral": 2}
        df["labels"] = df["labels"].apply(lambda x: label2id[x])
        ds = Dataset.from_pandas(df)
        tok_ds_train = ds.map(self._tokenize_function, batched=True)
        tok_ds_train = tok_ds_train.remove_columns(["text","aspect_category"])
        data_collator = DataCollatorWithPadding(tokenizer=self.model.lmtokenizer, padding=True, return_tensors='pt')
        dataloader = DataLoader(tok_ds_train, shuffle=train, batch_size=8, collate_fn=data_collator)
        return dataloader

    def _get_polarity_label(self, label: int):
        if label == 0:
            return "positive"
        elif label == 1:
            return "negative"
        elif label == 2:
            return "neutral"
