https://github.com/ThinamXx/Transformers_NLP/blob/main/02.%20NLP%20with%20Transformers/05.%20Text%20Generation/Text%20Generation.ipynb

# https://www.pytorchlightning.ai/

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)

##################################################################################

import re
import os


def dict_merge_sum(d1, d2):
    """ Two dicionaries d1 and d2 with numerical values and
    possibly disjoint keys are merged and the values are added if
    the exist in both values, otherwise the missing value is taken to
    be 0"""

    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}


d1 = dict(a=4, b=5, d=8)
d2 = dict(a=1, d=10, e=9)

dict_merge_sum(d1, d2)

######################################

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load library
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
Create Text Data
# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
Create Bag Of Words
# Create the bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Show feature matrix
bag_of_words.toarray()
array([[0, 0, 0, 2, 0, 0, 1, 0],
       [0, 1, 0, 0, 0, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 0, 0]], dtype=int64)
View Bag Of Words Matrix Column Headers
# Get feature names
feature_names = count.get_feature_names()

# View feature names
feature_names
['beats', 'best', 'both', 'brazil', 'germany', 'is', 'love', 'sweden']
View As A Data Frame
# Create data frame
pd.DataFrame(bag_of_words.toarray(), columns=feature_names)
beats	best	both	brazil	germany	is	love	sweden
0	0	0	0	2	0	0	1	0
1	0	1	0	0	0	1	0	1
2	1	0	1	0	1	0	0	0

##########################################################

