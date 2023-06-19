import spacy
from datetime import date
from spacy.cli.train import train
from spacy import displacy
import spacy_transformers


# # define train data path
configpath = 'config_beart.cfg'
print('GCS')
day = date.today()
# define model path
model = f'Models/newmodelbeart{day}'
# # define train data path
# configpath = './h1config.cfg'
# train_path = f'./spacy_data/train{day}.spacy'
train_path = 'data/train2023-06-14.spacy'
# valid_path = f"./spacy_data/dev{day}.spacy"
valid_path = 'data/dev2023-06-14.spacy'
print('train must be starting now')
train(config_path=configpath, output_path=model,overrides={"paths.train": train_path, "paths.dev": valid_path}, use_gpu=-1)
