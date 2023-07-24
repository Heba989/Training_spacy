from transformers import TFAutoModelForTokenClassification, TrainingArguments, Trainer, GPT2DoubleHeadsModel
from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback
from KayanresumeData import KayanResumeData as KRD
from transformers import TrainingArguments
from transformers import GPT2TokenizerFast, TFGPT2Tokenizer
import pandas as pd
import transformers
import numpy as np
import evaluate
import datasets
import torch
import os

# Referencing :: https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb
# https://huggingface.co/learn/nlp-course/chapter7/2
# https://colab.research.google.com/drive/1Vvju5kOyBsDr7RX_YAvp6ZsSOoSMjhKD?usp=sharing
# to handle the error :: For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def compute_metrics(p):

    seqeval = datasets.load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
def tokenize_and_align_labels(examples):
    if [] in examples['tokens']:
        examples["id"].pop(examples['tokens'].index([]))
        examples['ner_tags'].pop(examples['tokens'].index([]))
        examples['tokens'].pop(examples['tokens'].index([]))
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs

# Fetch data and convert it to ids using GPT2 tokenizer

data = datasets.load_dataset(path='KayanresumeData/KayanResumeData.py', name='KayanResumeData')

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
#tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-j-6B", add_prefix_space=True)

tokenizer.pad_token = tokenizer.eos_token
# data Preprocessing:
tokenized_data = data.map(tokenize_and_align_labels, batched=True)
print(len(tokenized_data['train']),len(tokenized_data['validation']))
id2label = {i: label for i, label in enumerate(data['train'].features['ner_tags'].feature.names)}
label2id = {v: k for k, v in id2label.items()}

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#
# token_data = tokenizer(
#     datasets_train['tokens'],
#     add_special_tokens=True,
#     return_attention_mask=True,
#     return_offsets_mapping=isinstance(tokenizer, PreTrainedTokenizerFast),
#     return_tensors="np",
#     return_token_type_ids=None,  # Sets to model default
#     padding='max_length',
# )
# token_data["input_texts"] = []
# for i in range(len(token_data["input_ids"])):
#     wp_texts = tokenizer.convert_ids_to_tokens(token_data["input_ids"][i])
#     token_data["input_texts"].append(wp_texts)
# token_data["pad_token"] = tokenizer.pad_token


## check the ids to tokens (inverse)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])


# create a batch of examples using DataCollatorWithPadding.
# It’s more efficient to dynamically pad the sentences to the longest length in a batch during collation,
# instead of padding the whole dataset to the maximum length
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Check GPU or CPU
#torch_device = torch.cuda.current_device()
model = GPT2DoubleHeadsModel.from_pretrained("gpt2", num_labels=41,
                                                        id2label=id2label,
                                                        label2id=label2id)
model.resize_token_embeddings(len(tokenizer))
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    load_best_model_at_end=True,
    save_strategy='epoch',       # The checkpoint save strategy to adopt during training.
    evaluation_strategy="epoch", #(:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"no"`):
                                     # The evaluation strategy to adopt during training
)


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_data["train"][:200],
    eval_dataset=tokenized_data["validation"][:100],
    data_collator=data_collator,         # training dataset
    compute_metrics=compute_metrics
    # evaluation dataset
)
# adding early stopper
# early_stop = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-3)

# trainer.add_callback(early_stop)

trainer.train()
trainer.evaluate()

predictions, labels, _ = trainer.predict(tokenized_data["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
metric = datasets.load_metric("seqeval")
results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)
import json
with open('meta_results.json', 'w') as fp:
    json.dump(results, fp)
