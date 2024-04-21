from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers import pipeline
from datasets import load_dataset,Dataset,DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import copy 
from transformers import TrainingArguments, Trainer,DataCollatorWithPadding
from transformers import GPT2Tokenizer, GPT2Model

squad = load_dataset('rajpurkar/squad')
train_len = int( 0.9*len(squad["train"]) )
train_data = squad["train"].shuffle(seed=123).select(idx for idx in range(10000))
shuffled_data =squad["validation"].shuffle(seed=123)
val_data = shuffled_data.select( [idx for idx in range(500)] )
test_data =shuffled_data.select( [idx for idx in range(500,1000)] )
squad = DatasetDict({"train":train_data,"validation":val_data,"test":test_data})

import copy
def load_squad(squad,
    split,
    tokenizer_name,
    model_name,
    max_input_length,
    batch_size,
    shuffle=True,
    keep_in_memory=False,
    print_info=False,
):
    """load webquestions dataset
    train: 3.78k, test: 2.03k
    """
    
    dataset = squad[split]
    dataset.cleanup_cache_files()
    if 'opt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    def _tokenize_fn(strings):
        """Tokenize a list of strings, memorize source length"""
        tokenizer.padding_side = "right" 
        tokenized_list = [
            tokenizer(
                text,
                max_length=max_input_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            mask.sum().item() for mask in attention_mask
        ]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )   
    
    def _truncate_context(context, max_length=10000):
        context_tokens = tokenizer(context)
        truncate_ratio = max_length / (len(context_tokens["input_ids"]) + 1e-5)
        if truncate_ratio < 1:
            return context[:int(len(context) * truncate_ratio)]
        else:
            return context 
    
    def preprocess_fn(raw_examples):
        """preprocess example strings, mask source part in labels"""
        prompt_template= '''Answer the question based on the context provided
        --------------------------------
        Question: {question}
        Context: {context}
        Answer: '''
        sources = [
            prompt_template.format(question=question,context= _truncate_context(context)) for question,context in zip(raw_examples["question"],raw_examples["context"])
        ]
        
        examples = [
            source_text + '\n'.join(answer) for source_text,answer in zip(sources,raw_examples["answers"])
        ]
        
        # left-padded source for validation & testing
        tokenizer.padding_side = "left"
        
            
        lp_sources = tokenizer(
            sources,
            max_length=max_input_length,
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        )
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100 # mask source
            label[label == tokenizer.pad_token_id] = -100 # mask paddings
        return dict(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            input_ids_lens=sources_tokenized["input_ids_lens"],
            lp_sources=lp_sources["input_ids"])
    
    processed_dataset = dataset.map(preprocess_fn, batched=True)
    processed_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "input_ids_lens", "lp_sources"]
    )
    return processed_dataset,tokenizer
    
train_ds, tokenizer = load_squad(squad, split="train",tokenizer_name="gpt2",model_name="gpt2",max_input_length=2048,batch_size=2)
valid_ds,_ = load_squad(squad, split="validation",tokenizer_name="gpt2",model_name="gpt2",max_input_length=2048,batch_size=2)
test_ds,_ = load_squad(squad, split="test",tokenizer_name="gpt2",model_name="gpt2",max_input_length=2048,batch_size=2)

model = GPT2Model.from_pretrained('gpt2')
device = torch.device("cuda")
model = model.to(device)
training_args = TrainingArguments(
    output_dir = 'temp', # rename to what you want it to be called
    num_train_epochs=1, # your choice
    warmup_steps = 500,
    per_device_train_batch_size=1, # keep a small batch size when working with a small GPU
    per_device_eval_batch_size=1,
    weight_decay = 0.01, # helps prevent overfitting
    logging_steps = 10,
    evaluation_strategy = 'steps',
    eval_steps=30000, # base this on the size of your dataset and number of training epochs
    save_steps=1e6,
    fp16=True
)
trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, 
                  train_dataset = train_ds, eval_dataset = valid_ds )
trainer.train()
trainer.save_model("model_dir")