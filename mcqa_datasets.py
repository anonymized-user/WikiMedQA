import pandas as pd
import random
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import string
from dataclasses import dataclass, field
import re, copy
import exrex
import joblib
from appdirs import user_cache_dir
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from transformers import DataCollatorForLanguageModeling ,DataCollatorForWholeWordMask
from typing import Optional
import torch
import pygrove
import funcy as fc
from left_truncation import cut_pad_encodings_left

def compute_metrics(eval_predictions):
    peek = eval_predictions
    predictions, label_ids = eval_predictions
    if type(predictions)==tuple:
        predictions=predictions[0]
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).mean().item(),'size':len(preds)}

def safe_search(x, args, silent=False):
    from med_ir_utils import search
    return search(x,size=args.n_results//args.n_highlights,index_name=args.get('index_name',None),
                  n_highlights=args.n_highlights)
    try:
        return search(x,size=args.n_results//args.n_highlights,
                      n_highlights=args.n_highlights)
    except:
        if not silent:
            print('!',x)
        return []
    
def inject_knowledge(dataset,args,prefix='option_'):
    splits=list(dataset.keys())
    for k in splits:
        df=pd.DataFrame(dataset[k])
        for i,c in enumerate(df.filter(regex='option\_\d')):
            knowledge=(df.sent1+((df[c]+ ' ')*args.query_weight)).parallel_map(lambda x:safe_search(x,args))\
                .map(lambda l: [x['text'] for x in l])\
                .map(lambda x: args.k_token+args.k_token.join(x[::-1]))
            df[f'k_{i}']=knowledge+f' {args.k_token}'
            
        dataset[k]=Dataset.from_pandas(df,preserve_index=False)         
    return dataset


def get_mc(args):
    
    text_key="text"
    if 'text_key' in args:
        print(args.text_key)
        text_key=args.text_key
    
    l=[]
    for x in args.dataset_name.split('+'):
        l+=[cap_options(pd.read_csv(f's3://{author}/mc_{x}.csv'),n_max=args.nb_options)]

    df=pd.concat(l).sample(frac=1.0)
    df['sent1']=df[text_key].map(str)
    df['sent2']=''  

            
    if args.get('no_diff',None):
        print('no_diff')
        df['options']=df['options_no_diffdiag']               
          
    for i in range(args.nb_options):
        df[f'option_{i}']=df.options.map(lambda x:x[i])
    
    dataset = Dataset.from_pandas(df).train_test_split(test_size=2500)

    to_drop = [k for k in dataset['test'].features.keys() if re.match('.*([5-9]|\d{2,}).*',k)]
    dataset['test']=dataset['test'].remove_columns(to_drop)
    return dataset

def get_medmcqa(args=None,export=False):
    dataset = load_dataset("medmcqa")
    medmcqa_mapping = {**fc.walk_values((lambda x: f'option_{x}'),
        fc.flip(dict(enumerate(['opa', 'opb', 'opc', 'opd', 'cop'])))
    ), 'question':'sent1','cop':'label'}
    dataset=dataset.rename_columns(medmcqa_mapping)
    if not export:
        dataset['test']=dataset['validation']
    if export:
        del dataset['train']

    dataset['mmlu']=get_mmlu()
    dataset['headqa']=get_headqa(args)['test']
    return dataset

def get_usmle(args,explanation=False):
    df = cap_options(pd.read_csv('https://www.dropbox.com/s/e8o9hadowrj3nbh/usmle.csv?dl=1'), args.nb_options)
    df['options']=df.options.map(lambda x: dict(zip(string.ascii_uppercase,eval(str(x)))))
    df['answer_idx']= df.label.map(lambda x: string.ascii_uppercase[x])
    df['answer'] = df.answer_text
    df['label']=df.answer_idx.rank(method='dense').map(int)-1
    df['sent1']=df.question
    df['sent2']=''
    
    df=df[['sent1','sent2','label','options'] +['explanation' for _ in [0] if explanation]]
    for i,k in enumerate('ABCDE'):
        df[f'option_{i}']=df.options.map(lambda x:x.get(k,''))
    
    #if args.inject_knowledge:
    #    df = inject_knowledge(df,args)
        
    dataset = Dataset.from_pandas(df).train_test_split()
    dataset['validation']=dataset['test']

    
    return dataset
medqa_path='/cw/working-arwen/damien/datasets/medqa/'


def get_headqa(args):
    dataset = load_dataset("head_qa", 'en')
    for split in ['train','test','validation']:
        df=pd.DataFrame(dataset[split])
        df=df[df.category=='medicine']  
        df['options']=df.answers.map(lambda a:[x['atext'] for x in a])
        df['label']=df['ra']-1
        df['sent1']=df.qtext
        df['sent2']=''
        df=df[['sent1','sent2','label','options']]
        n_options=df.options.map(len).max()
        for i in range(n_options):
            df[f'option_{i}'] = df.options.map(lambda x:x[i])
        df=df.drop('options',axis=1)

        dataset[split]=Dataset.from_pandas(df)
    return dataset

def get_mmlu():
    df=pd.read_csv('s3://{author}/mctest/professional_medicine_test.csv',
    names=['sent1',*exrex.generate('option_[0-3]'),'label'])
    df.label=df.label.rank(method='dense').map(int)-1
    return Dataset.from_pandas(df)

def get_medqa(args,explanation=False):

    dataset = DatasetDict()
    for split in ['test','dev','train']:
        df=pd.read_json(f's3://{author}/medqa/us4/{split}.jsonl',lines=True)
        l=[df]
        if split=='train':
                l+=[dft]
            #dfu = cap_options(pd.read_csv('https://www.dropbox.com/s/e8o9hadowrj3nbh/usmle.csv?dl=1'))
            #dfu['options']=dfu.options.map(lambda x: dict(zip(string.ascii_uppercase,eval(str(x)))))
            #dfu['answer_idx']= dfu.label.map(lambda x: string.ascii_uppercase[x])
            #dfu['answer'] = dfu.answer_text
            
            df=pd.concat(l).sample(frac=1.0)
        df['label']=df.answer_idx.rank(method='dense').map(int)-1
        df['sent1']=df.question
        df['sent2']=''
        df=df[['sent1','sent2','label','options']+['explanation' for _ in [0] if explanation]]
        for i,k in enumerate('ABCDE'):
            df[f'option_{i}']=df.options.map(lambda x:x.get(k,''))
        #if args.inject_knowledge:
        #    df = inject_knowledge(df,args)
        dataset[split.replace("dev","validation")]=Dataset.from_pandas(df)
        dataset['validation']=dataset['test']
    dataset['mmlu']=get_mmlu()
    dataset['headqa']=get_headqa(args)['test']
    return dataset



@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: Optional[float]=0.0
    max_length: Optional[int] = None
    xp: Optional = field(default_factory=dict)


    def __call__(self, features):
        labels = [feature.pop('label') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        def mask(batch):
            dclm = DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=self.mlm_probability)
            masked, _ = dclm.torch_mask_tokens(batch["input_ids"].cpu())
            batch["input_ids"]=masked
            return batch
        if self.mlm_probability:
            batch=mask(batch)

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        if random.random()<self.xp.get('knowledge_drop',0):
            batch['input_ids'][batch['token_type_ids']==2]=self.tokenizer.mask_token_id
        return batch

def new_token_type(model):
    m=getattr(model, model.config.model_type)
    e=m.embeddings.token_type_embeddings
    m.embeddings.token_type_embeddings=torch.nn.Embedding.from_pretrained(torch.vstack((e.weight,e.weight[[0]])),freeze=False)
    
def add_token_type(encoded, k_token=None):
    starts = list(pd.DataFrame((torch.tensor(encoded.input_ids)==k_token).nonzero().numpy()).groupby(0).agg(max)[1])
    for s,t in zip(starts, encoded.token_type_ids):
        t[1:s]=[2 for _ in t[1:s]]
    
def preprocess_function(examples, tokenizer, random_slices=False, **kwargs):
    nb_endings=len([x for x in list(examples.keys()) if 'option_' in x])
    ending_names = [f"option_{i}" for i in range(nb_endings)]
    
    f=fc.identity
    first_sentences = [[f(context)] * nb_endings for context in examples["sent1"]]
    second_sentences = [[f"{f(examples[end][i])}" for end in ending_names] for i, header in enumerate(examples["sent1"])]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation='only_first',padding=True,
                                  max_length=tokenizer.max_length)
    # Un-flatten
    return {k: [v[i:i+nb_endings] for i in range(0, len(v), nb_endings)] for k, v in tokenized_examples.items()}
