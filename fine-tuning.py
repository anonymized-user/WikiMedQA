from mcqa_datasets import *
import pygrove
import logging

base_path=''

def preprocess_function(examples, max_length, random_slices=False, left_truncate=False,inject_knowledge=False,**kwargs):
    
    nb_endings=len([x for x in list(examples.keys()) if 'option_' in x])
    ending_names = [f"option_{i}" for i in range(nb_endings)]
    
    if inject_knowledge:
        first_sentences = [[examples[f'k_{j}'][i]+context for j in range(nb_endings)] for i,context in enumerate(examples["sent1"])]
    else:
        first_sentences = [[context] * nb_endings for i,context in enumerate(examples["sent1"])]
    second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(examples["sent1"])]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    if not left_truncate:
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation='only_first',padding=True,
                                  max_length=tokenizer.max_length)
    else:
        tokenized_examples = tokenizer(first_sentences, second_sentences)
        cut_pad_encodings_left(tokenized_examples, tokenizer, tokenizer.max_length)
        add_token_type(tokenized_examples, tokenizer.vocab['<k>'])
        peek=tokenized_examples

    # Un-flatten
    return {k: [v[i:i+nb_endings] for i in range(0, len(v), nb_endings)] for k, v in tokenized_examples.items()}


class base(xpflow.Xp):
    batch_size=16
    gradient_accumulation_steps=16
    eval_accumulation_steps=gradient_accumulation_steps//2
    max_samples=None
    learning_rate=2e-5
    num_train_epochs=10
    weight_decay=0.01
    adam_beta2=0.98
    adam_epsilon=1e-6
    max_length=512


    mlm_probability=0.15

class pretraining(base):
    inject_knowledge=False
    eval_accumulation_steps=gradient_accumulation_steps=8
    dataset_name=['medwiki+wikem+wikidoc']
    mlm_probability=0.15
    num_train_epochs=5
    push=False
    model_checkpoint=tokenizer_checkpoint=biolinkbert
    warmup_ratio=0.06
    warmup_steps=1000
    max_length=256
    nb_options=8
    random_slices=False
    do_eval=True,
    evaluation_strategy ='epoch'
    fixed_options=True
    left_truncate=True

class masking(pretraining):
    text_key=['text']
    dataset_name='medwiki'
    do_save=True
    
class medqa(pretraining):
    max_length=256
    dataset_name = ['medqa']
    backbone = biolinkbert
    model_checkpoint= f'{base_path}/{biolinkbert}/medwiki+wikem+wikidoc-5'
    num_train_epochs =5 
    tokenizer_checkpoint=backbone
    learning_rate=[3e-5]
    nb_options=4
    mlm_probability=[0.0]
    warmup_steps=500
    do_save=False
    evaluation_strategy = "epoch"
    
class knowledge(medmcqa):
    inject_knowledge=[True]
    left_truncate=True
    n_results=10
    learning_rate=[3e-5]
    n_highlights=[2]
    query_weight=[2]
    mlm_probability=0.0
    knowledge_drop=0.5
    dataset_name=['medmcqa']
    backbone=biolinkbert
          
def name(xp):
    return f'{base_path}/{xp.model_checkpoint}/{xp.dataset_name}-{xp.num_train_epochs}'+('-'+str(xp.chunk) if xp.chunk else '')

for xp in tqdm(knowledge()+medmcqa()):
    xp._hash=str(xp._hash)

    if xp.xp_name in {"pretraining",'masking','lpretraining'}:
        dataset = get_mc(xp)
    else:
        dataset = eval(f'get_{xp.dataset_name}')(xp)
    
    if xp.inject_knowledge:
        dataset=inject_knowledge(dataset, xp)
        
    tokenizer = AutoTokenizer.from_pretrained(xp.tokenizer_checkpoint, use_fast=True,max_length=xp.max_length)
    tokenizer.max_length=xp.max_length

    model = AutoModelForMultipleChoice.from_pretrained(xp.model_checkpoint)
    model.config.output_hidden_states=False
    tokenizer.add_special_tokens({'additional_special_tokens': [xp.k_token,'<answer>','<unk>']})
    new_token_type(model)
    model.resize_token_embeddings(len(tokenizer))

    lrs='{:.0e}'.format(xp.learning_rate)
    run_name =  f"""{xp.dataset_name}-{xp.max_samples}-{lrs}-{xp.model_checkpoint.split('/')[-1]}"""
    run=wandb.init(name=run_name)
    wandb.config.update(xp)
    
    encoded_datasets = dataset.map(preprocess_function, fn_kwargs=xp, batched=True)
    
    args = TrainingArguments(
        run_name,
        seed=int(time.time()),
        save_strategy="no",
        fp16_opt_level="O1",
        logging_dir=None,
        per_device_train_batch_size=xp.batch_size//xp.gradient_accumulation_steps,
        per_device_eval_batch_size=xp.batch_size//xp.gradient_accumulation_steps,
        **fc.project(dict(xp), dir(TrainingArguments))

    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(
            tokenizer, max_length=xp.max_length, mlm_probability=xp.mlm_probability,xp=xp),
        compute_metrics=compute_metrics,
    )
    res=trainer.train()
    trainer.data_collator.mlm_probability=0.0
    wandb.log(trainer.predict(encoded_datasets["test"]).metrics)
    additional_tests=[k for k in dataset.keys() if k not in {'train','test','validation'}]
    run.finish()
    for k in additional_tests:
        xp.dataset_name=k
        run=wandb.init(name=run_name+f'/{k}',config=xp)
        wandb.log(trainer.predict(encoded_datasets[k]).metrics)
        wandb.finish()