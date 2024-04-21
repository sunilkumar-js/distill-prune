import pandas as pd
import os
from typing import List, Dict, Union
import requests
import numpy as np
tgi_endpoint = 'xxxxxxx'
key = 'xxxxx'
import json 


def create_payload(
        prompt: str,
        max_new_tokens= 2048,
        do_sample= False,
        temperature= 0.01,
        best_of= 0,
    ) -> Dict:

    return {
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': max_new_tokens,
                    'do_sample': do_sample,
                    #'temperature' : temperature,
                    #'details': True,
                    #'best_of':best_of,
                    #'eos_token_id': 0
                }
            }

def model_prediction(inputs, endpoint=tgi_endpoint)-> str:
    headers = {
        "Authorization" : key,
    }

    response = requests.post(
        endpoint,
        headers=headers,
        data=json.dumps(inputs)
    )
    try:
        return response.json()
    except: 
        return response

def get_response_from_tgi(prompt, endpoint=tgi_endpoint):     
    payload = create_payload(prompt)
    response = model_prediction(payload, endpoint=endpoint)
    return response



def create_prompt(
        prompt_template : str,
        query : str,
        context: str,
    )-> str:
    prompt = prompt_template.format(query=query, context=context )
    return prompt

prompt_template= '''Answer the question based on the context provided 
---------------------------------
Question: {query}
Context: {context}
--------------------------------
your output msut be in below format: 
Rational: [Let's think step by step and explain how to arrive at the answer]
Answer: [Actual answer to the question]
'''

from datasets import load_dataset , Dataset, DatasetDict
squad = load_dataset('rajpurkar/squad')
train_len = int( 0.9*len(squad["train"]) )
train_data = squad["train"].shuffle(seed=123).select(idx for idx in range(10000))
shuffled_data =squad["validation"].shuffle(seed=123)
val_data = shuffled_data.select( [idx for idx in range(500)] )
test_data =shuffled_data.select( [idx for idx in range(500,1000)] )
squad = DatasetDict({"train":train_data,"validation":val_data,"test":test_data})

for split in squad:
    split_data =[]
    counter = 0
    for row in squad[split]:
        counter+=1
        prompt = create_prompt(prompt_template,row["question"],row["context"])
        data ={}
        data["id"] = row["id"]
        response = get_response_from_tgi(prompt)
        data["rational"] = response["generated_text"]
        print(f'''========={counter}======={row['question']}==========''')
        print(response["generated_text"])
        print("===============================================")
        split_data.append(data)
    with open(f"outputs/rational_data_{split}.json","w") as f:
        json.dump(split_data,f)

        
       
