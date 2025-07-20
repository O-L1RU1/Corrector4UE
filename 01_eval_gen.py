import argparse
import json
import re
import tqdm
from transformers import pipeline
import requests
import requests

def chat_llama3(prompt: str,max_tokens=1024):
    url = 'http://localhost:8000/v1/chat/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    # , api_key: str
    data = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": prompt}],  
        "max_tokens": max_tokens,
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=data)
    res=response.json()
    # print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']

    

def extract_floats_from_string(s):
    floats = re.findall(r'\d*\.\d+|\d+', s)
    return [float(num) for num in floats]

def chat(prompt, generator):
    response = generator(prompt, temperature=0)
    print(response)
    res = response[0]['generated_text'][len(prompt) + 1:]
    print(res)
    return res




def eval_opt(dataset, max_length,num_to_use,model):
    if model=="opt_2.7b":
        opt_path="facebook/opt-2.7b"
    if model=="opt_6.7b":
        opt_path="facebook/opt-6.7b"
    generator = pipeline('text-generation', model=opt_path, max_length=max_length, device=0)
    
    with open(f'dataset_process/{dataset}_dataset/train.json', 'r') as f:
        data = json.load(f)
    
    new_data = []
    for i in tqdm.tqdm(range(len(data[:num_to_use]))):
        prompt = data[i]['text']
        answer = data[i]['answer']
        # print(prompt)
        response = chat(prompt, generator)
        # print(response)
        new_data.append({'text': prompt, f'{model}_response': response, 'answer': answer})

    with open(f'dataset_process/{dataset}_dataset/{model}_train_gen.json', 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def extract_last_qa(text):
    pattern = r'Q: (.*?)(?=A:|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[-1].strip()
    else:
        return None

def extract_last_question_answer(text):
    pattern = r'Question: (.*?)(?=Answer:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None

def eval_llama3(dataset, max_length,num_to_use,model):    

    with open(f'dataset_process/{dataset}_dataset/train.json', 'r') as f:
        data = json.load(f)
    new_data = []
    for i in tqdm.tqdm(range(len(data[:num_to_use]))):
        prompt = data[i]['text']
        if 'Question:' in prompt:
            prompt = extract_last_question_answer(prompt)
        else:
            prompt = extract_last_qa(prompt)
        prompt+="\nGive me the briefest possible answer."
        print(prompt)
        answer = data[i]['answer']
        response = chat_llama3(prompt,max_length)
        print(response)
        new_data.append({'text': prompt, f'{model}_response': response, 'answer': answer})
    with open(f'dataset_process/{dataset}_dataset/{model}_train_gen.json', 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a target model.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset')
    parser.add_argument('--max_length', type=int, help='The maximum length of generated text')
    parser.add_argument('--num_to_use', type=int, help='The sample number to use')
    parser.add_argument('--model', type=str, help='The name of the model')
    
    args = parser.parse_args()
    if 'llama3' in args.model:
        eval_llama3(args.dataset, args.max_length, args.num_to_use,args.model)
    if 'opt' in args.model:
        eval_opt(args.dataset, args.max_length, args.num_to_use,args.model)


#CUDA_VISIBLE_DEVICE=0 python 01_eval_gen.py --dataset triviaqa --max_length 250 --num_to_use 5000 --model llama3
