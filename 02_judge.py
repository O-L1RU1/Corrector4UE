import json
import tqdm
import argparse
import os
import re
import evaluate

from openai import OpenAI

from sentence_transformers import SentenceTransformer

rouge = evaluate.load('evaluate-main/metrics/rouge')


client = OpenAI(api_key="sk-xxxxxxxxxxxxxx", base_url="https://api.deepseek.com")
def chat(prompt: str):

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=False
)   
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def method_rouge(response,answer):
    results = rouge.compute(predictions=[response], references=[answer])
    return results['rougeL']

def method_similarity(response,answer):
    # Load the pre-trained model
    model= SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode(response, convert_to_tensor=True)
    embedding2 = model.encode(answer, convert_to_tensor=True)
    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()


def extract_numbers(text):
    pattern = r'\d+\.\d+|\d+'
    matches = re.findall(pattern, text)
    numbers = [float(match) for match in matches]
    return numbers[0]


def method_model(response,answer):
    prompt=f"""
Please determine whether the response is correct based on the reference answer. If it is correct, return 1; if not, return 0. Do not return any additional information:
Reference answer: {answer}
Response: {response}
    """
    score=chat(prompt)
    print(score)
    score=extract_numbers(score)
    return score
def method_match(response,answer):
    if response.lower() in answer.lower():
        return 1
    else:
        return 0


strings_to_filter_on = [
        '.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:',
        'ANSWER:'
    ]
def extract_before_question(text):
    pattern = '|'.join(re.escape(s) for s in strings_to_filter_on)
    regex = rf'(.*?)(?={pattern})'
    match = re.search(regex, text, re.DOTALL)    
    if match:
        return match.group(1).strip()  
    else:
        return None  

def ensure_file_exists(file_path, source_file_path):
    if not os.path.exists(file_path):
        with open(source_file_path, 'r') as f:
            data = json.load(f)
        with open(file_path,'w') as f:
            json.dump(data,f,indent=4)

def judge_coqa(dataset,model):
    path='dataset_process/coqa_dataset/train_gen.json'
    path_score='dataset_process/coqa_dataset/train_gen_score.json'
    ensure_file_exists(path_score,path)
    with open(path_score, 'r') as f:
        data = json.load(f)
    if 'clean_response' not in data[0]:
        for item in tqdm.tqdm(data):
            key=f"{model}_response"
            item['clean_response']=extract_before_question(item[key])
            if item['clean_response'] is None:
                item['clean_response']=item[key]
            print(item['clean_response'])

    for item in tqdm.tqdm(data):
        item['rougel']=method_rouge(item['clean_response'],item['answer']['normalized_value'])
        item['llama_score']=method_model(item['clean_response'],item['answer']['normalized_value'])
    with open(path_score, 'w') as f:
        json.dump(data, f, indent=4)
        

def judge_sciqa(dataset,model):
    
    path=f'dataset_process/{dataset}_dataset/{model}_train_gen.json'
    path_score=f'dataset_process/{dataset}_dataset/{model}_train_gen_score.json'
    ensure_file_exists(path_score,path)

    with open(path_score, 'r') as f:
        data = json.load(f)
    if 'clean_response' not in data[0]:
        key=f"{model}_response"
        for item in tqdm.tqdm(data):
            item['clean_response']=extract_before_question(item[key])
            if item['clean_response'] is None:
                item['clean_response']=item[key]
            print(item['clean_response'])
    for item in tqdm.tqdm(data):
        item['rougel']=method_rouge(item['clean_response'],item['answer'])
        item['llama_score']=method_model(item['clean_response'],item['answer'])
    with open(path_score, 'w') as f:
        json.dump(data, f, indent=4)


def judge_triviaqa(dataset,model):
    path=f'dataset_process/{dataset}_dataset/{model}_train_gen.json'
    path_score=f'dataset_process/{dataset}_dataset/{model}_train_gen_score.json'
    ensure_file_exists(path_score,path)

    with open(path_score, 'r') as f:
        data = json.load(f)
    if 'clean_response' not in data[0]:
        key=f"{model}_response"
        for item in tqdm.tqdm(data):
            item['clean_response']=extract_before_question(item[key])
            if item['clean_response'] is None:
                item['clean_response']=item[key]
            print(item['clean_response'])
    

    for item in tqdm.tqdm(data):
        item['rougel']=method_rouge(item['clean_response'],item['answer']['normalized_value'])
        item['llama_score']=method_model(item['clean_response'],item['answer']['normalized_value'])

    with open(path_score, 'w') as f:
        json.dump(data, f, indent=4)

def judge_medmcqa(dataset,model):
    path=f'dataset_process/{dataset}_dataset/{model}_train_gen.json'
    path_score=f'ataset_process/{dataset}_dataset/{model}_train_gen_score.json'
    ensure_file_exists(path_score,path)
    with open(path_score, 'r') as f:
        data = json.load(f)
    if 'clean_response' not in data[0]:
        key=f"{model}_response"
        for item in tqdm.tqdm(data):
            item['clean_response']=extract_before_question(item[key])
            if item['clean_response'] is None:
                item['clean_response']=item[key]
            print(item['clean_response'])
    for item in tqdm.tqdm(data):
        item['rougel']=method_rouge(item['clean_response'],item['answer'])
        item['llama_score']=method_model(item['clean_response'],item['answer'])
    with open(path_score, 'w') as f:
        json.dump(data, f, indent=4)


def judge_nq(dataset,model):
    path=f'dataset_process/{dataset}_dataset/{model}_train_gen.json'
    path_score=f'dataset_process/{dataset}_dataset/{model}_train_gen_score.json'
    ensure_file_exists(path_score,path)
    with open(path_score, 'r') as f:
        data = json.load(f)
    if 'clean_response' not in data[0]:
        key=f"{model}_response"
        for item in tqdm.tqdm(data):
            item['clean_response']=extract_before_question(item[key])
            if item['clean_response'] is None:
                item['clean_response']=item[key]
            print(item['clean_response'])

    for item in tqdm.tqdm(data):
        item['rougel']=method_rouge(item['clean_response'],item['answer'])
        item['llama_score']=method_model(item['clean_response'],item['answer'])


    with open(path_score, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a text-generation model.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset')
    # parser.add_argument('--method', type=str, help='')
    parser.add_argument('--model', type=str, help='')
    args = parser.parse_args()

    if args.dataset == 'sciqa':
        judge_sciqa(args.dataset, args.model)
    elif args.dataset == 'triviaqa':
        judge_triviaqa(args.dataset, args.model)
    elif args.dataset == 'medmcqa':
        judge_medmcqa(args.dataset, args.model)
    elif args.dataset == 'nq':
        judge_nq(args.dataset, args.model)


#CUDA_ADDRESS=0 python 02_judge.py --dataset triviaqa --model opt_6.7b
