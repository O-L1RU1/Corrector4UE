import json
import tqdm 
import re
import tqdm
import argparse
import os
import re

def extract_last_qa(text):
    pattern = r'Q:(.*?)(?=A:|$)'
    matches = re.findall(pattern, text, re.DOTALL)    
    if matches:
        return matches[-1].strip()
    else:
        return text.replace('Give me the briefest possible answer.','').strip()
def extract_last_question_answer(text):
    pattern = r'Question: (.*?)(?=Answer:|$)'    
    matches = re.findall(pattern, text, re.DOTALL)    
    if matches:
        return matches[-1].strip()
    else:
        return text.replace('Give me the briefest possible answer.','').strip()


def determine_correct(rougel,llama_score):
    rougel_threshold = 0.5  
    llama_score_threshold = 0.5 
    if rougel >= rougel_threshold or llama_score >= llama_score_threshold:
        return 1  
    else:
        return 0  
        

def process(dataset,model):
    path_score=f'dataset_process/{dataset}_dataset/{model}_train_gen_score.json'
    path_save=f'dataset_train_ready/{dataset}_{model}'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_save=f'dataset_train_ready/{dataset}_{model}/train.json'
    new_data=[]
    with open(path_score, 'r') as f:
        data = json.load(f)
    for item in tqdm.tqdm(data):
        text=extract_last_qa(item['text'])
        print(text)
        rougel=item['rougel']
        llama_score=item['llama_score']
        correct=determine_correct(rougel,llama_score)
        new_data.append({"text":text,"correct":correct})
    with open(path_save,'w') as f:
        json.dump(new_data,f,indent=4)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a text-generation model.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset')
    parser.add_argument('--model', type=str, help='The name of the model')
    args = parser.parse_args()
    dataset=args.dataset
    model=args.model
    process(dataset,model)

#python pipeline/03_correct.py --dataset triviaqa --model opt_6.7b
