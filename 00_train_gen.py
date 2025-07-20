import requests
import json
from datasets import Dataset
import datasets
import tqdm
import argparse
import os

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def create_few_shot_prompt_triviaqa(samples):
    prompt = 'This is a bot that correctly answers questions.\n'
    for sample in samples:
        prompt += f'\nQuestion: {sample["question"]} \nAnswer: {sample["answer"]["value"]} '
    return prompt

def triviaqa_process(save_path, prompt_sample_size=10, batch_size=4):
    save_path=save_path + '/triviaqa_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/train.json'

    train_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train")
    # Sample for few-shot prompt
    data_for_few_shot_prompt = train_data.select(range(prompt_sample_size))

    # Create the few-shot prompt
    few_shot_prompt = create_few_shot_prompt_triviaqa(data_for_few_shot_prompt)
    data_to_save = []
    for item in train_data:
    # Define the processing function
        answers = item['answer']
        question= item['question']
        text = few_shot_prompt + "\nQuestion: " + question + "\nAnswer:"
        data_to_save.append({'text': text, 'answer': answers})
    

    with open(save_path, 'w') as outfile:
        json.dump(data_to_save, outfile, indent=4)


def create_few_shot_prompt_sciqa(samples):
    prompt = 'This is a bot that correctly answers questions.\n'
    for sample in samples:
        prompt += f'\nQuestion: {sample["question"]} \nAnswer: {sample["correct_answer"]} '
    return prompt
def sciqa_process(save_path, prompt_sample_size=10):
    save_path=save_path + '/sciqa_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/train.json'

    train_data = datasets.load_dataset("sciq", split="train")
    data_for_few_shot_prompt = train_data.select(range(prompt_sample_size))
    few_shot_prompt = create_few_shot_prompt_sciqa(data_for_few_shot_prompt)
    
    data_to_save = []
    for item in train_data:
        question = item['question']
        answer = item['correct_answer']
        text = few_shot_prompt + "Question: " + question + " Answer:"
        data_to_save.append({'text': text, 'answer': answer})

    # Save data to file
    with open(save_path, 'w') as outfile:
        json.dump(data_to_save, outfile, indent=4)

def create_few_shot_prompt_medmcqa(samples):

    prompt = 'This is a bot that correctly answers questions.\n'
    for sample in samples:
        question = sample['question']
        if sample['cop'] == 1:
            answer=sample['opa']
        if sample['cop'] == 2:
            answer=sample['opb']
        if sample['cop'] == 3:
            answer=sample['opc']
        if sample['cop'] == 4:
            answer=sample['opd']
        prompt += f"\nQuestion:\n{question}\n\nAnswer: {answer} "
    return prompt

def medmcqa_process(save_path, prompt_sample_size=5):
    save_path=save_path + '/medmcqa_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/train.json'
    
    # Load datasets 
    train_data=datasets.load_dataset('dataset/medmcqa')
    train_data=train_data['train']
    train_data = [item for item in train_data if item['subject_name'] == 'Dental' and item['choice_type'] == 'single']
    data_to_save = []
    
    data_for_few_shot_prompt = train_data.select(range(prompt_sample_size))
    few_shot_prompt = create_few_shot_prompt_medmcqa(data_for_few_shot_prompt)
    for item in tqdm.tqdm(train_data):
        question = item['question']
        # options = f"A) {item['opa']}\nB) {item['opb']}\nC) {item['opc']}\nD) {item['opd']}"
        text = few_shot_prompt+f"\nQuestion:\n{question}\nAnswer:"
        if item['cop'] == 1:
            answer=item['opa']
        if item['cop'] == 2:
            answer=item['opb']
        if item['cop'] == 3:
            answer=item['opc']
        if item['cop'] == 4:
            answer=item['opd']
        data_to_save.append({'text': text, 'answer': answer})
    with open(save_path, 'w') as outfile:
        json.dump(data_to_save, outfile, indent=4)

    # Sample for few-shot prompt

def create_few_shot_prompt_nq(samples):
    ret = ''
    for item in samples:
        print(item['question'])
        question = item['question']
        answer = item['answer'][0]
        ret += f'\nQ: {question}\nA: {answer}'
    return ret

def nq_process(save_path, prompt_sample_size=5):
    save_path=save_path + '/nq_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/train.json'

    train_data = datasets.load_dataset("nq_open", split='train')
    data_for_few_shot_prompt = train_data.select(range(prompt_sample_size))
    few_shot_prompt = create_few_shot_prompt_nq(data_for_few_shot_prompt)
    data_to_save=[]
    for item in tqdm.tqdm(train_data):
        question = item['question']

        text = few_shot_prompt+f"\nQ:{question}\nA:"
        answer = item['answer'][0]
        data_to_save.append({'text': text, 'answer': answer})
    with open(save_path, 'w') as outfile:
        json.dump(data_to_save, outfile, indent=4)


def create_few_shot_prompt_gsm8k():
    ret=''
    question="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    answer="Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"
    ret += f'\nQuestion: {question}\nAnawer: {answer}'
    return ret


def gsm8k_process(save_path, prompt_sample_size=5):
    save_path=save_path + '/gsm8k_dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + '/train.json'

    train_data = datasets.load_dataset("gsm8k", split='train')
    data_for_few_shot_prompt = train_data.select(range(prompt_sample_size))
    few_shot_prompt = create_few_shot_prompt_nq(data_for_few_shot_prompt)
    data_to_save=[]
    for item in tqdm.tqdm(train_data):
        question = item['question']
        text = few_shot_prompt+f"\nQ:{question}\nA:"
        answer = item['answer']
        data_to_save.append({'text': text, 'answer': answer})
    with open(save_path, 'w') as outfile:
        json.dump(data_to_save, outfile, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Process datasets.")
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset to process')
    parser.add_argument('--save_dir', type=str, default='dataset_process', help='Directory to save the processed data')

    args = parser.parse_args()

    if args.dataset_name == 'trivia_qa':
        save_path=args.save_dir
        triviaqa_process(save_path)
        print("hello")
    elif args.dataset_name == 'sciqa':
        save_path=args.save_dir
        sciqa_process(save_path)
    elif args.dataset_name == 'medmcqa':
        save_path=args.save_dir
        medmcqa_process(save_path)
    elif args.dataset_name == 'nq':
        save_path=args.save_dir
        nq_process(save_path)    
if __name__ == '__main__':
    main()

# python /mnt/data2/lirui/hello/uncertainty/my_uncertainty/pipeline/000_train_gen.py --dataset_name trivia_qa --save_dir dataset_process
