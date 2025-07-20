import argparse
import json
import numpy as np
import torch
import tqdm
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
import numpy as np

import json
import os

from sklearn.model_selection import train_test_split

def get_best_f1(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    f1_scores = [f1_score(true_labels, (scores >= t).astype(int)) for t in thresholds]
    best_threshold_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_threshold_index]
    return best_f1

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    total_samples = len(y_true)
    bin_proportions = bin_counts / total_samples
    abs_errors = np.abs(prob_true - prob_pred)
    abs_errors_padded = np.pad(abs_errors, (0, len(bin_proportions) - len(abs_errors)), mode='constant', constant_values=0)
    ece = np.sum(abs_errors_padded * bin_proportions)
    return ece

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model performance.")
    parser.add_argument("--data_read_path", type=str, required=True, help="Path to the test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the corrector")
    parser.add_argument("--path_before", type=str, required=True, help="Path to ue score get from other ue method")
    parser.add_argument("--result_save_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--uncertainty_save_path", type=str, required=True, help="Path to save corrector scores")
    return parser.parse_args()

def main():
    args = parse_arguments()
    data = read_json_file(args.data_read_path)
    model_path= args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    labels = ['correct']
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, problem_type="multi_label_classification",
                                                                num_labels=1, id2label=id2label, label2id=label2id)
    uncertainty_save_path=args.uncertainty_save_path+str(model_path).replace('/','_')+"_uncertainty.json"
    print(uncertainty_save_path)
    if not os.path.exists(uncertainty_save_path):
        predictions = []
        for item in tqdm.tqdm(data):
            text = item['text']
            tokens = tokenizer.encode(text, add_special_tokens=False)
            length = len(tokens)
            if length > 512:
                last_512_tokens = tokens[-512:]
                decoded_text = tokenizer.decode(last_512_tokens, skip_special_tokens=True)
                text = decoded_text
            encoding = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
            outputs = model(**encoding)
            logits = outputs.logits
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits.squeeze().cpu())
            prediction = 1 if probs > 0.5 else 0
            pred = {'prompt': text, 'prediction': prediction, 'probs': float(probs), 'data_uncertainty': 1 - float(probs)}
            predictions.append(pred)
        with open(uncertainty_save_path, 'w') as file:
            json.dump(predictions, file, indent=4)
    else :
        with open(uncertainty_save_path, 'r') as file:
            predictions = json.load(file)
    
    with open(args.path_before, 'r') as file:
        data_old = json.load(file)
    true_labels = [1 - item['correct'] for item in data]
    add=0
    for j in range(len(data_old)):
        scores_old = data_old[j]["scores"]
        scores_new = [entry["data_uncertainty"] for entry in predictions]
        auc_max = 0.0
        weight_opt = 0.0

        # Split the data into the development set and the test set
        true_labels_dev, true_labels_test, scores_new_dev, scores_new_test, scores_old_dev, scores_old_test = train_test_split(
            true_labels, scores_new, scores_old, test_size=0.5, random_state=12
        )

        # Search for the best weights on the development set
        auc_max = -np.inf
        result_save_path=args.result_save_path
        with open(result_save_path, "a") as file:
            file.write(f"###########{data_old[j]['method']}\n"
                    f"\n\n\n")
        for i in range(1000):
            weight_new = float(i) / 1000
            weight_old = 1 - weight_new
            new_scores_dev = weight_new * np.array(scores_new_dev) + weight_old * np.array(scores_old_dev)
            auc_temp = roc_auc_score(true_labels_dev, new_scores_dev)
            if auc_temp > auc_max:
                auc_max = auc_temp
                weight_opt = weight_new

        auc_old = roc_auc_score(true_labels_test, scores_old_test)
        auc_new = roc_auc_score(true_labels_test, scores_new_test)
        
        ece_old = expected_calibration_error(true_labels_test, scores_old_test)
        ece_new = expected_calibration_error(true_labels_test, scores_new_test)
        f1_old = get_best_f1(true_labels_test, scores_old_test)
        f1_new = get_best_f1(true_labels_test, scores_new_test)

        print(f"({data_old[j]['method']}) AUC: {auc_old:.4f}\n"
                f"(training) AUC: {auc_new:.4f}\n"
                f"({data_old[j]['method']}) ECE: {ece_old:.4f}\n"
                f"(training) ECE: {ece_new:.4f}\n"
                f"({data_old[j]['method']}) f1: {f1_old:.4f}\n"
                f"(training) f1: {f1_new:.4f}\n")
        
        with open(result_save_path, "a") as file:
            file.write(f"({data_old[j]['method']}) AUC: {auc_old:.4f}\n"
                    f"(training) AUC: {auc_new:.4f}\n"
                    f"({data_old[j]['method']}) ECE: {ece_old:.4f}\n"
                    f"(training) ECE: {ece_new:.4f}\n"
                    f"({data_old[j]['method']}) f1: {f1_old:.4f}\n"
                    f"(training) f1: {f1_new:.4f}\n"
                    f"\n\n\n")

        # Calculate the metrics on the test set using the best weights
        new_scores_all = weight_opt * np.array(scores_new) + (1 - weight_opt) * np.array(scores_old)
        directory="result/correct/"+data_old[j]['method']+str(model_path).replace('/','_')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open("result/correct/"+data_old[j]['method']+str(model_path).replace('/','_')+"_uncertainty.json", "w") as file:
            json.dump(new_scores_all.tolist(), file)
        new_scores_test = weight_opt * np.array(scores_new_test) + (1 - weight_opt) * np.array(scores_old_test)
        auc_max=roc_auc_score(true_labels_test, new_scores_test)
        ece_new = expected_calibration_error(true_labels_test, new_scores_test)
        f1_new = get_best_f1(true_labels_test, new_scores_test)

        print(f"MAX AUC: {auc_max:.4f}\n"
            f"optimal weight: {weight_opt:.4f}\n"
            f"ece_new: {ece_new:.4f}\n"
            f"f1_new: {f1_new:.4f}\n")

        result_save_path =args.result_save_path  
        add+=(auc_max-auc_old)
        with open(result_save_path, "a") as file:
            file.write(f"MAX AUC: {auc_max:.4f}\n"
                    f"optimal weight: {weight_opt:.4f}\n"
                    f"ece_new: {ece_new:.4f}\n"
                    f"f1_new: {f1_new:.4f}\n"
                    f"\n\n\n")
            file.write(f"weight: {weight_opt:.4f}\n"
                    f"add_AUC: {auc_max-auc_old:.4f}\n"
                    f"add_ece: {ece_new-ece_old:.4f}\n"
                    f"add_f1: {f1_new-f1_old:.4f}\n"
                    f"\n\n\n")         

if __name__ == "__main__":
    main()


#CUDA_VISIBLE_DEVICES=${devices} python 05_inf.py --data_read_path ${data_read_path} --model_path ${save_path} --path_before ${path_before} --result_save_path ${result_save_path} --uncertainty_save_path ${uncertainty_save_path}
