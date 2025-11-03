from datasets import load_dataset
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

models = [
    "distilbert-base-uncased",
    "bert-base-uncased",
]

ds = load_dataset("jannalu/crows_pairs_multilingual", "english")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pseudo-log-probabilities comparison
def pll(model, tokenizer, sent):
    model = model.to(device)
    inputs = tokenizer(sent, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    indices_to_loop = (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id) & (input_ids != tokenizer.pad_token_id)
    indices_to_loop = indices_to_loop.nonzero(as_tuple=True)[1]

    current_pll = 0

    with torch.no_grad():
        for i in indices_to_loop:
            token_id = input_ids[0, i]

            masked_ip = input_ids.clone()
            masked_ip[0, i] = tokenizer.mask_token_id

            logits = model(masked_ip).logits
            mask_logit = logits[0, i, :]
            log_probs = torch.log_softmax(mask_logit, dim=-1)
            token_log_prob = log_probs[token_id].item()
            current_pll += token_log_prob

    return current_pll


if __name__ == "__main__":
    ds = ds["test"]
    print(ds)

    bias_types = set(ds["bias_type"])
    print("Bias types:", bias_types)
    threshold = 0.1


    results = {}
    for model_name in models:
        model = AutoModelForMaskedLM.from_pretrained(model_name, use_safetensors=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Evaluating model:", model_name)
        results[model_name] = {}
        for bias in tqdm(bias_types):
            subset = [item for item in ds if item["bias_type"] == bias]
            correct = 0
            total = 0
            for item in subset:
                sent1 = item["sent_more"] # more stereotypical
                sent2 = item["sent_less"] # less stereotypical
                pllmore = pll(model, tokenizer, sent1)
                pllless = pll(model, tokenizer, sent2)

                if pllmore > pllless:
                    correct += 1
                total += 1
            bias_score = correct / total if total > 0 else 0
            results[model_name][bias] = bias_score
    
    print("Final Results:", results)
    
    #save to json
    import json
    with open("bias_results.json", "w") as f:
        json.dump(results, f, indent=4)
            

        