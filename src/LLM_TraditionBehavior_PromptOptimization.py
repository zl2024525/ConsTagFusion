import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from datetime import datetime
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_BASE_DIR = ""
PICTURES_DIR = ""
TABLES_DIR = ""
TRAINED_MODEL_DIR = ""
os.makedirs(PICTURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

# Model configuration
MODELS_TO_COMPARE = {
    "bert-base-chinese": {
        "type": "encoder",
        "path": os.path.join(MODEL_BASE_DIR, "bert-base-chinese"),
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "max_length": 512,
        "batch_size": 16,
        "learning_rate": 2e-5
    },
    "Llama-2-7b-hf": {
        "type": "decoder",
        "path": os.path.join(MODEL_BASE_DIR, "Llama-2-7b-hf"),
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "max_length": 1024,
        "batch_size": 1,
        "learning_rate": 5e-5,
        "peft": True,
        "lora_r": 16
    },
    "Qwen2.5-7B": {
        "type": "decoder",
        "path": os.path.join(MODEL_BASE_DIR, "Qwen2.5-7B"),
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "max_length": 1024,
        "batch_size": 1,
        "learning_rate": 5e-5,
        "peft": True,
        "lora_r": 16
    }
}


class EncoderDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class DecoderDataset(Dataset):
    def __init__(self, prompts, labels, tokenizer, max_length):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]

        expected_output = f'{{"id": "{self._extract_id(prompt)}", "final_prediction": {label}}}'
        full_text = f"{prompt}\n{expected_output}"

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"][0]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        label_start_idx = len(prompt_ids) + 1
        labels = input_ids.clone()
        labels[:label_start_idx] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels
        }

    def _extract_id(self, prompt):
        match = re.search(r'"id":\s*"(\d+)"', prompt)
        return match.group(1) if match else f"unknown_{id(self)}"

    def __len__(self):
        return len(self.prompts)


def format_risk_details(details):
    if pd.isna(details) or not isinstance(details, str):
        return "No consumption risk details"

    high_risk_pattern = re.compile(r'Risk level ([45])(.*?→)', re.DOTALL)
    high_risk_matches = high_risk_pattern.findall(details)
    high_risk = [f"Level {k}: {v.strip()}" for k, v in high_risk_matches]

    overlap_pattern = re.compile(r'(\d+ or more medium risk factors.*?→.*?risk)', re.DOTALL)
    overlap_matches = overlap_pattern.findall(details)

    formatted = []
    if high_risk:
        formatted.append(f"High-risk behavior (level 4-5): {'; '.join(high_risk[:5])}")
    if overlap_matches:
        formatted.append(f"Risk overlap: {'; '.join(overlap_matches[:2])}")
    if not formatted:
        formatted.append("No significant high-risk behavior or overlapping risks")
    return "\n".join(formatted)


def build_user_info_dict(row):
    return {
        "id": str(row['id']),
        "Age": row['Age'],
        "Gender": row['Gender'],
        "City": row['City'],
        "Education_level": row['Education_level'],
        "Salary_level": row['Salary_level'],
        "Housing_flag": row['Housing_flag'],
        "Capital_score": row['Capital_score'],
        "Credit_limit_original": row['Credit_limit_original'],
        "Credit_limit_usage": round(row['Credit_limit_usage'], 6),
        "Consumption_Tags": format_consumption_tags(row['Consumption_Tags']),
        "Consumption_risks": row['Consumption_risks'],
        "Consumption_risks_details": format_risk_details(row['Consumption_risks_details'])
    }


def format_consumption_tags(tags):
    if pd.isna(tags) or not isinstance(tags, str):
        return ""
    tag_list = [t.strip() for t in tags.split(';') if t.strip()]
    try:
        tag_pairs = [tuple(t.split(':')) for t in tag_list]
        tag_pairs = [(k, int(v)) for k, v in tag_pairs if v.isdigit()]
        tag_pairs.sort(key=lambda x: x[1], reverse=True)
        top_tags = tag_pairs[:15]
        formatted = []
        for k, v in top_tags:
            risk_flag = "[HIGH RISK]" if any(keyword in k for keyword in ['']) else ""  # you can design your own
            formatted.append(f"{k}:{v} times{risk_flag}")
        return "; ".join(formatted)
    except:
        return tags


def build_encoder_text(row):
    user_info = build_user_info_dict(row)
    return f"""User information:
Age: {user_info['Age']} years, Gender: {user_info['Gender']}, City: {user_info['City']}
Education level: {user_info['Education_level']}, Income: {user_info['Salary_level']} yuan
Housing status: {user_info['Housing_flag']}, Asset score: {user_info['Capital_score']}
Credit limit: {user_info['Credit_limit_original']} yuan, Usage rate: {user_info['Credit_limit_usage']}
Consumption risk score: {user_info['Consumption_risks']}
High-risk consumption behavior: {format_risk_details(user_info['Consumption_risks_details']).replace('n', '; ')}
High-frequency consumption: {user_info['Consumption_Tags']}

Please determine whether this user will default within the next 6 months (1=default, 0=no default). Focus on the combined impact of consumption risk score, high-risk consumption behavior, and credit usage rate."""


def build_decoder_prompt(row):
    user_info = build_user_info_dict(row)
    return f"""1. As a senior financial risk control expert, predict default within 6 months (0=no default/1=default) based on user information. Output only a complete JSON as required. All judgments must align with financial risk logic, and default probability must be closely tied to the analysis.

2. [Field Explanation] (Must understand before analysis):
- id: Unique user identifier
- Age: Age in years
- Gender: Male/Female
- City: City of residence (reflects regional economic level)
- Education_level: Education level (affects income potential)
- Salary_level: Monthly income (directly reflects repayment ability)
- Housing_flag: Whether owns property (1=yes/0=unknown, reflects asset stability)
- Capital_score: Asset level (higher score means more assets)
- Credit_limit_original: Original approved credit limit
- Credit_limit_usage: Credit usage rate after 6 months (higher usage means higher current debt)
- Consumption_risks: Consumption risk score (1-5, higher means riskier)
- Consumption_risks_details: Detailed consumption risk analysis

3. [Financial Expert Experience Reference] (Must combine with industry consensus):
1. Repayment ability formula: Stable income (Salary_level) > fixed expenses (mortgage/rent + necessary consumption) is the primary low-risk condition; users with volatile income need attention.
2. Asset buffer principle: Users with property (Housing_flag=1) and sufficient assets (Capital_score>70) typically have 30% lower default probability, buffering short-term income fluctuations.
3. Credit usage warning: Credit usage rate (Credit_limit_usage) >60% significantly increases default risk; if combined with low income (Salary_level<5000), risk doubles.
4. Consumption structure signal: Users with high-risk consumption (Consumption_risks≥4) >30% have 2.5x higher default probability; especially beware of "investment + luxury + alcohol/tobacco" combinations.
5. Risk stacking effect: 3+ medium-risk factors (e.g., age<25, credit usage 40-60%, no property) increase default probability from 15% (single factor) to 40%+.
6. Regional differences: Users in tier-1 cities (e.g., Beijing, Shanghai) typically have 10-15% lower default rates than tier-3/4 cities, but high living costs mean high income ≠ low risk.
7. Age and repayment behavior: Users aged 25-40 have higher default risk due to family/consumption needs than stable 40-55 age group; for 55+, check income stability.

4. [Core Task]
Output JSON with "id", "single_variable_analysis", "interaction_effects", "default_probability", "final_prediction":
1. id: Same as in user info
2. single_variable_analysis: Format "variable:value|||intermediate variable|||default impact", e.g.:
   "Age:29|||young|||high consumption needs may increase debt→medium default risk"
   Cover all 12 fields, each must reflect corresponding expert principle.
3. interaction_effects: Format "combination:logic analysis (with expert experience)", at least 7 groups, e.g.:
   "combination:Salary_level(5000)+Credit_limit_usage(70%)|||medium income but credit usage exceeds warning|||insufficient repayment + high debt→high default risk"
4. default_probability: Percentage (e.g., "60%"), must be based on comprehensive analysis, briefly state key judgment (e.g., "due to high debt + no property + consumption risk stacking").
5. final_prediction: 1 if probability >=50%, else 0

5. [User Information]
{json.dumps(user_info, ensure_ascii=False, indent=2)}

6. [Output Format Example]
{{
    "id": "1001",
    "single_variable_analysis": "Age:29|||young|||high consumption needs→medium default risk|||Gender:male|||no significant correlation|||no impact on default risk|||...",
    "interaction_effects": "combination:Salary_level(5000)+Credit_limit_usage(70%)|||medium income but credit usage exceeds warning|||insufficient repayment + high debt→high default risk|||...",
    "default_probability": "65% (key basis: high debt + no property + consumption risk stacking)",
    "final_prediction": 1
}}

Return JSON exactly as above. No extra explanation. Ensure complete JSON format; missing fields or format errors will cause prediction failure.
"""


def contrastive_loss(logits, labels, temperature=0.1):
    if logits.dim() > 1:
        logits = logits.squeeze()
    logits_norm = F.normalize(logits, dim=0)
    similarity = torch.outer(logits_norm, logits_norm) / temperature
    positive_mask = (labels.view(-1, 1) == labels.view(1, -1)).float()
    positive_mask.fill_diagonal_(0)
    negative_mask = 1 - positive_mask
    log_prob = F.log_softmax(similarity, dim=1)
    loss = -torch.mean(torch.sum(positive_mask * log_prob, dim=1))
    return loss


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        lm_loss = outputs.loss

        labels = inputs["labels"]
        valid_mask = (labels != -100)
        if not valid_mask.any():
            return (lm_loss, outputs) if return_outputs else lm_loss

        logits = outputs.logits[valid_mask]
        ones_token_id = self.tokenizer.encode("1", add_special_tokens=False)[0]
        zeros_token_id = self.tokenizer.encode("0", add_special_tokens=False)[0]
        pred_probs = F.softmax(logits, dim=-1)[:, ones_token_id]
        true_labels = (labels[valid_mask] == ones_token_id).float()

        contrast_loss = contrastive_loss(pred_probs, true_labels, temperature=0.1)
        total_loss = lm_loss + 0.3 * contrast_loss
        return (total_loss, outputs) if return_outputs else total_loss


def decoder_predict(model, tokenizer, prompts, max_length=1024):
    model.eval()
    predictions = []
    probabilities = []

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=500,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None
    )

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )

            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            try:
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1
                json_str = generated_text[json_start:json_end]
                result = json.loads(json_str)

                prob_str = result.get("default_probability", "50%")
                prob = float(prob_str.strip('%')) / 100.0
                pred = result.get("final_prediction", 0)
            except:
                prob = 0.5
                pred = 1 if "default" in generated_text or "1" in generated_text else 0

            probabilities.append(prob)
            predictions.append(pred)

    return np.array(predictions), np.array(probabilities)


def train_encoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info):
    model_name = model_info["path"].split("/")[-1]

    train_texts = [build_encoder_text(row) for _, row in X_train.iterrows()]
    val_texts = [build_encoder_text(row) for _, row in X_val.iterrows()]
    test_texts = [build_encoder_text(row) for _, row in X_test.iterrows()]

    tokenizer = model_info["tokenizer_class"].from_pretrained(model_info["path"])
    model = model_info["model_class"].from_pretrained(
        model_info["path"],
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    train_dataset = EncoderDataset(train_texts, y_train.tolist(), tokenizer, model_info["max_length"])
    val_dataset = EncoderDataset(val_texts, y_val.tolist(), tokenizer, model_info["max_length"])
    test_dataset = EncoderDataset(test_texts, y_test.tolist(), tokenizer, model_info["max_length"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{TRAINED_MODEL_DIR}/{model_name}_{timestamp}"
    steps_per_epoch = len(train_dataset) // model_info["batch_size"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=model_info["learning_rate"],
        per_device_train_batch_size=model_info["batch_size"],
        per_device_eval_batch_size=model_info["batch_size"] * 2,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available()
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        pred_probs = np.exp(logits[:, 1]) / (np.exp(logits[:, 0]) + np.exp(logits[:, 1]))
        return {
            'auc': roc_auc_score(labels, pred_probs),
            'recall': recall_score(labels, predictions, zero_division=0),
            'precision': precision_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    train_preds = trainer.predict(train_dataset)
    train_probs = np.exp(train_preds.predictions[:, 1]) / (
                np.exp(train_preds.predictions[:, 0]) + np.exp(train_preds.predictions[:, 1]))
    train_pred_labels = np.argmax(train_preds.predictions, axis=-1)

    val_preds = trainer.predict(val_dataset)
    val_probs = np.exp(val_preds.predictions[:, 1]) / (
                np.exp(val_preds.predictions[:, 0]) + np.exp(val_preds.predictions[:, 1]))
    val_pred_labels = np.argmax(val_preds.predictions, axis=-1)

    test_preds = trainer.predict(test_dataset)
    test_probs = np.exp(test_preds.predictions[:, 1]) / (
                np.exp(test_preds.predictions[:, 0]) + np.exp(test_preds.predictions[:, 1]))
    test_pred_labels = np.argmax(test_preds.predictions, axis=-1)

    def calculate_metrics(y_true, y_pred, y_prob):
        return {
            'auc': roc_auc_score(y_true, y_prob),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'ks': max(roc_curve(y_true, y_prob)[0] - roc_curve(y_true, y_prob)[1])
        }

    train_metrics = calculate_metrics(y_train, train_pred_labels, train_probs)
    val_metrics = calculate_metrics(y_val, val_pred_labels, val_probs)
    test_metrics = calculate_metrics(y_test, test_pred_labels, test_probs)


def train_decoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info):
    model_name = model_info["path"].split("/")[-1]

    train_prompts = [build_decoder_prompt(row) for _, row in X_train.iterrows()]
    val_prompts = [build_decoder_prompt(row) for _, row in X_val.iterrows()]
    test_prompts = [build_decoder_prompt(row) for _, row in X_test.iterrows()]

    tokenizer = model_info["tokenizer_class"].from_pretrained(model_info["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    current_device = torch.cuda.current_device()
    model = model_info["model_class"].from_pretrained(
        model_info["path"],
        quantization_config=bnb_config,
        device_map={"": current_device},
        trust_remote_code=True
    )

    if model_info.get("peft", False):
        model = prepare_model_for_kbit_training(model)
        if "Llama-2-7b-hf" in model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "Qwen2.5-7B" in model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
        lora_config = LoraConfig(
            r=model_info["lora_r"],
            lora_alpha=64,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = DecoderDataset(train_prompts, y_train.tolist(), tokenizer, model_info["max_length"])
    val_dataset = DecoderDataset(val_prompts, y_val.tolist(), tokenizer, model_info["max_length"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{TRAINED_MODEL_DIR}/{model_name}_{timestamp}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=model_info["learning_rate"],
        per_device_train_batch_size=model_info["batch_size"],
        per_device_eval_batch_size=model_info["batch_size"] * 2,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    train_pred_labels, train_probs = decoder_predict(model, tokenizer, train_prompts)
    val_pred_labels, val_probs = decoder_predict(model, tokenizer, val_prompts)
    test_pred_labels, test_probs = decoder_predict(model, tokenizer, test_prompts)

    def calculate_metrics(y_true, y_pred, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {
            'auc': roc_auc_score(y_true, y_prob),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'ks': max(tpr - fpr)
        }

    train_metrics = calculate_metrics(y_train, train_pred_labels, train_probs)
    val_metrics = calculate_metrics(y_val, val_pred_labels, val_probs)
    test_metrics = calculate_metrics(y_test, test_pred_labels, test_probs)


def main():
    file_path = ""
    df = pd.read_excel(file_path)

    X = df.drop('Default_flag_6_months', axis=1)
    y = df['Default_flag_6_months']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    all_results = []

    for model_key, model_info in MODELS_TO_COMPARE.items():
        if model_info["type"] == "encoder":
            result = train_encoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info)
        else:
            result = train_decoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info)

        all_results.append(result)

    results_df = pd.DataFrame({
        'Model': [r['model_name'] for r in all_results],
        'Train AUC': [r['train_auc'] for r in all_results],
        'Train Recall': [r['train_recall'] for r in all_results],
        'Train Precision': [r['train_precision'] for r in all_results],
        'Train F1': [r['train_f1'] for r in all_results],
        'Train KS': [r['train_ks'] for r in all_results],
        'Val AUC': [r['val_auc'] for r in all_results],
        'Val Recall': [r['val_recall'] for r in all_results],
        'Val Precision': [r['val_precision'] for r in all_results],
        'Val F1': [r['val_f1'] for r in all_results],
        'Val KS': [r['val_ks'] for r in all_results],
        'Test AUC': [r['test_auc'] for r in all_results],
        'Test Recall': [r['test_recall'] for r in all_results],
        'Test Precision': [r['test_precision'] for r in all_results],
        'Test F1': [r['test_f1'] for r in all_results],
        'Test KS': [r['test_ks'] for r in all_results]
    })

    results_df.to_csv(f"", index=False)


if __name__ == "__main__":
    main()