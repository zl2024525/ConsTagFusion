import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, f1_score
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
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

# Model configurations
MODELS_TO_COMPARE = {
    "bert-base-chinese": {
        "type": "encoder",
        "path": os.path.join(MODEL_BASE_DIR, "bert-base-chinese"),
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "max_length": 256,
        "batch_size": 16,
        "learning_rate": 2e-5
    },
    "Llama-2-7b-hf": {
        "type": "decoder",
        "path": os.path.join(MODEL_BASE_DIR, "Llama-2-7b-hf"),
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "max_length": 384,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "peft": True,
        "lora_r": 8
    },
    "Qwen2.5-7B": {
        "type": "decoder",
        "path": os.path.join(MODEL_BASE_DIR, "Qwen2.5-7B"),
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "max_length": 384,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "peft": True,
        "lora_r": 8
    }
}


# Encoder model dataset
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


# Decoder model dataset
class DecoderDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = str(self.labels[idx])
        full_text = text + label
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"][0]
        label_start_idx = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        labels = input_ids.clone()
        labels[:label_start_idx] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels
        }

    def __len__(self):
        return len(self.texts)


# Format consumption tags (highlight high-frequency and risk-related behaviors)
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
            risk_flag = "【High Risk】" if any(
                keyword in k for keyword in ['credit card', 'repayment', 'tobacco', 'loan', 'securities']) else ""
            formatted.append(f"{k}:{v} times{risk_flag}")
        return "; ".join(formatted)
    except:
        return tags


def build_user_info(row):
    # Integrate consumption tags into user information
    consumption_tags = format_consumption_tags(row['Consumption_Tags'])
    return f"""User Information:
User ID: {row['id']}
Basic Attributes: {row['Age']} years old, {row['Gender']}, {row['City']}, {row['Education_level']}
Economic Status: Income {row['Salary_level']} yuan, {row['Housing_flag']} property, Asset score {row['Capital_score']}
Credit Data: Original limit {row['Credit_limit_original']} yuan, Utilization rate {row['Credit_limit_usage']:.6f}
Consumption Behavior (High Frequency): {consumption_tags}"""


def build_encoder_text(row):
    user_info = build_user_info(row)
    return f"{user_info}\n\nPlease determine whether this user will default in the next 6 months (1=default, 0=not default). Consider comprehensive consumption behavior, especially the impact of high-frequency behaviors related to credit cards and repayments."


def build_decoder_prompt(row):
    user_info = build_user_info(row)
    return f"""The following is the user's credit information and consumption behavior. Please predict whether they will default in the next 6 months (1=default, 0=not default).
【Key】: In consumption behavior, high-frequency credit card usage, frequent repayments, tobacco consumption, etc., may be related to default risk and should be重点 considered.
Output only 0 or 1, no other content.

{user_info}
Prediction result:"""


def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        if 'Consumption_Tags' not in df.columns:
            print("Warning: Consumption_Tags column not found in data, this field will be skipped")
        else:
            print(f"Data loaded successfully, {df.shape[0]} rows total, including Consumption_Tags column")
        return df
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    X = df.drop('Default_flag_6_months', axis=1)
    y = df['Default_flag_6_months']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state, stratify=y_temp
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


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


def decoder_predict(model, tokenizer, texts, max_length=256):
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                pred = int(output_text.strip())
                if pred not in [0, 1]:
                    pred = 0
            except:
                pred = 0
            predictions.append(pred)
    return np.array(predictions)


# Train encoder model
def train_encoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info):
    model_name = model_info["path"].split("/")[-1]

    train_texts = [build_encoder_text(row) for _, row in X_train.iterrows()]
    val_texts = [build_encoder_text(row) for _, row in X_val.iterrows()]
    test_texts = [build_encoder_text(row) for _, row in X_test.iterrows()]

    # Load tokenizer and model
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

    pos_weight = torch.tensor(sum(y_train == 0) / sum(y_train == 1))
    print(f"Class weight: {pos_weight.item():.4f}")

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
        eval_steps=steps_per_epoch,
        save_strategy="epoch",
        save_steps=steps_per_epoch,
        logging_steps=steps_per_epoch,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        seed=SEED,
        fp16=torch.cuda.is_available()
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    train_results = trainer.evaluate(train_dataset)
    train_preds = trainer.predict(train_dataset)
    train_probs = np.exp(train_preds.predictions[:, 1]) / (
                np.exp(train_preds.predictions[:, 0]) + np.exp(train_preds.predictions[:, 1]))
    train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
    train_ks = max(train_tpr - train_fpr)

    val_results = trainer.evaluate(val_dataset)
    val_preds = trainer.predict(val_dataset)
    val_probs = np.exp(val_preds.predictions[:, 1]) / (
                np.exp(val_preds.predictions[:, 0]) + np.exp(val_preds.predictions[:, 1]))
    val_fpr, val_tpr, _ = roc_curve(y_val, val_probs)
    val_ks = max(val_tpr - val_fpr)

    test_results = trainer.evaluate(test_dataset)
    test_preds = trainer.predict(test_dataset)
    test_probs = np.exp(test_preds.predictions[:, 1]) / (
                np.exp(test_preds.predictions[:, 0]) + np.exp(test_preds.predictions[:, 1]))
    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
    test_ks = max(test_tpr - test_fpr)

    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")

    return {
        'model_name': model_name,
        'train_auc': train_results['eval_auc'],
        'train_recall': train_results['eval_recall'],
        'train_precision': train_results['eval_precision'],
        'train_f1': train_results['eval_f1'],
        'train_ks': train_ks,
        'val_auc': val_results['eval_auc'],
        'val_recall': val_results['eval_recall'],
        'val_precision': val_results['eval_precision'],
        'val_f1': val_results['eval_f1'],
        'val_ks': val_ks,
        'test_auc': test_results['eval_auc'],
        'test_recall': test_results['eval_recall'],
        'test_precision': test_results['eval_precision'],
        'test_f1': test_results['eval_f1'],
        'test_ks': test_ks,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'train_probs': train_probs
    }


# Train decoder model
def train_decoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info):
    model_name = model_info["path"].split("/")[-1]
    print(f"\n===== Training decoder model: {model_name} =====")

    # Build text data containing consumption tags
    train_texts = [build_decoder_prompt(row) for _, row in X_train.iterrows()]
    val_texts = [build_decoder_prompt(row) for _, row in X_val.iterrows()]
    test_texts = [build_decoder_prompt(row) for _, row in X_test.iterrows()]

    # Load tokenizer and model
    tokenizer = model_info["tokenizer_class"].from_pretrained(model_info["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Setting pad_token_id to eos_token_id: {tokenizer.pad_token_id}")

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Dynamically get currently available device, ensure loading and training on the same device
    current_device = torch.cuda.current_device()  # Get currently available GPU device
    print(f"Using GPU device: {current_device}")

    model = model_info["model_class"].from_pretrained(
        model_info["path"],
        quantization_config=bnb_config,
        device_map={"": current_device},
        trust_remote_code=True
    )

    # LoRA configuration
    if model_info.get("peft", False):
        model = prepare_model_for_kbit_training(model)
        if "Qwen2.5-7B" in model_name:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = DecoderDataset(train_texts, y_train.tolist(), tokenizer, model_info["max_length"])
    val_dataset = DecoderDataset(val_texts, y_val.tolist(), tokenizer, model_info["max_length"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{TRAINED_MODEL_DIR}/{model_name}_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=model_info["learning_rate"],
        per_device_train_batch_size=model_info["batch_size"],
        per_device_eval_batch_size=model_info["batch_size"]*2,
        num_train_epochs=3,
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
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    

    train_preds = decoder_predict(model, tokenizer, train_texts)
    train_auc = roc_auc_score(y_train, train_preds)
    train_recall = recall_score(y_train, train_preds, zero_division=0)
    train_precision = precision_score(y_train, train_preds, zero_division=0)
    train_f1 = f1_score(y_train, train_preds, zero_division=0)
    train_fpr, train_tpr, _ = roc_curve(y_train, train_preds)
    train_ks = max(train_tpr - train_fpr)
    
    val_preds = decoder_predict(model, tokenizer, val_texts)
    val_auc = roc_auc_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds, zero_division=0)
    val_precision = precision_score(y_val, val_preds, zero_division=0)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)
    val_fpr, val_tpr, _ = roc_curve(y_val, val_preds)
    val_ks = max(val_tpr - val_fpr)
    
    test_preds = decoder_predict(model, tokenizer, test_texts)
    test_auc = roc_auc_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds, zero_division=0)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_preds)
    test_ks = max(test_tpr - test_fpr)
    

    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    return {
        'model_name': model_name,
        'train_auc': train_auc,
        'train_recall': train_recall,
        'train_precision': train_precision,
        'train_f1': train_f1,
        'train_ks': train_ks,
        'val_auc': val_auc,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'val_f1': val_f1,
        'val_ks': val_ks,
        'test_auc': test_auc,
        'test_recall': test_recall,
        'test_precision': test_precision,
        'test_f1': test_f1,
        'test_ks': test_ks,
        'val_probs': val_preds,
        'test_probs': test_preds,
        'train_probs': train_preds
    }


def main():
    file_path = ""
    df = load_data(file_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    all_results = []

    for model_key, model_info in MODELS_TO_COMPARE.items():
        if model_info["type"] == "encoder":
            result = train_encoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info)
        else:
            result = train_decoder_model(X_train, X_val, X_test, y_train, y_val, y_test, model_info)
        
        all_results.append(result)


    results_df = pd.DataFrame({
        'Model': [result['model_name'] for result in all_results],
        'Training Set AUC': [result['train_auc'] for result in all_results],
        'Training Set Recall': [result['train_recall'] for result in all_results],
        'Training Set Precision': [result['train_precision'] for result in all_results],
        'Training Set F1': [result['train_f1'] for result in all_results],
        'Training Set KS': [result['train_ks'] for result in all_results],
        'Validation Set AUC': [result['val_auc'] for result in all_results],
        'Validation Set Recall': [result['val_recall'] for result in all_results],
        'Validation Set Precision': [result['val_precision'] for result in all_results],
        'Validation Set F1': [result['val_f1'] for result in all_results],
        'Validation Set KS': [result['val_ks'] for result in all_results],
        'Test Set AUC': [result['test_auc'] for result in all_results],
        'Test Set Recall': [result['test_recall'] for result in all_results],
        'Test Set Precision': [result['test_precision'] for result in all_results],
        'Test Set F1': [result['test_f1'] for result in all_results],
        'Test Set KS': [result['test_ks'] for result in all_results]
    })

    results_df.to_csv(f"{TABLES_DIR}/model_comparison_results.csv", index=False)


if __name__ == "__main__":
    main()