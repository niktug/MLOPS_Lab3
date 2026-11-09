"""Обучение классификатора токсичности: Hydra + Transformers + MLflow."""
import random
import numpy as np
import torch
import pandas as pd
import mlflow
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import hydra
from omegaconf import DictConfig


def set_seed(seed: int):
    """Фиксируем seed везде, где требуется"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # у нас CPU, но пусть будет


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    set_seed(cfg.training.seed)

    # ---------- Данные ----------
    train_df = pd.read_csv(cfg.data.train_path)
    val_df = pd.read_csv(cfg.data.val_path)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # ---------- Токенизация ----------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    def tokenize(examples):
        return tokenizer(
            examples[cfg.data.text_column],
            truncation=True,
            max_length=cfg.model.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    # Trainer требует колонку 'labels'
    train_ds = train_ds.rename_column(cfg.data.label_column, "labels")
    val_ds = val_ds.rename_column(cfg.data.label_column, "labels")

    # ---------- Модель ----------
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
    )

    # ---------- TrainingArguments ----------
    args = TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=cfg.training.seed,
    )

    # ---------- MLflow ----------
    mlflow.set_experiment("toxicity-rubert")

    with mlflow.start_run():
        # 1. Гиперпараметры из Hydra-конфига
        mlflow.log_params({
            "model_name": cfg.model.name,
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "seed": cfg.training.seed,
            "max_length": cfg.model.max_length,
        })

        # 2. Сохраняем сам конфиг как артефакт
        mlflow.log_artifact("config/config.yaml", artifact_path="config")

        # 3. Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )

        # 4. Обучение
        trainer.train()

        # 5. Финальные метрики
        metrics = trainer.evaluate()
        mlflow.log_metrics({
            "final_eval_loss": metrics["eval_loss"],
            "final_eval_accuracy": metrics["eval_accuracy"],
            "final_eval_f1": metrics["eval_f1"],
        })

        # 6. Чекпоинт модели как артефакт
        trainer.save_model("./final_model")
        mlflow.log_artifact("./final_model", artifact_path="model")

        # 7. Данные (для полноты)
        mlflow.log_artifact(cfg.data.train_path, artifact_path="data")
        mlflow.log_artifact(cfg.data.val_path, artifact_path="data")

        print(f"\n{'='*50}")
        print(f"F1:       {metrics['eval_f1']:.4f}")
        print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"Loss:     {metrics['eval_loss']:.4f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    train()