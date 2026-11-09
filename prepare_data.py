"""Выгружает подмножество AlexSham/Toxic_Russian_Comments в CSV."""
import pandas as pd
from datasets import load_dataset


def main():
    ds = load_dataset("AlexSham/Toxic_Russian_Comments", split="train")

    # Перемешиваем с фиксированным seed — для воспроизводимости
    ds = ds.shuffle(seed=42)

    # Подмножество: 800 train, 200 val
    train_subset = ds.select(range(800))
    val_subset = ds.select(range(800, 1000))

    train_df = pd.DataFrame(train_subset)
    val_df = pd.DataFrame(val_subset)

    # Сохраняем в CSV
    train_df.to_csv("data/train_subset.csv", index=False)
    val_df.to_csv("data/val_subset.csv", index=False)

    # Отчёт
    print(f"Train: {len(train_df)} примеров")
    print(f"Val:   {len(val_df)} примеров")
    print("Колонки:", train_df.columns.tolist())
    print("\nРаспределение label в train:")
    print(train_df["label"].value_counts())
    print("\nРаспределение label в val:")
    print(val_df["label"].value_counts())


if __name__ == "__main__":
    main()