from datasets import load_dataset #load the dataset from HuggingFace datasets library

# Load WMT14 German-English
ds = load_dataset("wmt14", "de-en")

splits = ["train", "validation", "test"]

for split in splits:
    de_path = f"wmt14.{split}.de"
    en_path = f"wmt14.{split}.en"

    with open(de_path, "w", encoding="utf-8") as f_de, \
         open(en_path, "w", encoding="utf-8") as f_en:
        for ex in ds[split]:
            f_de.write(ex["translation"]["de"] + "\n")
            f_en.write(ex["translation"]["en"] + "\n")

    print(f"Saved {split} split: {de_path}, {en_path}")
