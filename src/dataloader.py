import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

class ToxicityDataset(Dataset):
    """
    Dataset for toxicity detection with language labels.
    """
    def __init__(self, hf_dataset, tokenizer, languages):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.language_labels = {lang: idx for idx, lang in enumerate(languages)}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset["text"][idx]
        toxicity_label = self.dataset["label"][idx]
        language_label = self.language_labels[self.dataset["lang"][idx]]
        
        tokenized = self.tokenizer(
            text,
            truncation=False
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'toxicity_label': toxicity_label,
            'language_label': language_label
        }


class ToxicityDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        toxicity_labels = [f.pop('toxicity_label') for f in features]
        language_labels = [f.pop('language_label') for f in features]
        
        batch = super().__call__(features)
        batch['toxicity_label'] = torch.tensor(toxicity_labels, dtype=torch.long)
        batch['language_label'] = torch.tensor(language_labels, dtype=torch.long)
        
        return batch


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    train_dataset = load_dataset("parquet", data_files="data/train_combined.parquet", split="train")
    toxic_dataset = ToxicityDataset(train_dataset, tokenizer, languages=['en', 'fi', 'de'])
    collator = ToxicityDataCollator(tokenizer)
    train_dataloader = DataLoader(
        toxic_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator
    )
    batch = next(iter(train_dataloader))
    print(batch)
