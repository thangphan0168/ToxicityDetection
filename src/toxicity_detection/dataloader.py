import itertools
import random

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import DataCollatorWithPadding


class ToxicityDataset(Dataset):
    """
    Dataset for toxicity detection with language labels.
    """
    def __init__(self, hf_dataset, tokenizer, languages, max_length=1024):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_labels = {lang: idx for idx, lang in enumerate(languages)}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset["text"][idx]
        toxicity_label = self.dataset["label"][idx]
        language_label = self.language_labels[self.dataset["lang"][idx]]
        
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'toxicity_label': toxicity_label,
            'language_label': language_label
        }
    

class MixedIterableToxicityDataset(IterableDataset):
    """Iterates through multiple datasets with specified ratios per batch"""
    def __init__(self, datasets, ratios, languages, tokenizer, batch_size, max_length=1024):
        self.datasets = datasets
        self.ratios = [r / sum(ratios) for r in ratios]
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_labels = {lang: idx for idx, lang in enumerate(languages)}
        
        # Calculate samples per dataset per batch
        self.samples_per_dataset = [max(1, int(batch_size * ratio)) for ratio in self.ratios]
        diff = batch_size - sum(self.samples_per_dataset)
        self.samples_per_dataset[0] += diff
        
    def __iter__(self):
        # Infinite iterators for each dataset
        iterators = [
            itertools.cycle(range(len(dataset))) 
            for dataset in self.datasets
        ]
        
        while True:
            batch = []
            for dataset_idx, num_samples in enumerate(self.samples_per_dataset):
                for _ in range(num_samples):
                    idx = next(iterators[dataset_idx])
                    batch.append(self.datasets[dataset_idx][idx])
            
            random.shuffle(batch)
            for item in batch:
                text = item["text"]
                toxicity_label = item["label"]
                language_label = self.language_labels[item["lang"]]
                
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length
                )
                yield {
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
    train_dataset = load_dataset("parquet", data_files="data/train_combined_v2.parquet", split="train")
    en_dataset = train_dataset.filter(lambda x: x['lang'] == 'en')
    fi_dataset = train_dataset.filter(lambda x: x['lang'] == 'fi')
    de_dataset = train_dataset.filter(lambda x: x['lang'] == 'de')

    # toxic_dataset = ToxicityDataset(train_dataset, tokenizer, languages=['en', 'fi', 'de'])
    toxic_dataset = MixedIterableToxicityDataset(
        datasets = [en_dataset, fi_dataset, de_dataset],
        ratios=[2, 1, 1],
        languages=['en', 'fi', 'de'],
        batch_size=16,
        tokenizer=tokenizer
    )
    collator = ToxicityDataCollator(tokenizer)
    train_dataloader = DataLoader(
        toxic_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator
    )
    batch = next(iter(train_dataloader))
    print(batch)
