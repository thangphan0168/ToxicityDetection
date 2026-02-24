import os
import glob
from collections import defaultdict
from typing import Any

import datasets
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from toxicity_detection.model import CrossLingualToxicityDetector
from toxicity_detection.dataloader import ToxicityDataset, MixedIterableToxicityDataset, ToxicityDataCollator


class CheckpointManager:
    """Manages model checkpoint saving and cleanup."""
    def __init__(self, save_path: str, num_checkpoints: int = 3):
        self.save_path = save_path
        self.num_checkpoints = num_checkpoints
        os.makedirs(save_path, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        step: int,
        val_metrics: dict[str, Any] | None = None,
        checkpoint_name: str | None = None
    ) -> str:
        """Save a training checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{step}.pth"
        
        checkpoint_path = os.path.join(self.save_path, checkpoint_name)
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        
        if val_metrics is not None:
            checkpoint_data['val_metrics'] = val_metrics
        
        torch.save(checkpoint_data, checkpoint_path)        
        return checkpoint_path
    
    def cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoint_files = glob.glob(os.path.join(self.save_path, "checkpoint_step_*.pth"))
        
        if len(checkpoint_files) > self.num_checkpoints:
            # Sort by step number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            for old_checkpoint in checkpoint_files[:-self.num_checkpoints]:
                os.remove(old_checkpoint)
    
    def save_best_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        val_metrics: dict[str, Any]
    ) -> None:
        """Save the best performing model."""
        best_model_path = os.path.join(self.save_path, "best_model.pth")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': val_metrics
        }, best_model_path)

     
class Trainer:
    def __init__(
        self,
        model_name: str,
        train_dataset: datasets.Dataset | list[datasets.Dataset],
        val_dataset: datasets.Dataset,
        languages: list[str],
        batch_size: int,
        learning_rate: float,
        max_steps_per_epoch: int | None = None,
        ratios: list[float] | None = None,
        grl_lambda: float = 1.0,
        adversarial_frequency: int = 1,
        warmup_steps: int = 0,
        accumulation_steps: int = 1,
        eval_steps: int | None = None,
        num_epochs: int = 1,
        max_num_checkpoints:int = 3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path: str = './model_checkpoints',
        toxicity_criterion: nn.Module | None = None,
        language_criterion: nn.Module | None = None,
        toxicity_weight: float = 1.0,
        language_weight: float = 1.0,
        resume_checkpoint: bool = False
    ):
        if isinstance(train_dataset, list) and max_steps_per_epoch is None:
            raise ValueError(
                "max_steps_per_epoch must be specified when train_dataset is a list"
            )
        self.model_name = model_name
        self.languages = languages
        self.num_languages = len(self.languages)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if ratios is None:
            self.ratios = [1] * len(train_dataset) if isinstance(train_dataset, list) else None
        else:
            self.ratios = ratios
        self.max_steps_per_epoch = max_steps_per_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.grl_lambda = grl_lambda
        self.adversarial_frequency = adversarial_frequency
        self.warmup_steps = warmup_steps
        self.accumulation_steps = accumulation_steps
        self.eval_steps = eval_steps
        self.num_epochs = num_epochs
        self.max_num_checkpoints = max_num_checkpoints
        self.device = device
        self.save_path = save_path
        self.best_val_f1 = 0
        self.step_count = 0
        self.toxicity_criterion = toxicity_criterion or nn.CrossEntropyLoss()
        self.language_criterion = language_criterion or nn.CrossEntropyLoss()
        self.toxicity_weight = toxicity_weight
        self.language_weight = language_weight
        self.resume_checkpoint = resume_checkpoint

    def _load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> tuple[int, int]:
        """
        Load the latest checkpoint (by step) if available.
        Returns:
            (start_epoch, start_step) where:
                start_epoch: epoch index to start from
                start_step: global step count to resume from
        """
        pattern = os.path.join(self.save_path, "checkpoint_step_*.pth")
        checkpoint_files = glob.glob(pattern)
        if not checkpoint_files:
            return 0, 0  # no checkpoint, start from scratch

        # Sort by step number
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
        print(f"Resuming from checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore epoch and step
        start_epoch = checkpoint.get('epoch', 0)
        start_step = checkpoint.get('step', 0)

        # If validation metrics were stored, update best_val_f1
        val_metrics = checkpoint.get('val_metrics')
        if val_metrics is not None and 'toxicity_macro_f1' in val_metrics:
            self.best_val_f1 = val_metrics['toxicity_macro_f1']

        return start_epoch, start_step

    def compute_loss(self, outputs, toxicity_labels, language_labels):
        """
        Compute combined loss for both tasks.
        
        Args:
            outputs: Model outputs dictionary
            toxicity_labels: Binary toxicity labels (batch_size,). Use -1 for unlabeled samples.
            language_labels: Language IDs (batch_size,)
            
        Returns:
            Dictionary with total loss and individual losses
        """
        labeled_mask = toxicity_labels != -1        
        if labeled_mask.any():
            labeled_logits = outputs['toxicity_logits'][labeled_mask, :]
            labeled_targets = toxicity_labels[labeled_mask]
            
            toxicity_loss = self.toxicity_criterion(labeled_logits, labeled_targets)
        else:
            toxicity_loss = torch.tensor(0.0, device=toxicity_labels.device)
        
        language_loss = self.language_criterion(
            outputs['language_logits'],
            language_labels
        )
        
        total_loss = (
            self.toxicity_weight * toxicity_loss +
            self.language_weight * language_loss
        )
        
        return {
            'total_loss': total_loss,
            'toxicity_loss': toxicity_loss,
            'language_loss': language_loss,
            'num_labeled': labeled_mask.sum().item()
        }

    def train_epoch(self, model, optimizer, scheduler, train_dataloader, val_dataloader, epoch, max_steps):
        total_loss = 0
        total_toxicity_loss = 0
        total_language_loss = 0
        total_labeled = 0
        checkpoint_manager = CheckpointManager(self.save_path, self.max_num_checkpoints)
        try:
            total_steps = min(len(train_dataloader), max_steps)
        except TypeError:
            total_steps = max_steps
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{self.num_epochs} (λ={1:.3f})",
            total=total_steps
        )
        
        optimizer.zero_grad()        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= total_steps:
                break
            model.train() 
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            toxicity_labels = batch['toxicity_label'].to(self.device)
            language_labels = batch['language_label'].to(self.device)

            # Set lambda to non zero only once every k batches
            if (batch_idx + 1) % self.adversarial_frequency == 0:
                model.set_grl_lambda(self.grl_lambda)
            else:
                model.set_grl_lambda(0)
            
            outputs = model(input_ids, attention_mask)
            losses = self.compute_loss(outputs, toxicity_labels, language_labels)
            loss = losses['total_loss']
            loss = loss / self.accumulation_steps
            loss.backward()
            
            total_loss += losses['total_loss'].item()
            total_toxicity_loss += losses['toxicity_loss'].item()
            total_language_loss += losses['language_loss'].item()
            total_labeled += losses['num_labeled']
            
            is_accumulation_step = (batch_idx + 1) % self.accumulation_steps == 0
            is_last_batch = (batch_idx + 1 == total_steps)
            
            if is_accumulation_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.step_count += 1
                
                if (self.eval_steps is not None and self.step_count % self.eval_steps == 0) or is_last_batch:
                    current_f1 = None
                    val_metrics = None
                    print("Cummulative Average Training Toxicity Loss: ", total_toxicity_loss / (batch_idx+1))
                    if val_dataloader is not None:
                        val_metrics = self.evaluate(model, val_dataloader)
                        current_f1 = val_metrics['toxicity_macro_f1']
                        print(f"Validation - Toxicity Loss: {val_metrics['toxicity_loss']}")
                        print(f"Validation - Macro F1: {current_f1:.4f}")
                        print(f"Toxicity F1:")
                        for lang in self.languages:
                            print(f"  - {lang}: {val_metrics['toxicity_f1'][lang]}")
                        if current_f1 > self.best_val_f1:
                            checkpoint_manager.save_best_model(
                                model, optimizer, scheduler, epoch, self.step_count, val_metrics
                            )
                            self.best_val_f1 = current_f1

                    checkpoint_manager.save_checkpoint(
                        model, optimizer, scheduler, epoch, self.step_count, val_metrics 
                    )
                    checkpoint_manager.cleanup_old_checkpoints()
            
            progress_bar.set_postfix({
                'loss': losses['total_loss'].item(),
                'tox_loss': losses['toxicity_loss'].item(),
                'lang_loss': losses['language_loss'].item(),
                'labeled': losses['num_labeled'],
                'step': self.step_count
            })
        
        avg_loss = total_loss / total_steps
        avg_tox_loss = total_toxicity_loss / total_steps
        avg_lang_loss = total_language_loss / total_steps
        
        return avg_loss, avg_tox_loss, avg_lang_loss, total_labeled
    
    def train_model(self):
        print(f"Training on device: {self.device}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = CrossLingualToxicityDetector.from_pretrained_base(
            model_name=self.model_name,
            num_languages=self.num_languages,
            grl_lambda=self.grl_lambda
        ).to(self.device)
        
        collator = ToxicityDataCollator(tokenizer)
        if not isinstance(self.train_dataset, list):
            train_dataset = ToxicityDataset(self.train_dataset, tokenizer, self.languages)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collator
            )
        else:
            train_dataset = MixedIterableToxicityDataset(
                datasets=self.train_dataset,
                ratios=self.ratios,
                languages=self.languages,
                tokenizer=tokenizer,
                batch_size=self.batch_size
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collator
            )
        
        val_dataset = ToxicityDataset(self.val_dataset, tokenizer, self.languages)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=collator
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        num_steps_per_epoch = len(train_loader) if self.max_steps_per_epoch is None else self.max_steps_per_epoch 
        total_steps = num_steps_per_epoch * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        start_epoch = 0
        self.step_count = 0
        if self.resume_checkpoint:
            start_epoch, self.step_count = self._load_latest_checkpoint(model, optimizer, scheduler)
            start_epoch = min(start_epoch, self.num_epochs - 1)
        
        model.train()
        for epoch in range(start_epoch, self.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_tox_loss, train_lang_loss, train_labeled = self.train_epoch(
                model, optimizer, scheduler, train_loader, val_loader, epoch, num_steps_per_epoch
            )
            print(f"\nTraining - Loss: {train_loss:.4f}, "
                f"Toxicity Loss: {train_tox_loss:.4f}, "
                f"Language Loss: {train_lang_loss:.4f}, "
                f"Labeled Samples: {train_labeled}")
            
        # Evaluate
        val_metrics = self.evaluate(model, val_loader)
        print(f"\nValidation Metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Toxicity Loss: {val_metrics['toxicity_loss']:.4f}")
        print(f"  Language Loss: {val_metrics['language_loss']:.4f}")
        print(f"  Toxicity F1:")
        for lang in self.languages:
            print(f"  - {lang}: {val_metrics['toxicity_f1'][lang]}")
        print(f"  - Avg: {val_metrics['toxicity_macro_f1']:.4f}")
        print(f"  Labeled Samples: {val_metrics['num_labeled_samples']}")
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"{'='*50}")
        
        return model, tokenizer
    
    def evaluate(self, model, dataloader):
        model.eval()
        total_loss = 0
        toxicity_loss = 0
        language_loss = 0
        all_toxicity_preds = {lang: [] for lang in self.languages}
        all_toxicity_labels = {lang: [] for lang in self.languages}
        total_labeled = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                toxicity_labels = batch['toxicity_label'].to(self.device)
                language_labels = batch['language_label'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                losses = self.compute_loss(outputs, toxicity_labels, language_labels)
                
                total_loss += losses['total_loss'].item()
                toxicity_loss += losses['toxicity_loss'].item()
                language_loss += losses['language_loss'].item()
                total_labeled += losses['num_labeled']
                
                # Get predictions
                toxicity_preds = torch.argmax(outputs['toxicity_logits'], dim=-1).cpu().numpy()
                toxicity_labels = toxicity_labels.cpu().numpy()
                language_labels = language_labels.cpu().numpy()

                for (toxic_pred, toxic_label, lang_id) in zip(toxicity_preds, toxicity_labels, language_labels):
                    lang = self.languages[lang_id]
                    all_toxicity_preds[lang].append(toxic_pred)
                    all_toxicity_labels[lang].append(toxic_label)
        
        avg_loss = total_loss / len(dataloader)
        avg_toxcity_loss = toxicity_loss / len(dataloader)
        avg_language_loss = language_loss / len(dataloader)
        
        metrics: dict[str, Any] = defaultdict(dict)
        for lang in self.languages:
            if len(all_toxicity_labels[lang]) > 0:
                tox_precision, tox_recall, tox_f1, _ = precision_recall_fscore_support(
                    all_toxicity_labels[lang], all_toxicity_preds[lang], average='binary', zero_division=0
                )
                metrics["toxicity_precision"][lang] = tox_precision
                metrics["toxicity_recall"][lang] = tox_recall
                metrics["toxicity_f1"][lang] = tox_f1
            else:
                metrics["toxicity_precision"][lang] = 0
                metrics["toxicity_recall"][lang] = 0
                metrics["toxicity_f1"][lang] = 0

        metrics = dict(metrics)
        macro_f1 = sum(metrics["toxicity_f1"].values()) / len(metrics['toxicity_f1'])
        metrics['toxicity_macro_f1'] = macro_f1
        metrics['loss'] = avg_loss
        metrics['toxicity_loss'] =  avg_toxcity_loss
        metrics['language_loss'] = avg_language_loss
        metrics['num_labeled_samples'] = total_labeled
        
        return metrics


if __name__ == "__main__":
    from datasets import load_dataset, concatenate_datasets

    train_dataset = load_dataset("parquet", data_files="data/train_combined.parquet", split="train")
    dev_dataset = load_dataset("parquet", data_files="data/dev.parquet", split="train")
    languages = ["en", "fi", "de"]
    n_train_samples = 10
    n_dev_samples = 4

    def stratified_sample_equal_per_lang(dataset, languages, total_samples, seed=0):
        """Sample equally from each language in `languages`."""
        num_langs = len(languages)
        base_n = total_samples // num_langs
        remainder = total_samples % num_langs
        per_lang_counts = {lang: base_n for lang in languages}
        for lang in languages[:remainder]:
            per_lang_counts[lang] += 1

        per_lang_datasets = []
        for lang in languages:
            lang_subset = dataset.filter(lambda ex, l=lang: ex["lang"] == l)
            lang_subset = lang_subset.shuffle(seed=seed)
            n = per_lang_counts[lang]
            if n > len(lang_subset):
                raise ValueError(f"Requested {n} samples for language '{lang}', "
                                f"but only {len(lang_subset)} available.")
            per_lang_datasets.append(lang_subset.select(range(n)))

        combined = concatenate_datasets(per_lang_datasets).shuffle(seed=seed)
        return combined
    
    train_samples = stratified_sample_equal_per_lang(
        train_dataset, languages, total_samples=n_train_samples, seed=0
    )
    dev_samples = stratified_sample_equal_per_lang(
        dev_dataset, languages, total_samples=n_dev_samples, seed=1
    )
    
    trainer = Trainer(
        model_name="google/gemma-3-270m",
        train_dataset=train_samples,
        val_dataset=dev_samples,
        languages=["en", "fi", "de"],
        batch_size=1,
        learning_rate=1e-4,
        warmup_steps=1,
        accumulation_steps=2,
        eval_steps=1,
        num_epochs=1,
        max_num_checkpoints=1
    )
    model, tokenizer = trainer.train_model()
