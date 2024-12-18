# Standard library imports
import gc
import logging
import os
import sys
import warnings
from typing import Dict

# Third-party library imports
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Hugging Face imports
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    DataCollatorForSeq2Seq,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    NllbTokenizerFast,
    # NllbForConditionalGeneration,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoModelForCausalLM,
    OPTForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline
)
import evaluate

# Environment and system setup
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()

# %%
def initialize_logs(logs_dir, output_dir_name):
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{output_dir_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Log file location: {log_file}")
    logger.info("=" * 50)

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)


    pl_logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
    class IgnorePLFilter(logging.Filter):
        def filter(self, record):
            return not any(msg in record.getMessage() for msg in [
                'available:', 
                'TPU available',
                'HPU available'
            ])
    pl_logger.addFilter(IgnorePLFilter())

    return logger

# %%
def import_data(name, subset=-1):
    login(os.environ.get("HUGGING_FACE_TOKEN"))
    if name == "IssakaAI/en-tw":
        dataset = load_dataset('IssakaAI/en-tw')
    else:
        print("Invalid Data Name")

    # Reduce the dataset size
    if subset > 0:
        train_size = subset
        test_size = round(subset/10)
        train_dataset = dataset['train'].shuffle(seed=42).select(range(train_size))
        test_dataset = dataset['test'].shuffle(seed=42).select(range(test_size))
        dataset = DatasetDict({'train': train_dataset,'test': test_dataset})

    return dataset

# %%
def evaluate_model(translator, dataset, src_col, tgt_col, logger, subset):
    try:
        test_sample = dataset['test'].shuffle(seed=42).select(range(subset))

        # log src_col and tgt_col
        logger.info(f"Source column: {src_col}")
        logger.info(f"Target column: {tgt_col}")
        
        # Collect all translations
        logger.info("Generating translations for test sample...")
        all_translations = []
        all_references = []
        source_texts = []
        
        for instance in tqdm(test_sample):
            src_text = instance[src_col]
            reference_english = instance[tgt_col]
            translated = translator.translate(src_text)

            all_translations.append(translated)
            all_references.append(reference_english)
            source_texts.append(src_text)
        
        # Initialize metrics dictionary
        metrics = {
            'bleu': 0.0,
            'sacrebleu': 0.0,
            'comet': 0.0,
            'meteor': 0.0
        }
        
        # Regular BLEU
        try:
            corpus_bleu = translator.metric.compute(
                predictions=all_translations,
                references=[[ref] for ref in all_references]
            )
            metrics['bleu'] = corpus_bleu['bleu'] if corpus_bleu else 0.0
        except Exception as e:
            logger.error(f"Error calculating BLEU: {str(e)}")

        # SacreBLEU
        try:
            sacrebleu = evaluate.load("sacrebleu")
            sacrebleu_score = sacrebleu.compute(
                predictions=all_translations,
                references=[[ref] for ref in all_references]
            )
            metrics['sacrebleu'] = sacrebleu_score['score']
        except Exception as e:
            logger.error(f"Error calculating SacreBLEU: {str(e)}")

        # COMET
        try:
            comet = evaluate.load('comet')
            comet_score = comet.compute(
                predictions=all_translations,
                references=all_references,
                sources=source_texts
            )
            metrics['comet'] = np.mean(comet_score['scores'])
        except Exception as e:
            logger.error(f"Error calculating COMET: {str(e)}")

        # METEOR
        try:
            meteor = evaluate.load('meteor')
            meteor_score = meteor.compute(
                predictions=all_translations,
                references=all_references
            )
            metrics['meteor'] = meteor_score['meteor']
        except Exception as e:
            logger.error(f"Error calculating METEOR: {str(e)}")

        # Calculate bootstrap confidence intervals with error handling
        bootstrap_metrics = {
            'bleu': [],
            'sacrebleu': [],
            'comet': [],
            'meteor': []
        }
        
        n_bootstrap = 100
        sample_size = len(all_translations)
        
        if sample_size > 0:
            for _ in range(n_bootstrap):
                indices = np.random.choice(sample_size, size=sample_size, replace=True)
                bootstrap_translations = [all_translations[i] for i in indices]
                bootstrap_references = [all_references[i] for i in indices]
                bootstrap_sources = [source_texts[i] for i in indices]
                
                try:
                    # BLEU bootstrap
                    bleu = translator.metric.compute(
                        predictions=bootstrap_translations,
                        references=[[ref] for ref in bootstrap_references]
                    )
                    if bleu and 'bleu' in bleu:
                        bootstrap_metrics['bleu'].append(bleu['bleu'])
                    
                    # SacreBLEU bootstrap
                    sb_score = sacrebleu.compute(
                        predictions=bootstrap_translations,
                        references=[[ref] for ref in bootstrap_references]
                    )
                    bootstrap_metrics['sacrebleu'].append(sb_score['score'])
                    
                    # COMET bootstrap
                    comet_score = comet.compute(
                        predictions=bootstrap_translations,
                        references=bootstrap_references,
                        sources=bootstrap_sources
                    )
                    bootstrap_metrics['comet'].append(np.mean(comet_score['scores']))
                    
                    # METEOR bootstrap
                    meteor_score = meteor.compute(
                        predictions=bootstrap_translations,
                        references=bootstrap_references
                    )
                    bootstrap_metrics['meteor'].append(meteor_score['meteor'])
                
                except Exception as e:
                    logger.error(f"Error in bootstrap iteration: {str(e)}")
                    continue

        # Calculate confidence intervals with safety checks
        confidence_intervals = {}
        for metric, scores in bootstrap_metrics.items():
            if scores:  # Only calculate if we have scores
                confidence_intervals[metric] = np.percentile(scores, [2.5, 97.5])
            else:
                confidence_intervals[metric] = [0.0, 0.0]  # Default values if no scores
        
        # Store results
        results = []
        for src, ref, trans in zip(source_texts, all_references, all_translations):
            results.append({
                'source': src,
                'reference': ref,
                'translation': trans,
                'bleu': metrics['bleu'],
                'sacrebleu': metrics['sacrebleu'],
                'comet': metrics['comet'],
                'meteor': metrics['meteor']
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'metric': [],
            'value': [],
            'ci_lower': [],
            'ci_upper': []
        }
        
        for metric_name in metrics.keys():
            aggregate_metrics['metric'].append(metric_name)
            aggregate_metrics['value'].append(metrics[metric_name])
            ci = confidence_intervals.get(metric_name, [0.0, 0.0])
            aggregate_metrics['ci_lower'].append(ci[0])
            aggregate_metrics['ci_upper'].append(ci[1])
        
        # Add number of samples
        aggregate_metrics['metric'].append('num_samples')
        aggregate_metrics['value'].append(len(results))
        aggregate_metrics['ci_lower'].append(None)
        aggregate_metrics['ci_upper'].append(None)
        
        # Save results and metrics
        results_path = os.path.join(translator.output_dir, 'sample_translations.csv')
        metrics_path = os.path.join(translator.output_dir, 'sample_metrics.csv')
        
        # Convert results to DataFrame and save as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        logger.info(f"Translation results saved to {results_path}")
        
        # Convert metrics to DataFrame and save as CSV
        metrics_df = pd.DataFrame(aggregate_metrics)
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Log example translations
        if results:
            logger.info("\nExample Translations:")
            logger.info("-" * 50)
            for _, row in results_df.head().iterrows():
                logger.info(f"Source: {row['source']}")
                logger.info(f"Reference: {row['reference']}")
                logger.info(f"Translation: {row['translation']}")
                logger.info(f"BLEU: {row['bleu']:.4f}")
                logger.info(f"SacreBLEU: {row['sacrebleu']:.4f}")
                logger.info(f"COMET: {row['comet']:.4f}")
                logger.info(f"METEOR: {row['meteor']:.4f}")
                logger.info("-" * 50)
        
        # Log aggregate metrics
        logger.info("\nAggregate Metrics:")
        for metric_name in metrics.keys():
            logger.info(f"{metric_name.upper()}: {metrics[metric_name]:.4f}")
            if metric_name in confidence_intervals:
                ci = confidence_intervals[metric_name]
                logger.info(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        logger.info(f"Number of samples: {len(results)}")
        
        return metrics, confidence_intervals, results
        
    except Exception as e:
        logger.error(f"Error during sample translation: {str(e)}", exc_info=True)
        raise

# %%
class mBART_Translator:
    def __init__(
        self,
        model_name: str = "facebook/mbart-large-50",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./mbart-twi-english",
        src_lang: str = "twi_GH",
        tgt_lang: str = "en_XX",
        src_col: str = "TWI", 
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col=src_col 
        self.tgt_col=tgt_col
        self.logger = logger

        # Log initialization parameters
        self.logger.info("=== Translator Configuration ===")
        self.logger.info(f"Model name: {model_name}")
        self.logger.info(f"Max length: {max_length}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Number of epochs: {num_epochs}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Weight decay: {weight_decay}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Source lang: {src_lang}")
        self.logger.info(f"Target lang: {tgt_lang}")
        self.logger.info(f"Source column: {src_col}")
        self.logger.info(f"Target column: {tgt_col}")
        self.logger.info("=" * 50)

        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        if model_name == "facebook/mbart-large-50":
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        else:
            self.logger.info("Specify tokenizer for this model")
            sys.exit("bad tokenizer")

        # Move model to specified device and set up metrics
        self.model = self.model.to(self.device)
        self.metric = evaluate.load("bleu")

    def preprocess_function(self, examples: Dict) -> Dict:
        try:
            # Add language tokens
            self.tokenizer.src_lang = self.src_lang
            self.tokenizer.tgt_lang = self.tgt_lang

            # Get inputs and targets from the correct column names
            inputs = [text for text in examples[self.src_col]]
            targets = [text for text in examples[self.tgt_col]]

            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            model_inputs["labels"] = labels["input_ids"]

            # Move tensors to correct device
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

            return model_inputs
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def compute_metrics(self, eval_preds) -> Dict:
        try:
            preds, labels = eval_preds

            # Decode predictions
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Compute BLEU score
            result = self.metric.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels]
            )

            return {"bleu": result["bleu"]}
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset: DatasetDict):
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )

            # Set up training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir = True,
                evaluation_strategy="epoch",
                do_predict = True,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                num_train_epochs=self.num_epochs,
                predict_with_generate=True,
                logging_dir=f"{self.output_dir}/logs",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_bleu",
                fp16=True,
                fp16_opt_level="O1",
                max_grad_norm=1.0,
                warmup_steps=100,
                logging_steps=10,
                no_cuda=(self.device == 'cpu')
            )

            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    padding=True,
                    return_tensors="pt"),
                compute_metrics=self.compute_metrics
            )

            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")

            # Return training metrics
            self.logger.info("Starting evaluation...")
            return trainer.evaluate()
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, text: str) -> str:
        try:
            # Prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            )

            # Move input tensors to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate translation
            translated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                max_length=self.max_length,
                num_beams=3,
                early_stopping=True
            )

            # Decode and return translation
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Error during translation: {str(e)}")
            raise

# %%
class mT5_Translator:
    def __init__(
        self,
        model_name: str = "google/mt5-small",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./mbart-twi-english",
        device: str = None,
        logger = None,
        src_col: str = "TWI", 
        tgt_col: str = "ENGLISH",
        src_lang: str = "twi",
        tgt_lang: str = "en"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        self.src_col=src_col 
        self.tgt_col=tgt_col
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Initialize tokenizer and model
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)

        # Move model to device
        self.model.to(self.device)

        # Setup metrics
        self.metric = evaluate.load('bleu')

        # Verify model is in training mode
        self.model.train()

        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params}")
        self.logger.info(f"Trainable parameters: {trainable_params}")

    def preprocess_function(self, examples):
        # Add prefix to source texts
        source_texts = [f"Translate {self.src_col} to {self.tgt_col}: " + text for text in examples[self.src_col]]
        target_texts = examples[self.tgt_col]

        # Tokenize inputs
        model_inputs = self.tokenizer(
            source_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        model_inputs["labels"] = labels["input_ids"]

        # Verify tensor shapes and values
        self.logger.debug(f"Input shape: {model_inputs['input_ids'].shape}")
        self.logger.debug(f"Label shape: {model_inputs['labels'].shape}")

        # Convert tensors to lists for dataset storage
        return {
            "input_ids": model_inputs["input_ids"].tolist(),
            "attention_mask": model_inputs["attention_mask"].tolist(),
            "labels": model_inputs["labels"].tolist()
        }

    def compute_metrics(self, eval_preds) -> Dict:
        try:
            preds, labels = eval_preds

            # Decode predictions
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Compute BLEU score
            result = self.metric.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels]
            )

            return {"bleu": result["bleu"]}
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset_dict: DatasetDict):
        self.logger.info("Starting preprocessing...")

        # Preprocess datasets
        processed_train = dataset_dict["train"].map(
            self.preprocess_function,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=dataset_dict["train"].column_names,
            desc="Preprocessing training dataset"
        )

        processed_test = dataset_dict["test"].map(
            self.preprocess_function,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=dataset_dict["test"].column_names,
            desc="Preprocessing test dataset"
        )

        # Verify processed datasets
        self.logger.info(f"Processed train features: {processed_train.features}")
        self.logger.info(f"Sample processed input: {processed_train[0]}")

        # Define training arguments with careful parameter tuning
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu",
            predict_with_generate=False,
            generation_max_length=self.max_length,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=100,
            fp16=True,
            fp16_opt_level="O1"
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_test,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding='max_length',
                max_length=self.max_length
            ),
            compute_metrics=self.compute_metrics
        )

        # Verify model is on correct device
        self.logger.info(f"Model device: {next(self.model.parameters()).device}")

        # Train the model with exception handling
        try:
            self.logger.info("Starting training...")
            trainer.train()
            trainer.save_model()
            self.logger.info(f"Model saved to {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise

    def translate(self, texts):
        source_texts = [f"Translate {self.src_col} to {self.tgt_col}: " + text for text in texts]
        inputs = self.tokenizer(
            source_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate translations
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=3,
                temperature=0.7,
                length_penalty=1.0,
                early_stopping=True
            )

        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations = [t.strip() for t in translations]

        return translations[0]

# https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865
class NLLB_Translator:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-3.3B",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./nllb-translation",
        src_lang: str = "twi_Latn",
        tgt_lang: str = "eng_Latn",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Log initialization
        self._log_init_params()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self.metric = evaluate.load("bleu")
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Using {torch.cuda.device_count()} GPUs")

    def _log_init_params(self):
        self.logger.info("=== NLLB Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 50)

    def preprocess_function(self, examples: Dict) -> Dict:
        try:
            inputs = examples[self.src_col]
            targets = examples[self.tgt_col]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                # src_lang=self.src_lang
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                # tgt_lang=self.tgt_lang
            )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # Move tensors to device
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            
            return model_inputs
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def compute_metrics(self, eval_preds) -> Dict:
        try:
            preds, labels = eval_preds
            
            # Decode predictions
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Compute BLEU score
            result = self.metric.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels]
            )
            
            return {"bleu": result["bleu"]}
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            # Set up training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                evaluation_strategy="epoch",
                do_predict=True,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=max(1, self.batch_size // 4),
                weight_decay=self.weight_decay,
                num_train_epochs=self.num_epochs,
                logging_dir=f"{self.output_dir}/logs",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_bleu",
                fp16=True,
                fp16_opt_level="O1",
                max_grad_norm=1.0,
                warmup_steps=100,
                logging_steps=10,
                no_cuda=(self.device == 'cpu'),
                generation_max_length=self.max_length,
                generation_num_beams=4
            )
            
            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    padding=True,
                    return_tensors="pt"
                ),
                compute_metrics=self.compute_metrics
            )
            
            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            # Return evaluation metrics
            self.logger.info("Starting evaluation...")
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, text: str) -> str:
        try:
            # Prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True,
                # src_lang=self.src_lang
            )
            
            # Move input tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            translated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
                max_length=self.max_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
            
            # Decode and return translation
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"Error during translation: {str(e)}")
            raise

class M2M_Translator:
    def __init__(
        self,
        model_name: str = "facebook/m2m100_1.2B",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./m2m-translation",
        src_lang: str = "twi",
        tgt_lang: str = "eng",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self.metric = evaluate.load("bleu")
        
        # Log initialization
        self._log_init_params()

    def _log_init_params(self):
        self.logger.info("=== M2M Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 50)

    def preprocess_function(self, examples: Dict) -> Dict:
        try:
            # Set source language for tokenization
            # self.tokenizer.src_lang = self.src_lang
            
            inputs = examples[self.src_col]
            targets = examples[self.tgt_col]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # Move tensors to device
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            
            return model_inputs
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def compute_metrics(self, eval_preds) -> Dict:
        try:
            preds, labels = eval_preds
            
            # Decode predictions
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Compute BLEU score
            result = self.metric.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels]
            )
            return {"bleu": result["bleu"]}
            
        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            # Set up training arguments with gradient accumulation
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                evaluation_strategy="epoch",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                gradient_accumulation_steps=4, 
                weight_decay=self.weight_decay,
                num_train_epochs=self.num_epochs,
                predict_with_generate=True,
                fp16=True,
                fp16_opt_level="O1",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_bleu",
                greater_is_better=True,
                warmup_steps=100,
                logging_steps=10
            )
            
            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    padding=True,
                    return_tensors="pt"
                ),
                compute_metrics=self.compute_metrics
            )
            
            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, text: str) -> str:
        try:
            # Prepare input
            self.tokenizer.src_lang = self.src_lang
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate translation
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                max_length=self.max_length,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            # Decode and return translation
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
        except Exception as e:
            self.logger.error(f"Error during translation: {str(e)}")
            raise

class Llama_Translator:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./llama-translation",
        src_lang: str = "twi",
        tgt_lang: str = "eng",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self.metric = evaluate.load("bleu")
        
        # Log initialization
        self._log_init_params()

    def _log_init_params(self):
        self.logger.info("=== Llama Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 50)
    
    def preprocess_function(self,examples):
        inputs = [f"<s>You are an accurate, precise, and honest multilingul large language model that translates {self.src_lang}. Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence : {sentence}</s>" for sentence in examples[self.src_lang]]
        targets = [f"<s>{translation}</s>" for translation in examples[self.tgt_lang]]

        # Tokenize the input text (instruction + English sentence)
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # Tokenize the target text (TWI translation)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=512,
                padding="max_length",
                truncation=True,
            )

        # Assign labels and mask padding tokens with -100
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]

        return model_inputs


    def compute_metrics(self, eval_preds) -> Dict:
        try:
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids
            if len(predictions.shape) == 3:
                predictions = predictions.argmax(axis=-1)

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [
                [(token if token != -100 else self.tokenizer.pad_token_id) for token in label]
                for label in labels
            ]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # log the first few examples
            for i in range(3):
                self.logger.info(f"Prediction: {decoded_preds[i]}")
                self.logger.info(f"Reference: {decoded_labels[i]}")
                self.logger.info("=" * 50)

            references = [[label] for label in decoded_labels]

            bleu_metric = evaluate.load("bleu")
            result = bleu_metric.compute(predictions=decoded_preds, references=references)

            bleu_score = result["bleu"]

            self.logger.info(f"BLEU score: {bleu_score:.4f}")

            return {"bleu": bleu_score}

        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            # Set up training arguments with gradient accumulation

            training_args = TrainingArguments(
                        output_dir=self.output_dir,
                        evaluation_strategy="epoch",
                        logging_dir="./logs",
                        save_strategy="epoch",
                        learning_rate=self.learning_rate,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        num_train_epochs=3,
                        warmup_steps=500,
                        weight_decay=0.01,
                        fp16=True,  # Mixed precision training for efficiency
                        save_total_limit=2,
                    )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, sentence, max_length=128):
        prompt = f"<s>Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence: {sentence}.\nTranslation:"
        print("sentence : ", sentence)
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the translation
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,              
            early_stopping=True,      
            no_repeat_ngram_size=3,   
            repetition_penalty=2.0,    
            temperature=1.0,           
            top_k=50,                 
            top_p=0.95,                
        )
        raw_translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("raw translation : ", raw_translation)

        cleaned_translation = raw_translation.split("Translation:")[-1].strip()

        return cleaned_translation

class OPT_Translator:
    def __init__(
        self,
        model_name: str ="facebook/opt-350m",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./opt-translation",
        src_lang: str = "TWI",
        tgt_lang: str = "ENGLISH",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self.metric = evaluate.load("bleu")
        
        # Log initialization
        self._log_init_params()

    def _log_init_params(self):
        self.logger.info("=== OPT Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 50)
    
    def preprocess_function(self,examples):
        inputs = [f"<s>You are an accurate, efficient, and honest multilingul LLM that specializes in translating from {self.src_lang}.  Instruction: Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence: {sentence}</s>" for sentence in examples[self.src_lang]]
        targets = [f"<s>{translation}</s>" for translation in examples[self.tgt_lang]]

        # Tokenize the input text (instruction + English sentence)
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # Tokenize the target text (TWI translation)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=512,
                padding="max_length",
                truncation=True,
            )

        # Assign labels and mask padding tokens with -100
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]

        return model_inputs


    def compute_metrics(self, eval_preds) -> Dict:
        try:
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids
            if len(predictions.shape) == 3:  # Shape: (batch_size, sequence_length, vocab_size)
                predictions = predictions.argmax(axis=-1)  # Take the token with the highest probability

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [
                [(token if token != -100 else self.tokenizer.pad_token_id) for token in label]
                for label in labels
            ]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            references = [[label] for label in decoded_labels]

            bleu_metric = evaluate.load("bleu")
            result = bleu_metric.compute(predictions=decoded_preds, references=references)

            bleu_score = result["bleu"]

            self.logger.info(f"BLEU score: {bleu_score:.4f}")

            return {"bleu": bleu_score}

        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            # Set up training arguments with gradient accumulation

            training_args = TrainingArguments(
                        output_dir=self.output_dir,
                        evaluation_strategy="epoch",
                        logging_dir="./logs",
                        save_strategy="epoch",
                        learning_rate=self.learning_rate,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        num_train_epochs=3,
                        warmup_steps=500,
                        weight_decay=0.01,
                        fp16=True,  # Mixed precision training for efficiency
                        save_total_limit=2,
                    )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, sentence, max_length=128):
        prompt = f"<s>Instruction: Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence: {sentence}.\nTranslation:"
        print("sentence : ", sentence)
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the translation
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,              
            early_stopping=True,      
            no_repeat_ngram_size=3,   
            repetition_penalty=2.0,    
            temperature=1.0,           
            top_k=50,                 
            top_p=0.95,                
        )
        raw_translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("raw translation : ", raw_translation)

        cleaned_translation = raw_translation.split("Translation:")[-1].strip()

        return cleaned_translation

class Mistral_Translator:
    def __init__(
        self,
        model_name: str ="mistralai/Mistral-7B-Instruct-v0.1",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./mistral-translation",
        src_lang: str = "TWI",
        tgt_lang: str = "ENGLISH",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model

        login(token="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")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as PAD token
        self.tokenizer.padding_side = "left"  # Important for XGLM
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self.metric = evaluate.load("bleu")
        
        # Log initialization
        self._log_init_params()

    def _log_init_params(self):
        self.logger.info("=== Mistral Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 50)
    
    def preprocess_function(self,examples):
        inputs = [f"<s>You are an accurate, precise, and honest multilingul large language model that translates {self.src_lang}. Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence : {sentence}</s>" for sentence in examples[self.src_lang]]
        targets = [f"<s>{translation}</s>" for translation in examples[self.tgt_lang]]

        # Tokenize the input text (instruction + English sentence)
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # Tokenize the target text (TWI translation)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=512,
                padding="max_length",
                truncation=True,
            )

        # Assign labels and mask padding tokens with -100
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]

        return model_inputs


    def compute_metrics(self, eval_preds) -> Dict:
        try:
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids
            if len(predictions.shape) == 3:  # Shape: (batch_size, sequence_length, vocab_size)
                predictions = predictions.argmax(axis=-1)  # Take the token with the highest probability

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [
                [(token if token != -100 else self.tokenizer.pad_token_id) for token in label]
                for label in labels
            ]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            references = [[label] for label in decoded_labels]

            bleu_metric = evaluate.load("bleu")
            result = bleu_metric.compute(predictions=decoded_preds, references=references)

            bleu_score = result["bleu"]

            self.logger.info(f"BLEU score: {bleu_score:.4f}")

            return {"bleu": bleu_score}

        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            # Set up training arguments with gradient accumulation

            training_args = TrainingArguments(
                        output_dir=self.output_dir,
                        evaluation_strategy="epoch",
                        logging_dir="./logs",
                        save_strategy="epoch",
                        learning_rate=self.learning_rate,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        num_train_epochs=3,
                        warmup_steps=500,
                        weight_decay=0.01,
                        fp16=True,  # Mixed precision training for efficiency
                        save_total_limit=2,
                    )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, sentence, max_length=128):
        prompt = f"<s>Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence: {sentence}.\nTranslation:"
        print("sentence : ", sentence)
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the translation
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,              
            early_stopping=True,      
            no_repeat_ngram_size=3,   
            repetition_penalty=2.0,    
            temperature=1.0,           
            top_k=50,                 
            top_p=0.95,                
        )
        raw_translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("raw translation : ", raw_translation)

        cleaned_translation = raw_translation.split("Translation:")[-1].strip()

        return cleaned_translation
        
class Aya_Translator:
    def __init__(
        self,
        model_name: str ="CohereForAI/aya-23-8B",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./aya-translation",
        src_lang: str = "TWI",
        tgt_lang: str = "ENGLISH",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device and check for multiple GPUs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_gpu = torch.cuda.device_count() > 1
        
        # Initialize tokenizer and model
        login(token="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")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as PAD token
        self.tokenizer.padding_side = "left"  # Important for XGLM
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Log initialization
        self._log_init_params()
        
        # Move model to device and wrap with DataParallel if multiple GPUs are available
        self.logger.info("Moving model to device...")
        self.model = self.model.to(self.device)
        if self.multi_gpu:
            self.logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        self.logger.info("Model successfully moved to device.")
        
        # Setup metrics
        self.metric = evaluate.load("bleu")

    def _log_init_params(self):
        self.logger.info("=== Aya Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 50)
    
    def preprocess_function(self,examples):
        inputs = [f"<s>You are an accurate, precise, and honest multilingul large language model that translates {self.src_lang}. Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence : {sentence}</s>" for sentence in examples[self.src_lang]]
        targets = [f"<s>{translation}</s>" for translation in examples[self.tgt_lang]]

        # Tokenize the input text (instruction + English sentence)
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # Tokenize the target text (TWI translation)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=512,
                padding="max_length",
                truncation=True,
            )

        # Assign labels and mask padding tokens with -100
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]

        return model_inputs

    def compute_metrics(self, eval_preds) -> Dict:
        try:
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids
            if len(predictions.shape) == 3:  # Shape: (batch_size, sequence_length, vocab_size)
                predictions = predictions.argmax(axis=-1)  # Take the token with the highest probability

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [
                [(token if token != -100 else self.tokenizer.pad_token_id) for token in label]
                for label in labels
            ]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            references = [[label] for label in decoded_labels]

            bleu_metric = evaluate.load("bleu")
            result = bleu_metric.compute(predictions=decoded_preds, references=references)

            bleu_score = result["bleu"]

            self.logger.info(f"BLEU score: {bleu_score:.4f}")

            return {"bleu": bleu_score}

        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
               
            # Set up training arguments with adjusted batch size
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                evaluation_strategy="epoch",
                logging_dir="./logs",
                save_strategy="epoch",
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=3,
                warmup_steps=500,
                weight_decay=0.01,
                fp16=True, 
                save_total_limit=2,
                remove_unused_columns=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save the model (handle DataParallel wrapper)
            model_to_save = self.model.module if self.multi_gpu else self.model
            model_to_save.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, sentence, max_length=128):
        # Get the actual model (handle DataParallel wrapper)
        model = self.model.module if self.multi_gpu else self.model
        
        prompt = f"<s>Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence: {sentence}.\nTranslation:"
        print("sentence : ", sentence)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the translation
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,              
            early_stopping=True,      
            no_repeat_ngram_size=3,   
            repetition_penalty=2.0,    
            temperature=1.0,           
            top_k=50,                 
            top_p=0.95,                
        )
        raw_translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("raw translation : ", raw_translation)

        cleaned_translation = raw_translation.split("Translation:")[-1].strip()
        return cleaned_translation 

class XGLM_Translator:
    def __init__(
        self,
        model_name: str ="facebook/xglm-564M",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./xglm-translation",
        src_lang: str = "TWI",
        tgt_lang: str = "ENGLISH",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token as PAD token
        self.tokenizer.padding_side = "left"  # Important for XGLM
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self.metric = evaluate.load("bleu")
        
        # Log initialization
        self._log_init_params()

    def _log_init_params(self):
        self.logger.info("=== XGLM Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 50)
    
    def preprocess_function(self,examples):
        inputs = [f"<s>You are an accurate, precise, and honest multilingul large language model that translates {self.src_lang}. Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence : {sentence}</s>" for sentence in examples[self.src_lang]]
        targets = [f"<s>{translation}</s>" for translation in examples[self.tgt_lang]]

        # Tokenize the input text (instruction + English sentence)
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # Tokenize the target text (TWI translation)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=512,
                padding="max_length",
                truncation=True,
            )

        # Assign labels and mask padding tokens with -100
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]

        return model_inputs


    def compute_metrics(self, eval_preds) -> Dict:
        try:
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids
            if len(predictions.shape) == 3:  # Shape: (batch_size, sequence_length, vocab_size)
                predictions = predictions.argmax(axis=-1)  # Take the token with the highest probability

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [
                [(token if token != -100 else self.tokenizer.pad_token_id) for token in label]
                for label in labels
            ]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            references = [[label] for label in decoded_labels]

            bleu_metric = evaluate.load("bleu")
            result = bleu_metric.compute(predictions=decoded_preds, references=references)

            bleu_score = result["bleu"]

            self.logger.info(f"BLEU score: {bleu_score:.4f}")

            return {"bleu": bleu_score}

        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            # Set up training arguments with gradient accumulation

            training_args = TrainingArguments(
                        output_dir=self.output_dir,
                        evaluation_strategy="epoch",
                        logging_dir="./logs",
                        save_strategy="epoch",
                        learning_rate=self.learning_rate,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        num_train_epochs=3,
                        warmup_steps=500,
                        weight_decay=0.01,
                        fp16=True,  # Mixed precision training for efficiency
                        save_total_limit=2,
                    )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, sentence, max_length=128):
        prompt = f"<s>Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence: {sentence}.\nTranslation:"
        print("sentence : ", sentence)
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate the translation
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,              
            early_stopping=True,      
            no_repeat_ngram_size=3,   
            repetition_penalty=2.0,    
            temperature=1.0,           
            top_k=50,                 
            top_p=0.95,                
        )
        raw_translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("raw translation : ", raw_translation)

        cleaned_translation = raw_translation.split("Translation:")[-1].strip()

        return cleaned_translation
        
class Falcon_Translator:
    def __init__(
        self,
        model_name: str = "tiiuae/falcon-7b",
        max_length: int = 128,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "./falcon-translation",
        src_lang: str = "twi",
        tgt_lang: str = "eng",
        src_col: str = "TWI",
        tgt_col: str = "ENGLISH",
        device: str = None,
        logger = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.logger = logger
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self.metric = evaluate.load("bleu")
        
        # Log initialization
        self._log_init_params()

    def _log_init_params(self):
        self.logger.info("=== Falcon Translator Configuration ===")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Max length: {self.max_length}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Source language: {self.src_lang}")
        self.logger.info(f"Target language: {self.tgt_lang}")
        self.logger.info(f"Device: {self.device}")
        
        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 50)

    def preprocess_function(self,examples):
        inputs = [f"<s>You are an accurate, precise, and honest multilingul large language model that translates {self.src_lang}. Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence : {sentence}</s>" for sentence in examples[self.src_lang]]
        targets = [f"<s>{translation}</s>" for translation in examples[self.tgt_lang]]

        # Tokenize the input text (instruction + English sentence)
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        # Tokenize the target text (TWI translation)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=512,
                padding="max_length",
                truncation=True,
            )

        # Assign labels and mask padding tokens with -100
        model_inputs["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in labels["input_ids"]
        ]

        return model_inputs

    def compute_metrics(self, eval_preds) -> Dict:
        try:
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids
            if len(predictions.shape) == 3:  # Shape: (batch_size, sequence_length, vocab_size)
                predictions = predictions.argmax(axis=-1)  # Take the token with the highest probability

            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

            labels = [
                [(token if token != -100 else self.tokenizer.pad_token_id) for token in label]
                for label in labels
            ]
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            references = [[label] for label in decoded_labels]

            bleu_metric = evaluate.load("bleu")
            result = bleu_metric.compute(predictions=decoded_preds, references=references)

            bleu_score = result["bleu"]

            self.logger.info(f"BLEU score: {bleu_score:.4f}")

            return {"bleu": bleu_score}

        except Exception as e:
            self.logger.error(f"Error computing metrics: {str(e)}")
            return {"bleu": 0.0}

    def train(self, dataset):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            # Process datasets
            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            # Set up training arguments with gradient accumulation
            training_args = TrainingArguments(
                                output_dir=self.output_dir,
                                overwrite_output_dir=True,
                                evaluation_strategy="epoch",
                                learning_rate=self.learning_rate,
                                per_device_train_batch_size=self.batch_size,
                                per_device_eval_batch_size=self.batch_size,
                                gradient_accumulation_steps=4,
                                weight_decay=self.weight_decay,
                                num_train_epochs=self.num_epochs,
                                # predict_with_generate=True,
                                fp16=True,
                                save_strategy="epoch",
                                load_best_model_at_end=True,
                                metric_for_best_model="eval_bleu",
                                greater_is_better=True,
                                warmup_steps=100,
                                logging_steps=10
                            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_datasets["train"],
                eval_dataset=processed_datasets["test"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )
            
            # Train the model
            self.logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            self.logger.info(f"Model saved to {self.output_dir}")
            
            return trainer.evaluate()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def translate(self, sentence, max_length=128):
        prompt = f"<s>Instruction: Translate the following {self.src_lang} sentence to {self.tgt_lang}.\nSentence: {sentence}</s>"

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=5,  # Use beam search for better translations
            early_stopping=True,
            no_repeat_ngram_size=2,  # Avoid repeating phrases
        )

        # Decode the output tokens to text
        raw_translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove instructions from the output to clean the translation
        cleaned_translation = raw_translation.split("Sentence:")[-1].strip()

        return cleaned_translation
