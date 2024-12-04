import warnings
import torch
import gc
import os
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import login
from functions import (initialize_logs, import_data, evaluate_model)
from model_configs import MODEL_CONFIGS

warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Centralized configuration
def create_translator(base_config, output_dir, logger, src_col, tgt_col):
    """Create a translator instance with the specified configuration."""
    config = base_config["params"].copy()
    translator_class = base_config["class"]
    
    if config.get("requires_login"):
        login(os.environ.get("llama_token"))
        config.pop("requires_login")
    
    if config.get("requires_gpu_config"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config.pop("requires_gpu_config")
    
    # Common parameters for all translators
    common_params = {
        "output_dir": output_dir,
        "device": None,
        "logger": logger,
        "src_col": src_col,
        "tgt_col": tgt_col
    }
    
    return translator_class(**{**config, **common_params})

def run_translation_pipeline(model_name, translator, dataset, logger):
    """Execute the translation pipeline for a given model."""
    try:
        # Training phase
        logger.info("Beginning model training procedures...")
        metrics = translator.train(dataset)
        logger.info(f"Training completed. Final metrics: {metrics}")
        
        # Evaluation phase
        logger.info("Beginning model evaluation...")
        evaluate_model(
            translator=translator,
            dataset=dataset,
            src_col=translator.src_col,
            tgt_col=translator.tgt_col,
            logger=logger,
            subset=500
        )
        
        # Save training completion status
        with open(os.path.join(translator.output_dir, "training_completed.txt"), "w") as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Final metrics: {metrics}\n")
        
        logger.info("Translation pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in {model_name} pipeline: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    src_col = "ENGLISH"
    tgt_col = "TWI"
    dataset = import_data('IssakaAI/en-tw', subset=2000)
    
    for model_name in tqdm(MODEL_CONFIGS):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"{model_name}_{src_col}_{tgt_col}_{timestamp}"
        logger = initialize_logs("logs", output_dir_name)
        
        try:
            logger.info(f"Initializing {model_name} translator...")
            translator = create_translator(
                MODEL_CONFIGS[model_name],
                f"./{output_dir_name}",
                logger,
                src_col,
                tgt_col
            )
            
            run_translation_pipeline(model_name, translator, dataset, logger)
            
        except Exception as e:
            logger.error(f"Error in {model_name} initialization: {str(e)}", exc_info=True)
            continue

# torchrun --nproc_per_node=4 unified_model_trainer.py