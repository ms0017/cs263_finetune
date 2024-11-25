#!pip install datasets evaluate sacrebleu transformers==4.45.2


import warnings
import torch, gc
import os
from datetime import datetime
from functions import initialize_logs, import_data, mT5_Translator, evaluate_model
warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    try: 
        src_col="TWI"
        tgt_col="ENGLISH"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"mbart_{src_col}_{tgt_col}_{timestamp}"
        logger = initialize_logs("logs", output_dir_name)
        dataset = import_data('IssakaAI/en-tw', subset=-1)
        
        # Initialize and train the translator
        translator = mT5_Translator(
        model_name="google/mt5-small",
        max_length= 128,
        batch_size= 16,
        num_epochs= 3,
        learning_rate= 1e-5,
        weight_decay= 0.01,
        output_dir=f"./{output_dir_name}",
        device= None,
        logger = logger,
        src_col=src_col, 
        tgt_col=tgt_col
        )
        
        # Train
        logger.info("Beginning model training procedures")
        metrics = translator.train(dataset)
        logger.info(f"After training metrics {metrics}")

        # Evaluate
        logger.info(f"Beginning sampled model evaluation")
        evaluate_model(translator, dataset, src_col, tgt_col, logger, subset=500)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise