# python=3.12.4
#!pip install datasets==3.0.1 evaluate==0.4.3 sacrebleu==2.4.3 transformers==4.45.2 numpy==1.26.4 torch==2.5.0

import warnings
import torch, gc
from datetime import datetime
from functions import initialize_logs, import_data, mBART_Translator, evaluate_model
warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()


if __name__ == "__main__":
    try: 
        src_col="TWI"
        tgt_col="ENGLISH"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"mbart_{src_col}_{tgt_col}_{timestamp}"
        logger = initialize_logs("logs", output_dir_name)
        dataset = import_data('IssakaAI/en-tw', subset=-1)
        
        # Initialize and train the translator
        translator = mBART_Translator(
            model_name="facebook/mbart-large-50",
            batch_size=32,
            num_epochs=5,
            learning_rate=1e-5,
            output_dir=f"./{output_dir_name}",
            src_lang="twi_GH",
            tgt_lang="en_XX",
            src_col=src_col,
            tgt_col=tgt_col,
            logger=logger)
        
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