from datetime import datetime
import warnings
import torch
import os
import gc
from functions import initialize_logs, import_data, evaluate_model, Llama_Translator

warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        src_col = "TWI"
        tgt_col = "ENGLISH"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"nllb_{src_col}_{tgt_col}_{timestamp}"
        logger = initialize_logs("logs", output_dir_name)
        dataset = import_data('IssakaAI/en-tw', subset=-1)
        
        # Initialize and train the translator
        translator = Llama_Translator(
            model_name="meta-llama/Llama-3.2-3B",
            max_length=128,
            batch_size=16,
            num_epochs=1,
            learning_rate=1e-5,
            weight_decay=0.01,
            output_dir=f"./{output_dir_name}",
            src_lang="TWI",
            tgt_lang="ENGLISH",
            src_col=src_col,
            tgt_col=tgt_col,
            device=None,
            logger=logger
        )
        
        # Train
        logger.info("Beginning model training procedures")
        metrics = translator.train(dataset)
        logger.info(f"After training metrics {metrics}")
        
        # Evaluate
        logger.info("Beginning sampled model evaluation")
        evaluate_model(translator, dataset, src_col, tgt_col, logger, subset=1000)

        # Save training completion status
        with open(os.path.join(translator.output_dir, "training_completed.txt"), "w") as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Final metrics: {metrics}\n")
        
        logger.info("Translation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise