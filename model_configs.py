from functions import (
    Aya_Translator, Falcon_Translator, Llama_Translator, M2M_Translator,
    mBART_Translator, Mistral_Translator, mT5_Translator, NLLB_Translator,
    OPT_Translator, XGLM_Translator
)

# Shared parameters
SHARED_PARAMS = {
    "max_length": 128,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "batch_size": 64,
    "num_epochs": 1,
    "src_lang": "TWI",
    "tgt_lang": "ENGLISH",
}

MODEL_CONFIGS = {
    # "aya": {
    #     "class": Aya_Translator,
    #     "params": {
    #         "model_name": "CohereForAI/aya-23-8B",
    #         "batch_size": 1
    #     }
    # }, OOM
    "llama": {
        "class": Llama_Translator,
        "params": {
            "model_name": "meta-llama/Llama-3.2-1B",
            "requires_login": True
        }
    },
    # "m2m": {
    #     "class": M2M_Translator,
    #     "params": {
    #         "model_name": "facebook/m2m100_1.2B",
    #         "src_lang": "fr",
    #         "tgt_lang": "en"
    #     }
    # },
    # "mbart": {
    #     "class": mBART_Translator,
    #     "params": {
    #         "model_name": "facebook/mbart-large-50",
    #         "src_lang": "twi_GH",
    #         "tgt_lang": "en_XX"
    #     }
    # },
    # # "mistral": {
    # #     "class": Mistral_Translator,
    # #     "params": {
    # #         "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
    # #         "batch_size": 1
    # #     }
    # # },
    # "mt5": {
    #     "class": mT5_Translator,
    #     "params": {
    #         "model_name": "google/mt5-small",
    #         "requires_gpu_config": True
    #     }
    # },
    # "nllb": {
    #     "class": NLLB_Translator,
    #     "params": {
    #         "model_name": "facebook/nllb-200-3.3B",
    #         "src_lang": "twi_Latn",
    #         "tgt_lang": "eng_Latn"
    #     }
    # },
    # "opt": {
    #     "class": OPT_Translator,
    #     "params": {
    #         "model_name": "facebook/opt-350m"
    #     }
    # },
    # "xglm": {
    #     "class": XGLM_Translator,
    #     "params": {
    #         "model_name": "facebook/xglm-564M",
    #     }
    # },
    # # "falcon": {
    # #     "class": Falcon_Translator,
    # #     "params": {
    # #         "model_name": "tiiuae/falcon-7b",
    # #         "requires_login": True
    # #     }
    # # }
}

for model_config in MODEL_CONFIGS.values():
    model_config["params"] = {**SHARED_PARAMS, **model_config["params"]}
