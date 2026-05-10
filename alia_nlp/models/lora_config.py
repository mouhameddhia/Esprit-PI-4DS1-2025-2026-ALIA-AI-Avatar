"""QLoRA fine-tuning hyperparameters."""

LORA_R:     int   = 16
LORA_ALPHA: int   = 32
LORA_DROPOUT: float = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

TRAINING_ARGS = {
    "num_train_epochs":    3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate":       2e-4,
    "warmup_ratio":        0.03,
    "lr_scheduler_type":   "cosine",
    "fp16":                True,
    "logging_steps":       10,
    "save_strategy":       "epoch",
    "evaluation_strategy": "epoch",
}
