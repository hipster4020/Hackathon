import os
import warnings

import hydra
import torch
import wandb
from dataloader import load
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

warnings.filterwarnings(action="ignore")


@hydra.main(config_path="./", config_name="config")
def main(cfg):
    torch.manual_seed(42)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        cfg.PATH.model_name,
        bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>',
    )
    # 

    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    model = GPT2LMHeadModel.from_pretrained(cfg.PATH.model_name)
    # model = AutoModelForCausalLM.from_pretrained(**cfg.PATH.model_config)

    if cfg.ETC.get("wandb_project") and os.environ.get("LOCAL_RANK", 0) == 0:
        os.environ["WANDB_PROJECT"] = cfg.ETC.wandb_project
        wandb.init(
            project=cfg.ETC.wandb_project,
            entity=cfg.ETC.wandb_entity,
            name=cfg.TRAININGARGS.run_name,
        )

    args = TrainingArguments(
        do_train=True,
        do_eval=True if eval_dataset is not None else None,
        logging_dir=cfg.PATH.logging_dir,
        output_dir=cfg.PATH.checkpoint_dir,
        **cfg.TRAININGARGS,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    trainer.save_model(cfg.PATH.output_dir)


if __name__ == "__main__":
    main()