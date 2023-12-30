#/usr/bin/env bash

cd ../sft/

deepspeed main.py \
	--data_path ../kse_sample_dataset/ \
	--model_name_or_path 01-ai/Yi-6B \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--max_seq_len 4096 \
	--learning_rate 2e-6 \
	--weight_decay 0. \
	--num_train_epochs 3 \
	--training_debug_steps 20 \
	--gradient_accumulation_steps 4 \
	--lr_scheduler_type cosine \
	--num_warmup_steps 0 \
	--seed 1234 \
	--gradient_checkpointing \
	--zero_stage 2 \
	--deepspeed \
	--offload \
	--lora_dim 128 \
	--lora_module_name "layers." \
	--output_dir ./output_Yi_6b_sft_lora #\
#	--output_dir ./output_Yi_6b_chat_sft_lora #\
#	--bf16 True \
#	--fp16 False \
#	--torch_dtype torch.bfloat16
