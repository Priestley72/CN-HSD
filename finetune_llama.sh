gpus=0
epoch=8
batch_size=2
lr=5e-5
# train_dataset="format_train"
train_dataset="format_enhanced_train"
valid_dataset="format_positive_test1"
model_path="./models/Qwen3-8B"
export_name="Qwen3-8B-llama-epoch${epoch}"
model_save_path="./models/${export_name}"
# export WANDB_MODE=offline
export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICES=${gpus}
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path ${model_path} \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --dataset_dir "./data" \
    --dataset ${train_dataset} \
    --template empty \
    --output_dir ${model_save_path} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps $((16 / batch_size)) \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps $((4000 / batch_size / epoch)) \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --plot_loss \
    --bf16 \
    --warmup_ratio 0.05 \
    --save_total_limit 5 \
    --report_to "tensorboard" \
#    --evaluation_strategy "steps" \
#    --eval_dataset ${valid_dataset} \
#    --metric_for_best_model "loss" \
#    --eval_steps $((4000 / batch_size / epoch / 2)) \
#    --load_best_model_at_end True \
#    --temperature 0.3 \
#    --repetition_penalty 1.1
#    --top_k 50 \
#    --top_p 0.9 \
#    --load_best_model_at_end \
