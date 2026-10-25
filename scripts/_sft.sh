# Stand-in SFT (text -> BRep sequence) with verl's FSDP SFT trainer. Called by scripts/run02_sft_*.sbatch
# after scripts/_common.sh. Reads: BASE_MODEL, SFT_EXP, EPOCHS (2), LR (1e-5), TRAIN (sft_train.parquet).
# Extra Hydra overrides can be appended as arguments.
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}
TRAIN=${TRAIN:-$DATA/sft_train.parquet}
echo "[sft] model=$BASE_MODEL exp=$SFT_EXP epochs=$EPOCHS lr=$LR gpus=$NGPUS train=$TRAIN"

torchrun --standalone --nnodes=1 --nproc-per-node="$NGPUS" -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN" \
  data.val_files="$DATA/sft_val.parquet" \
  data.messages_key=messages \
  data.max_length=16384 \
  data.truncation=right \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu=32768 \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=1 \
  model.path="$BASE_MODEL" \
  model.use_remove_padding="$REMOVE_PADDING" \
  +model.override_config.attn_implementation="$ATTN_IMPL" \
  model.enable_gradient_checkpointing=True \
  engine=fsdp \
  optim=fsdp \
  engine.strategy=fsdp2 \
  optim.lr="$LR" \
  optim.lr_warmup_steps_ratio=0.03 \
  optim.weight_decay=0.0 \
  optim.lr_scheduler_type=cosine \
  optim.min_lr_ratio=0.1 \
  optim.clip_grad=1.0 \
  trainer.total_epochs="$EPOCHS" \
  trainer.save_freq=after_each_epoch \
  trainer.test_freq=100 \
  trainer.logger="$LOGGER" \
  trainer.project_name=rltuned-cad \
  trainer.experiment_name="$SFT_EXP" \
  trainer.default_local_dir="$CKPT/$SFT_EXP" \
  trainer.n_gpus_per_node="$NGPUS" \
  trainer.nnodes=1 \
  trainer.resume_mode=auto \
  trainer.max_ckpt_to_keep=2 \
  checkpoint.save_contents=[model,optimizer,extra,hf_model] \
  "$@"

echo "[sft] done. HF export: $(latest_hf "$CKPT/$SFT_EXP")"
