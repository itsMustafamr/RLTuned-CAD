# GRPO on verl v0.9.1 with our BRep reward (brepgrpo/verl_reward.py). Called by scripts/run03/run04 after
# scripts/_common.sh. Reads:
#   POLICY   HF checkpoint to start from (default: newest export of $SFT_EXP; Bhavika's checkpoint later)
#   STAGE    8k (simple + short complex shapes, 8k-token answers, curriculum order) | 16k (everything)
#   SMOKE    1 -> 6 steps, 2k-token answers, batch 8: checks the plumbing only
#   EXP      run name under $CKPT and runs/
#   N BATCH LR KL TP GPU_UTIL  (defaults 8, 64, 1e-6, 0.001, 1, 0.6)
# Extra Hydra overrides can be appended as arguments.
STAGE=${STAGE:-8k}
POLICY=${POLICY:-$(latest_hf "$CKPT/$SFT_EXP")}
EXP=${EXP:-grpo_${STAGE}}
[ -n "$POLICY" ] || { echo "ERROR: no POLICY (run02 not finished, or set POLICY=/path/to/hf)"; exit 1; }

if [ "$STAGE" = "16k" ]; then RESP=16384; BATCH=${BATCH:-32}; TOKGPU=36864; else RESP=8192; BATCH=${BATCH:-64}; TOKGPU=20480; fi
EPOCHS=${EPOCHS:-2}; SAVE=${SAVE:-15}; TEST=${TEST:-15}; STEPS=${STEPS:-null}
if [ "${SMOKE:-0}" = "1" ]; then RESP=2048; BATCH=8; TOKGPU=8192; EPOCHS=1; SAVE=6; TEST=3; STEPS=6; fi
N=${N:-8}                                                 # rollouts per prompt (ReCAD: 8)

# CPU budget: verl's placement group takes 3 CPUs per GPU and the driver takes 1; the rest can be reward
# workers (Ray actors, 1 CPU each). Ray must be told the Slurm core count explicitly.
REWARD_WORKERS=${REWARD_WORKERS:-$(( NCPUS - 3 * NGPUS - 2 ))}
[ "$REWARD_WORKERS" -ge 2 ] || REWARD_WORKERS=2
export BREP_REWARD_TIMEOUT=${BREP_REWARD_TIMEOUT:-60}

echo "[grpo] policy=$POLICY stage=$STAGE resp=$RESP batch=$BATCH n=$N steps=$STEPS reward_workers=$REWARD_WORKERS exp=$EXP"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA/rl_train_${STAGE}.parquet" \
  data.val_files="$DATA/rl_val.parquet" \
  data.train_batch_size="$BATCH" \
  data.max_prompt_length=1280 \
  data.max_response_length="$RESP" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="$POLICY" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=$(( BATCH / 2 )) \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOKGPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL:-0.6}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOKGPU" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$TOKGPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$REPO/brepgrpo/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=remote \
  reward.num_workers="$REWARD_WORKERS" \
  ray_kwargs.ray_init.num_cpus="$NCPUS" \
  trainer.balance_batch=True \
  trainer.val_before_train=True \
  trainer.logger="$LOGGER" \
  trainer.project_name=rltuned-cad \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$CKPT/$EXP" \
  trainer.validation_data_dir="$REPO/runs/$EXP/val_generations" \
  trainer.n_gpus_per_node="$NGPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE" \
  trainer.test_freq="$TEST" \
  trainer.total_epochs="$EPOCHS" \
  trainer.total_training_steps="$STEPS" \
  trainer.max_actor_ckpt_to_keep=3 \
  "$@"

echo "[grpo] done. HF export: $(latest_hf "$CKPT/$EXP")"
