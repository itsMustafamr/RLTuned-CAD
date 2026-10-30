"""Offline check that every override in scripts/_sft.sh and scripts/_grpo.sh is a real verl v0.9.1
config key (Hydra compose in struct mode), including the FSDP wrap list the launchers add for InternVL
checkpoints. No GPU or verl install needed, only a verl checkout + hydra-core:
    python tests/check_verl_configs.py /path/to/verl"""
import os, re, sys
from hydra import compose, initialize_config_dir

VERL = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/work/mech-ai-scratch/mohd7/verl")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(VERL, "verl", "trainer", "config")


def overrides_from(script, marker, env):
    body = open(os.path.join(REPO, "scripts", script)).read().split(marker)[-1]
    out = []
    for line in body.splitlines():
        line = line.strip().rstrip("\\").strip()
        if not line or line.startswith("#") or line.startswith('"$@"') or line.startswith("echo") or "=" not in line:
            continue
        line = re.sub(r"\$\(\(.*?\)\)", "8", line)
        for k in sorted(env, key=len, reverse=True):          # longest names first: $NCPUS before $N
            line = line.replace("${%s}" % k, env[k]).replace("$%s" % k, env[k])
        line = re.sub(r"\$\{[A-Z_]+:-([^}]*)\}", r"\1", line)
        out.append(line.replace('"', ""))
    return out


def wrap_list():
    """The layer list scripts/_common.sh prints for InternVL models, read from the file so the two cannot drift."""
    src = open(os.path.join(REPO, "scripts", "_common.sh")).read()
    m = re.search(r'echo "(\[[A-Za-z0-9_,]+\])"', src)
    assert m, "fsdp_wrap_layers() in scripts/_common.sh no longer prints a bracketed layer list"
    return m.group(1)


env = dict(DATA="/d", TRAIN="/d/sft_train.parquet", BATCH="64", RESP="8192", TOKGPU="20480", POLICY="/m", BASE_MODEL="/m",
           LR="1e-6", EPOCHS="2", SAVE="15", TEST="15", STEPS="null", N="8", REWARD_WORKERS="3", NCPUS="48", REPO="/r",
           CKPT="/c", EXP="x", SFT_EXP="s", STAGE="8k", NGPUS="8", LOGGER="[console,tensorboard]",
           REMOVE_PADDING="False", ATTN_IMPL="sdpa", DYNAMIC_BSZ="False")
WRAP = wrap_list()
for script, marker, name in (("_sft.sh", "-m verl.trainer.sft_trainer", "sft_trainer_engine"),
                             ("_grpo.sh", "-m verl.trainer.main_ppo", "ppo_trainer")):
    ovs = overrides_from(script, marker, env)
    # the InternVL case: the launcher appends these (WRAP_ARGS) when fsdp_wrap_layers() prints a list
    extra = ([f"+engine.wrap_policy.transformer_layer_cls_to_wrap={WRAP}"] if name == "sft_trainer_engine" else
             [f"+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap={WRAP}",
              f"+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap={WRAP}"])
    with initialize_config_dir(config_dir=CONF, version_base=None):
        cfg = compose(config_name=name, overrides=ovs)                 # Qwen2.5-VL: no wrap override
        cfg_iv = compose(config_name=name, overrides=ovs + extra)      # InternVL: explicit layer list
    print(f"{script}: {len(ovs)} overrides compose cleanly against verl ({name}); +{len(extra)} InternVL wrap override(s) compose")
    layers = [s.strip() for s in WRAP.strip("[]").split(",")]
    if name == "ppo_trainer":
        assert cfg.reward.custom_reward_function.name == "compute_score"
        assert cfg.reward.reward_manager.name == "remote" and cfg.reward.num_workers == 3
        assert cfg.actor_rollout_ref.rollout.n == 8 and cfg.data.shuffle is False
        assert cfg.ray_kwargs.ray_init.num_cpus == 48
        assert cfg.actor_rollout_ref.model.override_config.attn_implementation == "sdpa"
        a = cfg.actor_rollout_ref   # without flash-attn: one sample per micro-batch, actor and rollout must agree
        assert a.actor.use_dynamic_bsz is False and a.rollout.log_prob_use_dynamic_bsz is False and a.ref.log_prob_use_dynamic_bsz is False
        assert a.actor.ppo_micro_batch_size_per_gpu == a.rollout.log_prob_micro_batch_size_per_gpu == a.ref.log_prob_micro_batch_size_per_gpu == 1
        assert "transformer_layer_cls_to_wrap" not in cfg.actor_rollout_ref.actor.fsdp_config.wrap_policy
        assert list(cfg_iv.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap) == layers
        assert list(cfg_iv.actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap) == layers
    else:
        assert cfg.data.use_dynamic_bsz is False and cfg.data.micro_batch_size_per_gpu == 1
        assert "transformer_layer_cls_to_wrap" not in cfg.engine.wrap_policy
        assert list(cfg_iv.engine.wrap_policy.transformer_layer_cls_to_wrap) == layers
print(f"InternVL FSDP wrap list: {WRAP}")
