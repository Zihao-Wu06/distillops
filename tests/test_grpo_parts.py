from distillops.grpo import filter_by_signal, signal_stats


def test_signal_filter_keeps_prompts_whose_samples_differ():
    rows = [{"id": i} for i in range(6)]
    scores = [[0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0.2, 0.7, 0.7]]
    assert [r["id"] for r in filter_by_signal(rows, scores)] == [1, 3, 5]  # equal partial credit: no signal
    st = signal_stats(scores)
    assert (st["mixed"], st["all_wrong"], st["all_right"], st["all_same_partial"]) == (3, 1, 1, 1)
    assert st["samples_per_example"] == 3


def test_epochs_to_steps_and_sparse_reward_advice():
    from distillops.grpo import GRPORunConfig, diagnose_pass_rates, steps_for_epochs

    cfg = GRPORunConfig(model="m", output_dir="o", batch_size=64, grad_accum=4, num_generations=8)  # 32 prompts/step
    assert steps_for_epochs(661, 3, cfg) == 62
    sparse = diagnose_pass_rates({"examples": 100, "mixed": 5, "all_wrong": 94, "all_right": 1})
    assert len(sparse) == 2 and "sparse" in sparse[1]
    assert len(diagnose_pass_rates({"examples": 100, "mixed": 40, "all_wrong": 30, "all_right": 30})) == 1


def test_relative_weight_change_detects_a_model_that_did_not_move():
    torch = __import__("pytest").importorskip("torch")
    from distillops.grpo import _relative_change, _snapshot

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    before = _snapshot(model)
    assert _relative_change(before, model) == 0.0
    with torch.no_grad():
        model[0].weight.mul_(1.1)
    assert _relative_change(before, model) > 0.01


def test_expected_change_floor_scales_with_steps_and_lr():
    from distillops.preflight import min_expected_change

    short, long_ = min_expected_change(1e-6, 10, 0.02), min_expected_change(1e-6, 151, 0.02)
    assert short < long_ < 1e-4  # bf16 run measured 3.8e-5 over 151 steps -> flagged; fp32 10 steps 8.5e-5 -> ok
    assert 3.8e-5 < long_ and 8.5e-5 > short


def test_check_disk_refuses_when_too_small(tmp_path):
    import pytest

    from distillops.preflight import check_disk

    check_disk(tmp_path / "out", 1, "tiny")
    with pytest.raises(SystemExit):
        check_disk(tmp_path / "out", 10**18, "huge")


def test_precision_settings_cpu_fallback():
    torch = __import__("pytest").importorskip("torch")
    from distillops.precision import train_settings

    s = train_settings(full_finetune=True)
    assert s["model_dtype"] == torch.float32  # full fine-tuning always keeps fp32 master weights
    assert not (s["bf16"] and s["fp16"])
    assert s["use_cpu"] == (not torch.cuda.is_available())  # no CUDA: CPU, not Apple's MPS


def test_entropy_guard_is_relative_to_the_starting_entropy():
    __import__("pytest").importorskip("transformers")
    from types import SimpleNamespace

    from distillops.grpo import entropy_guard

    def run(series):
        g = entropy_guard(0.1, 2)
        control, state = SimpleNamespace(should_training_stop=False), SimpleNamespace(global_step=0)
        for step, ent in enumerate(series, 1):
            state.global_step = step
            g.on_log(None, state, control, logs={"entropy": ent})
        return g.stopped_at

    assert run([0.87, 0.5, 0.05, 0.3, 0.02, 0.01]) == 6  # floor .087; one low reading alone is not enough
    assert run([0.08, 0.06, 0.05, 0.04]) is None  # a task whose entropy is naturally low is not flagged


def test_micro_batch_shrinks_for_long_completions_and_keeps_the_effective_batch():
    from distillops.grpo import GRPORunConfig, fit_micro_batch

    gib = 2**30
    long_ = GRPORunConfig(model="m", output_dir="o", batch_size=64, grad_accum=4, max_completion_length=512,
                          vllm_gpu_memory_utilization=0.25)
    b, a = fit_micro_batch(long_, total_bytes=83 * gib, vocab=151936, train_state_bytes=10 * gib)
    assert b < 64 and b * a == 256  # math solutions: smaller micro-batch, same 256 completions per step
    short = GRPORunConfig(model="m", output_dir="o", batch_size=64, grad_accum=4, max_completion_length=96,
                          vllm_gpu_memory_utilization=0.3)
    assert fit_micro_batch(short, total_bytes=83 * gib, vocab=151936, train_state_bytes=27 * gib) == (64, 4)
