"""Behaviour that must hold on any GPU and for any task (no GPU, network or vLLM needed)."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _fake_gpu(monkeypatch, major, hip=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (major, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)  # torch says yes, emulated or not
    monkeypatch.setattr(torch.version, "hip", hip)


@pytest.mark.parametrize("major, expected", [(7, torch.float16), (8, torch.bfloat16), (9, torch.bfloat16)])
def test_bf16_only_where_native(monkeypatch, major, expected):
    from distillops.precision import compute_dtype

    _fake_gpu(monkeypatch, major)  # 7 = T4 / V100: bf16 would only be emulated
    assert compute_dtype() == expected


def test_rocm_follows_torch(monkeypatch):
    from distillops.precision import compute_dtype

    _fake_gpu(monkeypatch, 0, hip="6.0")
    assert compute_dtype() == torch.bfloat16


class _Tok:
    chat_template = "x"
    eos_token = "</s>"
    pad_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        assert kw == {"enable_thinking": False}
        return messages[0]["content"]


def test_pass_rate_filter_samples_like_grpo(monkeypatch):
    seen = {}

    class LLM:
        def __init__(self, **kw):
            seen["llm"] = kw

        def get_tokenizer(self):
            return _Tok()

        def generate(self, prompts, params):
            seen["params"] = params
            # prompt "a": samples differ; "b": the same partial score every time
            texts = {"a": ["1", "0", "1"], "b": ["0.5", "0.5", "0.5"]}
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t) for t in texts[p]]) for p in prompts]

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    monkeypatch.setattr("distillops.memory.require_free_memory", lambda *a: None)  # no real GPU check in a unit test
    from distillops.grpo import estimate_pass_rates, filter_by_signal

    task = types.SimpleNamespace(student_messages=lambda ex: [{"role": "user", "content": ex}],
                                 score=lambda ex, text: float(text))
    scores, stats = estimate_pass_rates("m", task, ["a", "b"], n=3, max_tokens=256, max_prompt_length=512,
                                        chat_template_kwargs={"enable_thinking": False})
    assert seen["llm"]["max_model_len"] == 768  # prompt + completion, as TRL sets it
    assert seen["params"].kw["truncate_prompt_tokens"] == 512 and seen["params"].kw["max_tokens"] == 256
    assert scores == [[1.0, 0.0, 1.0], [0.5, 0.5, 0.5]]
    assert filter_by_signal(["a", "b"], scores) == ["a"]
    assert stats["mixed"] == 1 and stats["all_same_partial"] == 1


# -- A5: runs with different settings never share an output folder --------------------------------

def test_task_args_change_default_paths():
    from distillops.cli import args_suffix

    assert args_suffix({}) == ""
    assert args_suffix({"label_reasoning": "human"}).startswith("_label_reasoning-human-")
    assert args_suffix({"b": 1, "a": "x/y"}).startswith("_a-x-y_b-1-")  # sorted, filesystem-safe
    assert len(args_suffix({"k": "v" * 200})) <= 58
    # settings that read the same once normalised still get different paths
    assert args_suffix({"a": "x/y"}) != args_suffix({"a": "x-y"})
    assert args_suffix({"a": "x", "b": 1}) != args_suffix({"a": "x_b-1"})
    assert args_suffix({"a": 1, "b": 2}) == args_suffix({"b": 2, "a": 1})


def test_resume_refuses_a_different_teacher(tmp_path, capsys):
    from distillops.rejection import check_run_settings

    run = {"task": "m:T", "task_args": {}, "split": "train", "teacher": {"model": "A"}}
    check_run_settings(tmp_path, run)
    (tmp_path / "log.jsonl").write_text('{"id": "1"}\n')  # the run produced results
    check_run_settings(tmp_path, run)  # same settings: resume is fine
    with pytest.raises(SystemExit, match="teacher"):
        check_run_settings(tmp_path, {**run, "teacher": {"model": "B"}})
    empty = tmp_path / "empty"  # a run that produced nothing (e.g. a typo in the teacher config) may be redone
    check_run_settings(empty, run)
    check_run_settings(empty, {**run, "teacher": {"model": "fixed"}})
    old = tmp_path / "old"
    old.mkdir()
    (old / "log.jsonl").write_text("{}\n")
    check_run_settings(old, run)  # older run without run.json: recorded, with a note
    assert "no run.json" in capsys.readouterr().out and (old / "run.json").exists()


def test_teacher_command_keeps_runs_apart(tmp_path, monkeypatch):
    from distillops import cli

    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"base_url: http://x\nmodel: A\ncache_path: {tmp_path / 'c.sqlite'}\n")
    monkeypatch.setenv("TEACHER_API_KEY", "test")

    async def fake_rejection(task, teacher, examples, out, max_attempts):
        return {}

    monkeypatch.setattr("distillops.rejection.rejection_sample", fake_rejection)
    out = tmp_path / "run"
    base = ["teacher", "--task", "tests.test_framework:EvenTask", "--teacher", str(cfg), "--out", str(out)]
    monkeypatch.setattr("tests.test_framework.EvenTask.load_examples", lambda self, split: [1, 2], raising=False)
    cli.main(base)
    cfg.write_text(f"base_url: http://x\nmodel: B\ncache_path: {tmp_path / 'c.sqlite'}\n")
    (out / "log.jsonl").write_text('{"id": "1"}\n')  # the first run produced results
    with pytest.raises(SystemExit, match="different settings"):
        cli.main(base)


def test_empty_template_options_in_grpo_config(tmp_path, monkeypatch):
    from distillops import cli

    cfg = tmp_path / "g.yaml"
    cfg.write_text("model: m\noutput_dir: o\nchat_template_kwargs:\n#  enable_thinking: false\n")
    seen = {}
    monkeypatch.setattr("tests.test_framework.EvenTask.load_examples", lambda self, split: [], raising=False)
    monkeypatch.setattr("distillops.grpo.train", lambda c, *a, **k: seen.setdefault("cfg", c))
    monkeypatch.setattr("distillops.chat.load_tokenizer", lambda model, template=None: _Tok())
    cli.main(["grpo", str(cfg), "--task", "tests.test_framework:EvenTask", "--eval-every", "0",
              "--template-arg", "x=1"])
    assert seen["cfg"].chat_template_kwargs == {"x": 1}


# -- A6: `extra` can set any trainer argument -------------------------------------------------------

def test_extra_overrides_framework_defaults(capsys):
    from distillops.trainargs import merge_extra

    args = merge_extra({"report_to": "none", "learning_rate": 1e-4}, {"report_to": "wandb", "lr_scheduler_type": "linear"},
                       {"learning_rate": "lr"}, "SFT")
    assert args == {"report_to": "wandb", "learning_rate": 1e-4, "lr_scheduler_type": "linear"}
    assert "report_to" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="learning_rate -> `lr`"):
        merge_extra({"learning_rate": 1e-4}, {"learning_rate": 1e-5}, {"learning_rate": "lr"}, "SFT")


def test_extra_merges_dict_arguments():
    from distillops.trainargs import merge_extra

    args = merge_extra({"model_init_kwargs": {"dtype": "float32"}},
                       {"model_init_kwargs": {"attn_implementation": "sdpa"}}, {}, "GRPO")
    assert args["model_init_kwargs"] == {"dtype": "float32", "attn_implementation": "sdpa"}  # dtype kept


def test_sft_warmup_field_depends_on_transformers_version():
    from distillops.sft import _warmup

    key = next(iter(_warmup(0.03)))
    major = int(__import__("transformers").__version__.split(".")[0])
    assert key == ("warmup_steps" if major >= 5 else "warmup_ratio")  # on 4.x warmup_steps stays free for extra


def test_trainer_configs_accept_overrides_of_framework_arguments():
    trl = pytest.importorskip("trl")
    from distillops.precision import train_settings
    from distillops.trainargs import merge_extra

    prec = train_settings(full_finetune=True)  # as sft/grpo pass it (a CPU-only machine must not get bf16)
    own = {"output_dir": "o", "report_to": "none", "gradient_checkpointing": True,
           "bf16": prec["bf16"], "fp16": prec["fp16"], "use_cpu": prec["use_cpu"]}
    for cls in (trl.SFTConfig, trl.GRPOConfig):  # the framework's own keys plus a user override build fine
        cls(**merge_extra(own, {"report_to": [], "gradient_checkpointing": False, "lr_scheduler_type": "linear"},
                          {}, "x"))


def test_grpo_lora_settings_are_configurable():
    from distillops.grpo import GRPORunConfig

    cfg = GRPORunConfig(model="m", output_dir="o", lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.1)
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout) == (8, 16, 0.1)


# -- B7: any OpenAI-compatible teacher --------------------------------------------------------------

class _Client:
    def __init__(self, text="ok", reasoning_field=None):
        self.kwargs, self.text, self.reasoning_field = [], text, reasoning_field
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kw):
        self.kwargs.append(kw)
        msg = types.SimpleNamespace(content=self.text)
        if self.reasoning_field:
            setattr(msg, self.reasoning_field, "thought")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def _teacher(tmp_path, client, **kw):
    from distillops.teacher import Teacher, TeacherConfig

    cfg = TeacherConfig(base_url=kw.pop("base_url", "https://a/v1"), model="m", cache_path=str(tmp_path / "c.sqlite"), **kw)
    return Teacher(cfg, client=client)


async def test_reasoning_models_get_their_own_sampling_fields(tmp_path):
    c = _Client()
    await _teacher(tmp_path, c, temperature=None, max_tokens=2000, max_tokens_param="max_completion_tokens").complete(
        [{"role": "user", "content": "q"}])
    assert "temperature" not in c.kwargs[0] and "max_tokens" not in c.kwargs[0]
    assert c.kwargs[0]["max_completion_tokens"] == 2000


def test_unknown_token_field_is_refused():
    from distillops.teacher import TeacherConfig

    with pytest.raises(ValueError, match="extra_body"):
        TeacherConfig(base_url="x", model="m", max_tokens_param="max_output_tokens")


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_reasoning_text_under_either_field_name(tmp_path, field):
    out = await _teacher(tmp_path, _Client(reasoning_field=field)).complete([{"role": "user", "content": "q"}])
    assert out.reasoning == "thought"


async def test_cache_is_per_endpoint_and_reads_older_caches(tmp_path):
    import json

    msgs = [{"role": "user", "content": "q"}]
    a = _teacher(tmp_path, _Client("from A"), base_url="https://a/v1")
    assert (await a.complete(msgs)).text == "from A"
    b = _teacher(tmp_path, _Client("from B"), base_url="https://b/v1")  # same model name, other provider
    assert (await b.complete(msgs)).text == "from B"
    assert (await b.complete(msgs)).cached  # second call: from the cache, still B's answer
    # an entry written before endpoints were recorded is still served (re-runs stay free)
    legacy = _teacher(tmp_path, _Client("new"), base_url="https://c/v1")
    params = {**legacy.cfg.request_params(), "extra_body": {}}
    legacy._cache.put(legacy._key(msgs, 7, params, with_endpoint=False),
                      {"text": "old", "reasoning": None, "prompt_tokens": 1, "completion_tokens": 1, "cache_hit_tokens": 0})
    hit = await legacy.complete(msgs, sample_id=7)
    assert hit.text == "old" and hit.cached
    rows = [json.loads(v) for (v,) in legacy._cache.db.execute("SELECT v FROM kv")]
    assert {r.get("base_url") for r in rows} == {"https://a/v1", "https://b/v1", None}


def test_default_request_fields_keep_older_cache_keys():
    from distillops.teacher import TeacherConfig

    assert TeacherConfig(base_url="x", model="m").request_params() == {"temperature": 1.0, "max_tokens": 1024}


async def test_usage_counts_only_this_teacher(tmp_path, capsys):
    from distillops import cli

    msgs = [{"role": "user", "content": "q"}]
    await _teacher(tmp_path, _Client(), base_url="https://a/v1").complete(msgs)
    await _teacher(tmp_path, _Client(), base_url="https://b/v1").complete(msgs)
    cfg = tmp_path / "t.yaml"
    cfg.write_text(f"base_url: https://a/v1\nmodel: m\ncache_path: {tmp_path / 'c.sqlite'}\n"
                   "price: {input_cache_miss: 1.0, output: 4.0}\n")  # an older config: price is ignored
    cli.main(["usage", "--teacher", str(cfg)])
    out = capsys.readouterr().out
    assert "price` is no longer used" in out
    assert "calls 1 " in out and "prompt tokens 10 " in out and "completion tokens 5" in out
    assert "1 cached calls from other teachers" in out and "cost" not in out.lower()


# -- B8 / B9: generic chat-template options and dataset-level metrics -------------------------------

def test_template_args_and_no_think_shorthand():
    from distillops.cli import template_args

    assert template_args([]) == {}
    assert template_args(["enable_thinking=false", "reasoning_effort=low"]) == {"enable_thinking": False,
                                                                                "reasoning_effort": "low"}
    assert template_args([], no_think=True) == {"enable_thinking": False}


def test_eval_passes_template_options_only_when_given(monkeypatch, tmp_path):
    from distillops import cli

    seen = []
    monkeypatch.setattr("tests.test_framework.EvenTask.load_examples", lambda self, split: [1], raising=False)
    monkeypatch.setattr("tests.test_framework.EvenTask.evaluate", lambda self, model, ex, out, **kw: seen.append(kw),
                        raising=False)
    base = ["eval", "--task", "tests.test_framework:EvenTask", "--model", "m", "--out", str(tmp_path)]
    cli.main(base)
    cli.main(base + ["--no-think", "--template-arg", "x=1"])
    assert "chat_template_kwargs" not in seen[0]
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": False, "x": 1}


def test_extra_metrics_are_reported(tmp_path):
    from distillops.evaluate import evaluate_task

    class Exact:
        name = "exact"

        def student_messages(self, ex):
            return [{"role": "user", "content": ex}]

        def score(self, ex, completion):
            return float(ex == completion)

        def example_id(self, ex):
            return ex

        def extra_metrics(self, examples, completions, scores):
            return {"distinct_outputs": len(set(completions))}

    gen = lambda model, msgs, **kw: ["a", "a", "a"]  # noqa: E731
    m = evaluate_task(Exact(), "m", ["a", "b", "c"], tmp_path, generate_fn=gen)
    assert m["score"] == round(1 / 3, 4) and m["distinct_outputs"] == 1

    class Clash(Exact):
        def extra_metrics(self, examples, completions, scores):
            return {"score": 1.0}

    with pytest.raises(ValueError, match="score"):
        evaluate_task(Clash(), "m", ["a"], tmp_path, generate_fn=gen)


def test_rec_evaluate_reads_the_generic_option():
    import inspect

    from tasks.rec.evaluate import evaluate_rec

    assert "chat_template_kwargs" in inspect.signature(evaluate_rec).parameters


# -- B10: model size from the weights, or a config estimate that knows MoE --------------------------

def _write_safetensors(path, shapes):
    import json
    import struct

    header = {f"t{i}": {"dtype": "F32", "shape": s, "data_offsets": [0, 0]} for i, s in enumerate(shapes)}
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_param_count_reads_safetensors_headers_whatever_the_dtype(tmp_path):
    import json

    from distillops.preflight import param_count

    _write_safetensors(tmp_path / "a.safetensors", [[10, 20], [5]])  # fp32 files: still 205 parameters
    assert param_count(str(tmp_path)) == 205
    shard = tmp_path / "sharded"
    shard.mkdir()
    _write_safetensors(shard / "m-1.safetensors", [[100]])
    _write_safetensors(shard / "m-2.safetensors", [[50, 2]])
    _write_safetensors(shard / "consolidated.safetensors", [[999]])  # not in the index: ignored
    (shard / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}}))
    assert param_count(str(shard)) == 200


def test_size_estimate_knows_mixture_of_experts_and_gated_mlps():
    from types import SimpleNamespace

    from distillops.preflight import estimate_params

    moe = SimpleNamespace(num_hidden_layers=48, num_attention_heads=32, num_key_value_heads=4, hidden_size=2048,
                          head_dim=128, vocab_size=151936, intermediate_size=6144, moe_intermediate_size=768,
                          num_experts=128, hidden_act="silu", tie_word_embeddings=False)  # Qwen3-30B-A3B: 30.5B
    assert 29e9 < estimate_params(moe) < 32e9
    dense = SimpleNamespace(num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, hidden_size=1024,
                            head_dim=128, vocab_size=151936, intermediate_size=3072, hidden_act="silu",
                            tie_word_embeddings=True)  # Qwen3-0.6B: 0.60B
    assert 0.55e9 < estimate_params(dense) < 0.65e9
    gpt2 = SimpleNamespace(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, activation_function="gelu_new")
    assert 0.12e9 < estimate_params(gpt2) < 0.13e9  # 124M


def test_micro_batch_warns_when_training_state_cannot_fit(capsys):
    from distillops.grpo import GRPORunConfig, fit_micro_batch

    cfg = GRPORunConfig(model="m", output_dir="o", batch_size=8, grad_accum=4)
    assert fit_micro_batch(cfg, total_bytes=24 * 2**30, vocab=151936, train_state_bytes=30 * 2**30) == (1, 32)
    assert "does not fit" in capsys.readouterr().out


# -- B11: SFT disk estimate and clear errors --------------------------------------------------------

def test_sft_disk_estimate_matches_what_is_saved():
    from types import SimpleNamespace

    from distillops.sft import sft_disk_bytes

    a = SimpleNamespace(save_strategy="epoch", save_total_limit=1, save_only_model=True, num_train_epochs=2,
                        per_device_train_batch_size=8, gradient_accumulation_steps=4, max_steps=-1, save_steps=500)
    p = 10**9
    assert sft_disk_bytes(p, lora=False, args=a, n_rows=1000) == 8 * p  # one fp32 checkpoint + fp32 final
    a.save_only_model, a.save_total_limit = False, None  # optimizer state, one checkpoint per epoch
    assert sft_disk_bytes(p, lora=False, args=a, n_rows=1000) == (2 * 12 + 4) * p
    assert sft_disk_bytes(p, lora=True, args=a, n_rows=1000) == 2 * p  # merged copy; adapters are small
    assert sft_disk_bytes(0, lora=False, args=a, n_rows=1000) == 0


def test_masked_example_warns_instead_of_crashing():
    from distillops.sft import show_masked_example

    class Rewriting:  # a template that renders earlier turns differently once a reply follows
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "P:" if add_generation_prompt else "Q:" + messages[-1]["content"]

    out = show_masked_example(Rewriting(), {"prompt": [{"role": "user", "content": "x"}],
                                            "completion": [{"role": "assistant", "content": "y"}]})
    assert out.startswith("WARNING")


def test_sft_row_errors_are_clear():
    from distillops.sft import _require_rows

    with pytest.raises(SystemExit, match="has no rows"):
        _require_rows({"rows_in": 0, "rows_kept": 0, "len_max": 0}, "train.jsonl", 1024)
    with pytest.raises(SystemExit, match="longer than max_length=64"):
        _require_rows({"rows_in": 3, "rows_kept": 0, "len_max": 99}, "eval.jsonl", 64)
    _require_rows({"rows_in": 3, "rows_kept": 1, "len_max": 99}, "x", 64)


# -- B13: LoRA weight-change check ------------------------------------------------------------------

def test_weight_change_ignores_matrices_that_start_at_zero():
    from distillops.grpo import _relative_change, _snapshot

    m = torch.nn.Module()
    m.A = torch.nn.Parameter(torch.randn(4, 8))
    m.B = torch.nn.Parameter(torch.zeros(8, 4))  # LoRA's B
    before = _snapshot(m)
    assert list(before) == ["A"]
    with torch.no_grad():
        m.B += 1.0
    assert _relative_change(before, m) == 0.0  # B moving from zero says nothing about A


def test_grpo_disk_estimate():
    from types import SimpleNamespace

    from distillops.grpo import grpo_disk_bytes

    a = SimpleNamespace(save_only_model=True, save_total_limit=1, max_steps=200, save_steps=100, save_strategy="steps")
    p = 10**9
    assert grpo_disk_bytes(p, lora=False, args=a, validation=True) == (4 + 2 * 4) * p  # checkpoint, best, last
    assert grpo_disk_bytes(p, lora=True, args=a, validation=True, start_bytes=4 * p) == 4 * p  # fp32 start copy
    assert grpo_disk_bytes(p, lora=True, args=a, validation=False) == 0
    a.save_strategy = "no"  # no periodic checkpoints: the best and the last model only
    assert grpo_disk_bytes(p, lora=False, args=a, validation=True) == 2 * 4 * p


def test_eval_folders_of_different_runs_do_not_collide():
    from distillops.cli import model_tag

    assert model_tag("checkpoints/gsm8k_0.6b_teacher/merged") == "gsm8k_0.6b_teacher-merged"
    assert model_tag("checkpoints/gsm8k_0.6b_human/merged/") == "gsm8k_0.6b_human-merged"
    assert model_tag("checkpoints/run/last") == "run-last"
    assert model_tag("checkpoints/gsm8k_0.6b_grpo") == "gsm8k_0.6b_grpo"
    assert model_tag("Qwen/Qwen3-0.6B") == "Qwen3-0.6B"
    assert model_tag("merged") == "merged"
