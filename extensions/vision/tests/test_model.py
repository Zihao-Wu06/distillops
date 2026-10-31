"""The model layer on two architectures: detection, rendering, token counts, vision modules, stops."""

import pytest

from distillops_vision import images, model as vm


def _row(task, ex):
    return task.gold_sft_example(ex)


def test_vision_models_are_detected_and_text_models_are_not(vlm, tmp_path):
    from transformers import GPT2Config

    assert vm.is_vision(vlm)
    GPT2Config(n_layer=1, n_head=1, n_embd=8).save_pretrained(tmp_path / "gpt2")
    assert not vm.is_vision(str(tmp_path / "gpt2"))


def test_render_places_the_templates_image_placeholder(vlm, toy_data):
    from vision_toy import ColourTask

    proc = vm.load_processor(vlm)
    task = ColourTask()
    text = vm.render(proc, task.student_messages(task.load_examples("test")[0]), add_generation_prompt=True)
    placeholder = proc.image_token
    assert text.count(placeholder) == 1 and "What colour is this image?" in text


def test_lengths_count_the_image_tokens_the_processor_inserts(vlm, toy_data):
    from vision_toy import ColourTask

    proc = vm.load_processor(vlm)
    lm = vm.load_model(vlm)
    task = ColourTask()
    ex = task.load_examples("test")[0]
    p_len, f_len, aligned = vm.lengths(proc, _row(task, ex), None)
    assert aligned and f_len > p_len
    text = vm.render(proc, task.student_messages(ex), add_generation_prompt=True)
    enc = vm.encode(proc, [text], [images.image_parts(task.student_messages(ex))])
    assert p_len == enc["input_ids"].shape[1]  # the length the model will really see

    def prompt_len(max_pixels):
        msgs = [{"role": "user", "content": [{"type": "image", "path": ex["image"], "max_pixels": max_pixels},
                                             {"type": "text", "text": "What colour?"}]}]
        return vm.prompt_tokens(proc, msgs, None)

    small, large = prompt_len(30 * 30), prompt_len(None)
    kind = type(lm).__name__
    if kind.startswith("Mllama"):  # cross-attention: the image is one token whatever its size
        assert small == large
    elif kind.startswith("Qwen2VL"):  # images in the sequence, tokens grow with the picture
        assert large > small
    else:  # LLaVA resizes every image to a fixed size: a fixed number of tokens, more than one
        assert small == large and int((enc["input_ids"] == min(vm.placeholder_ids(lm, proc))).sum()) > 1


def test_vision_modules_found_by_name_in_every_architecture(qwen2vl, llava, mllama):
    q, l, m = vm.load_model(qwen2vl), vm.load_model(llava), vm.load_model(mllama)
    assert vm.vision_modules(m) == ["model.vision_model"] and vm.projector_modules(m) == ["model.multi_modal_projector"]
    assert vm.vision_modules(q) == ["model.visual"] and vm.projector_modules(q) == ["model.visual.merger"]
    assert vm.vision_modules(l) == ["model.vision_tower"] and vm.projector_modules(l) == ["model.multi_modal_projector"]
    assert vm.vision_modules(q, [r"visual\.blocks"]) == ["model.visual.blocks"]  # user override


@pytest.mark.parametrize("train_projector", [True, False])
def test_freeze_vision_keeps_the_language_model_and_optionally_the_projector(vlm, train_projector):
    lm = vm.load_model(vlm)
    info = vm.freeze_vision(lm, train_vision=False, train_projector=train_projector)
    (vis,), (proj,) = info["vision_modules"], info["projector_modules"]
    for path, p in lm.named_parameters():
        if path.startswith(proj + "."):
            assert p.requires_grad is train_projector
        elif path.startswith(vis + "."):
            assert not p.requires_grad
        else:
            assert p.requires_grad  # the language model always trains
    assert info["frozen_params"] > 0


def test_freeze_without_a_vision_module_is_a_note_not_an_error(vlm, capsys):
    lm = vm.load_model(vlm)
    info = vm.freeze_vision(lm, patterns=["no_such_module"])
    assert info["vision_modules"] == [] and "no module matches" in capsys.readouterr().out


def test_stop_tokens_and_placeholders(qwen2vl, llava):
    from distillops.chat import end_of_turn_id, stop_token_ids

    for path, eot in ((qwen2vl, "<|im_end|>"), (llava, "<eot>")):  # mllama's EOS is its end of turn
        proc = vm.load_processor(path)
        view = vm.TemplateView(proc)
        assert view.convert_ids_to_tokens(end_of_turn_id(view)) == eot  # distillops' helper, through the view
        assert len(stop_token_ids(view)) == 2
        lm = vm.load_model(path)
        ph = vm.placeholder_ids(lm, proc)
        assert proc.tokenizer.convert_tokens_to_ids(proc.image_token) in ph
        assert set(vm.suppressed_ids(lm, proc)) >= ph


def test_plain_template_is_refused_for_vision_models(vlm):
    with pytest.raises(SystemExit, match="plain"):
        vm.load_processor(vlm, chat_template="plain")
