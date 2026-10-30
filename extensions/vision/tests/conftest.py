"""Tiny vision-language models and a toy task, built offline (no network, no GPU).

- tiny Qwen2-VL: ViT + merger; images become <|image_pad|> tokens in the sequence;
- tiny LLaVA: CLIP ViT + a separate projector; images become <image> tokens;
- tiny Llama 3.2 Vision (Mllama): the image stays one <|image|> token and is read through
  cross-attention layers.
Three architectures with different module names and placeholders, so nothing passes by matching one.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))  # the toy task module (vision_toy)

QWEN_SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
                "<|video_pad|>"]
QWEN_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for c in m['content'] %}{% if c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LLAVA_SPECIAL = ["</s>", "<eot>", "<image>"]
LLAVA_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {% for c in m['content'] %}{% if c['type'] == 'image' %}<image>"
    "{% else %}{{ c['text'] }}{% endif %}{% endfor %}<eot>\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}")
CORPUS = ["user\nWhat colour is this image?\nassistant\nCOLOUR: red\n", "COLOUR: blue green red"] * 50


def _bpe(special, cls, **kw):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=special,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(CORPUS, trainer)
    tok = cls(tokenizer_object=tk, **kw)
    tok.add_special_tokens({"additional_special_tokens": special[1:]})
    return tok


def build_qwen2vl(d: Path) -> Path:
    import torch
    from transformers import (Qwen2TokenizerFast, Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLImageProcessor,
                              Qwen2VLProcessor, Qwen2VLVideoProcessor)

    tok = _bpe(QWEN_SPECIAL, Qwen2TokenizerFast, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok.chat_template = QWEN_TEMPLATE
    ip = Qwen2VLImageProcessor(min_pixels=28 * 28, max_pixels=112 * 112, patch_size=14, merge_size=2, temporal_patch_size=2)
    proc = Qwen2VLProcessor(image_processor=ip, tokenizer=tok, video_processor=Qwen2VLVideoProcessor(),
                            chat_template=QWEN_TEMPLATE)
    ids = {t: tok.convert_tokens_to_ids(t) for t in QWEN_SPECIAL}
    cfg = Qwen2VLConfig(
        vision_config=dict(depth=1, embed_dim=32, num_heads=2, hidden_size=32, mlp_ratio=2, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, in_chans=3),
        text_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         intermediate_size=64, vocab_size=len(tok), max_position_embeddings=512,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    Qwen2VLForConditionalGeneration(cfg).save_pretrained(d)
    proc.save_pretrained(d)
    return d


def build_llava(d: Path) -> Path:
    import torch
    from transformers import (CLIPImageProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                              LlavaForConditionalGeneration, LlavaProcessor, PreTrainedTokenizerFast)

    tok = _bpe(LLAVA_SPECIAL, PreTrainedTokenizerFast, eos_token="</s>", pad_token="</s>")
    tok.chat_template = LLAVA_TEMPLATE
    ip = CLIPImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28})
    proc = LlavaProcessor(image_processor=ip, tokenizer=tok, patch_size=14, vision_feature_select_strategy="default",
                          num_additional_image_tokens=1, image_token="<image>", chat_template=LLAVA_TEMPLATE)
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
                                       image_size=28, patch_size=14),
        text_config=LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=1, vocab_size=len(tok), max_position_embeddings=256),
        image_token_index=tok.convert_tokens_to_ids("<image>"), vision_feature_select_strategy="default",
        vision_feature_layer=-1, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    LlavaForConditionalGeneration(cfg).save_pretrained(d)
    proc.save_pretrained(d)
    return d


MLLAMA_SPECIAL = ["<|eot_id|>", "<|image|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]
MLLAMA_TEMPLATE = (
    "{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}<|image|>{% else %}{{ c['text'] }}{% endif %}{% endfor %}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n{% endif %}")


def build_mllama(d: Path) -> Path:
    import torch
    from transformers import (MllamaConfig, MllamaForConditionalGeneration, MllamaImageProcessor, MllamaProcessor,
                              PreTrainedTokenizerFast)

    tok = _bpe(MLLAMA_SPECIAL, PreTrainedTokenizerFast, eos_token="<|eot_id|>", pad_token="<|eot_id|>",
               bos_token="<|begin_of_text|>")
    tok.chat_template = MLLAMA_TEMPLATE
    ip = MllamaImageProcessor(size={"height": 28, "width": 28}, max_image_tiles=1)
    proc = MllamaProcessor(image_processor=ip, tokenizer=tok, chat_template=MLLAMA_TEMPLATE)
    cfg = MllamaConfig(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_global_layers=1,
                           attention_heads=2, image_size=28, patch_size=14, max_num_tiles=1,
                           intermediate_layers_indices=[0], vision_output_dim=64, supported_aspect_ratios=[[1, 1]]),
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, vocab_size=len(tok), cross_attention_layers=[1],
                         max_position_embeddings=256, rope_scaling={"rope_type": "default"},
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id),
        image_token_index=tok.convert_tokens_to_ids("<|image|>"))
    torch.manual_seed(0)
    MllamaForConditionalGeneration(cfg).save_pretrained(d)
    proc.save_pretrained(d)
    return d


@pytest.fixture(scope="session")
def mllama(tmp_path_factory):
    return str(build_mllama(tmp_path_factory.mktemp("models") / "tiny-mllama"))


@pytest.fixture(scope="session")
def qwen2vl(tmp_path_factory):
    pytest.importorskip("torchvision")  # Qwen2-VL's video processor needs it
    return str(build_qwen2vl(tmp_path_factory.mktemp("models") / "tiny-qwen2vl"))


@pytest.fixture(scope="session")
def llava(tmp_path_factory):
    return str(build_llava(tmp_path_factory.mktemp("models") / "tiny-llava"))


@pytest.fixture(params=["qwen2vl", "llava", "mllama"])
def vlm(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def toy_data(tmp_path, monkeypatch):
    """Colour images (train / valid / test) for the toy task, in tmp_path/data/colours."""
    from vision_toy import make_data

    monkeypatch.chdir(tmp_path)
    make_data(tmp_path / "data" / "colours")
    return tmp_path
