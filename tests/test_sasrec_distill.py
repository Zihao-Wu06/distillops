"""LLM -> SASRec distillation pieces: fused item representation and ranking soft labels."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tasks.rec.baselines import SASRecConfig, _build_model, _soft_labels  # noqa: E402


def test_soft_labels_follow_rank_and_skip_missing_rows():
    ids, w = _soft_labels([[3, 1, 2], [], [5]], n_items=10)
    assert ids.shape == (3, 3) and ids[1].tolist() == [-1, -1, -1]
    assert torch.allclose(w.sum(1), torch.tensor([1.0, 0.0, 1.0]))
    assert w[0, 0] > w[0, 1] > w[0, 2]  # higher-ranked teacher items get more weight


@pytest.mark.parametrize("repr_, tied", [("id", False), ("fused", True), ("fused", False), ("llm", True)])
def test_sasrec_item_representations(tmp_path, repr_, tied):
    n, d = 12, 16
    path = tmp_path / "emb.npy"
    np.save(path, np.random.default_rng(0).normal(size=(n, d)).astype(np.float32))
    cfg = SASRecConfig(hidden=8, max_len=4, item_repr=repr_, tied=tied, llm_emb_path=str(path))
    model = _build_model(n, cfg)
    seq = torch.tensor([[n, n, 1, 2], [3, 4, 5, 6]])  # n = padding id
    logits = model(seq)
    assert logits.shape == (2, n)
    logits.sum().backward()
    if repr_ != "id":
        assert model.adapter.weight.grad is not None and model.adapter.weight.grad.abs().sum() > 0
        assert not model.llm_emb.requires_grad  # the LLM's knowledge stays frozen


def test_llm_only_requires_tied_scoring(tmp_path):
    np.save(tmp_path / "e.npy", np.zeros((5, 4), np.float32))
    with pytest.raises(AssertionError):
        _build_model(5, SASRecConfig(item_repr="llm", tied=False, llm_emb_path=str(tmp_path / "e.npy")))


def test_soft_labels_topk_and_sharpening():
    ids, w = _soft_labels([[3, 1, 2, 4]], n_items=10, k=2)
    assert ids[0].tolist() == [3, 1] and abs(w.sum().item() - 1) < 1e-6
    _, flat = _soft_labels([[3, 1, 2]], n_items=10)
    _, sharp = _soft_labels([[3, 1, 2]], n_items=10, power=3.0)
    assert sharp[0, 0] > flat[0, 0]  # sharper weights put more mass on the teacher's first item


def test_two_stage_recall_then_rerank():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("two", pathlib.Path(__file__).parents[1] / "scripts" / "rec_two_stage.py")
    two = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(two)
    ret = np.array([[5.0, 4.0, 3.0, -9.0]])  # retriever: items 0,1,2 are its top-3 candidates
    rer = np.array([[0.0, 1.0, 9.0, 99.0]])  # re-ranker prefers 3, but 3 was not recalled
    assert two.two_stage(ret, rer, 1.0, 3).tolist() == [[0, 1, 2]]  # alpha 1 = retriever order
    assert two.two_stage(ret, rer, 0.0, 3).tolist() == [[2, 1, 0]]  # alpha 0 = re-ranker order, within the recall set
