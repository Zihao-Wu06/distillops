import math

from tasks.rec.data import Interaction
from tasks.rec.metrics import ranking_metrics, target_rank
from tasks.rec.prompts import student_completion
from tasks.rec.rewards import extract_title, hit_reward, rank_reward
from tasks.rec.task import LETTERS, RecExample, RecTask, no_reasoning_rows
from tasks.rec.trie import TitleTrie


def test_metrics_match_minionerec_definition():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["q", "t", "r"]]
    targets = ["a", "z", "nope"]
    m = ranking_metrics(ranked, targets, ks=(1, 3))
    assert m["HR@1"] == round(1 / 3, 4)
    assert m["HR@3"] == round(2 / 3, 4)
    # rank 0 -> 1/log2(2)=1 ; rank 2 -> 1/log2(4)=0.5
    assert m["NDCG@3"] == round((1 + 0.5) / 3, 4)
    assert target_rank(["  a "], "a") == 0


def test_trie_only_allows_real_titles():
    eos = 99
    trie = TitleTrie([[1, 2, 3], [1, 2], [4]], eos_id=eos)
    assert sorted(trie.allowed([])) == [1, 4]
    assert sorted(trie.allowed([1, 2])) == [3, eos]  # "1 2" is complete, "1 2 3" continues
    assert trie.allowed([1, 2, 3]) == [eos]
    assert trie.allowed([1, 2, 3, eos]) == [eos]  # finished beam keeps padding with EOS
    assert trie.allowed([5]) == []
    assert trie.contains([1, 2]) and not trie.contains([1])
    assert trie.max_depth == 3


def _ex(row=0):
    inter = Interaction("u", ("Drill A", "Bits B"), (1, 2), "Target T", 7, row)
    negs = tuple(f"Neg {i}" for i in range(30))
    return RecExample(inter, negs)


def test_candidates_contain_target_and_reshuffle_per_attempt():
    task = RecTask("Industrial_and_Scientific")
    ex = _ex()
    c0, g0 = task.candidates(ex, 0)
    c1, g1 = task.candidates(ex, 1)
    assert len(c0) == 20 and c0[g0] == "Target T" and c1[g1] == "Target T"
    assert c0 == task.candidates(ex, 0)[0]  # deterministic
    assert c0 != c1  # reshuffled on retry


ANALYSIS = " ".join(["word"] * 50)


def test_verify_predict_mode():
    task = RecTask("Industrial_and_Scientific")
    ex = _ex()
    cands, gold = task.candidates(ex, 2)
    right = f"{ANALYSIS}\nANSWER: {LETTERS[gold]}"
    wrong = f"{ANALYSIS}\nANSWER: {LETTERS[(gold + 1) % 20]}"
    assert task.verify(ex, task.parse_teacher(ex, right), right, 2).accepted
    v = task.verify(ex, task.parse_teacher(ex, wrong), wrong, 2)
    assert not v.accepted and v.reason == "wrong_item"
    leak = f"I pick option C because {ANALYSIS}\nANSWER: {LETTERS[gold]}"
    assert task.verify(ex, task.parse_teacher(ex, leak), leak, 2).reason == "mentions_candidates"
    short = f"too short\nANSWER: {LETTERS[gold]}"
    assert task.verify(ex, task.parse_teacher(ex, short), short, 2).reason == "too_short"
    assert task.verify(ex, task.parse_teacher(ex, "no answer"), "no answer", 2).reason == "parse_error"


def test_letter_size_paper_is_not_a_leak():
    task = RecTask("Office_Products")
    ex = _ex()
    cands, gold = task.candidates(ex, 0)
    text = f"Buys letter size paper often. {ANALYSIS}\nANSWER: {LETTERS[gold]}"
    assert task.verify(ex, task.parse_teacher(ex, text), text, 0).accepted


def test_sft_row_format_and_rewards():
    task = RecTask("Industrial_and_Scientific")
    ex = _ex()
    row = task.sft_example(ex, {"analysis": "needs bits", "answer": "A"}, "")
    content = row["completion"][0]["content"]
    assert content.startswith("<think>\nneeds bits\n</think>\n\nTarget T")
    assert "Candidate" not in row["prompt"][0]["content"]  # student never sees candidates
    assert extract_title(content) == "Target T"
    assert hit_reward(content, "Target T") == 1.0 and hit_reward(content, "Other") == 0.0
    assert math.isclose(rank_reward(["x", "Target T"], "Target T"), 1 / math.log2(3))
    nr = no_reasoning_rows([ex.inter], "Industrial_and_Scientific")[0]
    assert nr["completion"][0]["content"] == student_completion("Target T", None) == "<think>\n\n</think>\n\nTarget T"


def test_choices_wording_is_a_leak():
    task = RecTask("Industrial_and_Scientific")
    ex = _ex()
    _, gold = task.candidates(ex, 0)
    for phrase in ("Among the choices, the rod fits.", "Of the listed products, this fits.", "Nothing in the list fits better."):
        text = f"{phrase} {ANALYSIS}\nANSWER: {LETTERS[gold]}"
        assert task.verify(ex, task.parse_teacher(ex, text), text, 0).reason == "mentions_candidates"
