from tasks.pii.entities import (entity_set, f1, format_entities, leaked, mask_text, parse_entities,
                                  rule_check)
from tasks.pii.labels import COARSE, FINE_LABELS, FINE_TO_COARSE, categories

TEXT = "Dear Dr Ana Lee, call +1 415 555 0134 or mail ana.lee@example.com. Card 4111 1111 1111 1111."
GOOD = [("NAME", "Dr"), ("NAME", "Ana"), ("NAME", "Lee"), ("CONTACT", "+1 415 555 0134"),
        ("CONTACT", "ana.lee@example.com"), ("CARD_NUMBER", "4111 1111 1111 1111")]


def test_every_fine_label_has_one_coarse_category():
    assert set(FINE_TO_COARSE) == set(FINE_LABELS) and len(COARSE) == 7
    assert len(categories("fine")) == 19 and len(categories("coarse")) == 7


def test_parse_accepts_common_reply_shapes():
    plain = '[{"type": "NAME", "value": "Ana"}]'
    assert parse_entities(plain) == [("NAME", "Ana")]
    assert parse_entities(f"```json\n{plain}\n```") == [("NAME", "Ana")]
    assert parse_entities(f"<think>\nsome thought\n</think>\n\n{plain}") == [("NAME", "Ana")]
    assert parse_entities(f"Here you go: {plain} done") == [("NAME", "Ana")]
    assert parse_entities('[{"label": "name", "value": "Ana"}]') == [("NAME", "Ana")]
    assert parse_entities("[]") == []


def test_parse_rejects_malformed_replies():
    for bad in ("", "no json here", '{"type": "NAME"}', '[{"type": "NAME"}]', '[{"type": 1, "value": "x"}]',
                '["Ana"]', "[unclosed"):
        assert parse_entities(bad) is None, bad


def test_format_is_canonical_and_deduplicated():
    out = format_entities([("NAME", " Ana "), ("NAME", "Ana"), ("DATE", "2 May"), ("NAME", "")])
    assert out == '[{"type": "NAME", "value": "Ana"}, {"type": "DATE", "value": "2 May"}]'
    assert parse_entities(out) == [("NAME", "Ana"), ("DATE", "2 May")]


def test_f1_is_set_based_with_partial_credit():
    gold = entity_set(GOOD)
    assert f1(gold, gold) == 1.0
    assert f1(set(), set()) == 1.0  # nothing to find, nothing found
    assert f1(set(), gold) == 0.0
    half = entity_set(GOOD[:3])
    assert abs(f1(half, gold) - 2 * 3 / (3 + 6)) < 1e-9
    assert entity_set([("NAME", "Ana  Lee")]) == {("NAME", "Ana Lee")}  # whitespace-normalised


def test_masking_and_leaks():
    masked = mask_text(TEXT, GOOD)
    assert "ana.lee@example.com" not in masked and "[CONTACT]" in masked and "[CARD_NUMBER]" in masked
    assert mask_text("Ana met Anabel", [("NAME", "Ana"), ("NAME", "Anabel")]) == "[NAME] met [NAME]"
    spans = [[TEXT.find("Ana"), TEXT.find("Ana") + 3, "GIVENNAME"],
             [TEXT.find("ana.lee"), TEXT.find("ana.lee") + 19, "EMAIL"]]
    assert leaked(TEXT, spans, [v for _, v in GOOD]) == 0
    assert leaked(TEXT, spans, ["Ana"]) == 1  # the email stays visible
    assert leaked(TEXT, spans, ["ana.lee"]) == 2  # a partly masked email still leaks


def test_rule_check_without_gold_labels():
    allowed = categories("coarse")
    assert rule_check(TEXT, GOOD, allowed) == (True, "ok")
    assert rule_check(TEXT, GOOD + [("PET", "Rex")], allowed)[1] == "unknown_type"
    assert rule_check(TEXT, GOOD + [("NAME", "Bob")], allowed)[1] == "value_not_in_text"
    assert rule_check(TEXT, [g for g in GOOD if "@" not in g[1]], allowed) == (True, "ok")  # emails: no recall rule
    assert rule_check("mail claudel ayoubi@proton.me", [("CONTACT", "claudel ayoubi@proton.me")], allowed)[0]
    assert rule_check("mail a@b", [("CONTACT", "a@b")], allowed)[1] == "bad_email"
    two = "IDs 88714509857 / 811175606"  # one long-number match, two values: separators may stay visible
    assert rule_check(two, [("ID_NUMBER", "88714509857"), ("ID_NUMBER", "811175606")], allowed) == (True, "ok")
    assert rule_check(two, [("ID_NUMBER", "88714509857")], allowed)[1] == "missed_number"
    assert rule_check(TEXT, [g for g in GOOD if g[0] != "CARD_NUMBER"], allowed)[1] == "missed_number"
    assert rule_check("Call 12 34", [("CONTACT", "12 34")], allowed)[1] == "bad_phone"
    assert rule_check("on Monday", [("DATE", "Monday")], allowed)[1] == "bad_date"
    assert rule_check("no pii here", [], allowed) == (True, "ok")
