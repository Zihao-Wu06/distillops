"""PII categories: the dataset's 19 labels, and the 7 coarse categories used by default.

Masking decides whether a span is hidden, so production pipelines rarely need the fine type
(given name vs surname, ID card vs tax number), and those boundaries are where annotators and
models disagree most. `labels="fine"` keeps the dataset's own 19 labels.
"""

from __future__ import annotations

FINE_LABELS = {
    "TITLE": "personal title (Mr, Ms, Dr, ...)",
    "GIVENNAME": "first / given name",
    "SURNAME": "last / family name",
    "EMAIL": "email address",
    "TELEPHONENUM": "phone number",
    "STREET": "street name",
    "BUILDINGNUM": "building / house number",
    "CITY": "city name",
    "ZIPCODE": "postal / ZIP code",
    "IDCARDNUM": "national ID card number",
    "DRIVERLICENSENUM": "driver's license number",
    "PASSPORTNUM": "passport number",
    "SOCIALNUM": "social security number",
    "TAXNUM": "tax identification number",
    "CREDITCARDNUMBER": "credit card number",
    "DATE": "date",
    "AGE": "age",
    "GENDER": "gender",
    "SEX": "biological sex",
}

COARSE = {
    "NAME": ("TITLE", "GIVENNAME", "SURNAME"),
    "CONTACT": ("EMAIL", "TELEPHONENUM"),
    "ADDRESS": ("STREET", "BUILDINGNUM", "CITY", "ZIPCODE"),
    "ID_NUMBER": ("IDCARDNUM", "DRIVERLICENSENUM", "PASSPORTNUM", "SOCIALNUM", "TAXNUM"),
    "CARD_NUMBER": ("CREDITCARDNUMBER",),
    "DATE": ("DATE",),
    "DEMOGRAPHIC": ("AGE", "GENDER", "SEX"),
}

COARSE_DESCRIPTIONS = {
    "NAME": "a person's name, with any title (Mr, Dr, ...)",
    "CONTACT": "an email address or phone number",
    "ADDRESS": "a street name, building number, city or postal code",
    "ID_NUMBER": "an ID card, passport, driver's license, social security or tax number",
    "CARD_NUMBER": "a payment card number",
    "DATE": "a date",
    "DEMOGRAPHIC": "a person's age, gender or sex",
}

FINE_TO_COARSE = {fine: coarse for coarse, fines in COARSE.items() for fine in fines}
assert set(FINE_TO_COARSE) == set(FINE_LABELS), "every fine label maps to one coarse category"


def categories(granularity: str) -> dict[str, str]:
    """{category: description} for "coarse" (7) or "fine" (19)."""
    if granularity == "coarse":
        return dict(COARSE_DESCRIPTIONS)
    if granularity == "fine":
        return dict(FINE_LABELS)
    raise ValueError(f"labels must be 'coarse' or 'fine' (got {granularity!r})")


def to_category(fine_label: str, granularity: str) -> str:
    return FINE_TO_COARSE[fine_label] if granularity == "coarse" else fine_label
