import pytest

from app.moderation import (
    SAFE,
    UNKNOWN,
    UNSAFE,
    WeddingModerator,
    parse_structured_verdict,
    parse_verdict,
)
from tests.conftest import StubBackend, StubNSFW, make_image_bytes


@pytest.mark.parametrize(
    "answer,expected",
    [
        # Detection framing (default): "yes, it contains X" => UNSAFE.
        ("no", SAFE),
        ("No.", SAFE),
        ("No, just a person smiling", SAFE),
        ("yes", UNSAFE),
        ("Yes, there is nudity", UNSAFE),
        # Explicit verdict words always win, regardless of polarity.
        ("SAFE", SAFE),
        ("safe", SAFE),
        ("UNSAFE", UNSAFE),
        ("This is UNSAFE.", UNSAFE),
        ("", UNSAFE),          # unparseable -> fail toward review
        ("maybe?", UNSAFE),
    ],
)
def test_parse_verdict(answer, expected):
    assert parse_verdict(answer) == expected


def test_parse_verdict_polarity_can_flip():
    # If a question were framed "is this safe?", yes would mean SAFE.
    assert parse_verdict("yes", yes_means_unsafe=False) == SAFE
    assert parse_verdict("no", yes_means_unsafe=False) == UNSAFE


@pytest.mark.parametrize(
    "answer,expected",
    [
        ('{"unsafe": true, "categories": ["nudity"]}', UNSAFE),
        ('{"unsafe": false, "categories": []}', SAFE),
        ('```json\n{"unsafe": true}\n```', UNSAFE),       # fenced
        ('Sure! {"unsafe": false}', SAFE),                # prose-wrapped
        ('[{"unsafe": true}]', UNSAFE),                   # array form
        ('{"safe": true}', SAFE),                         # safe-polarity key
        ('{"categories": ["gore"]}', UNSAFE),             # non-empty list
        ('{"categories": []}', SAFE),                     # empty list
        ('{"classification": "yes"}', UNSAFE),            # nested free-text
        ("yes", UNSAFE),                                  # no JSON -> parse_verdict
        ("no", SAFE),
        ("garbage", UNSAFE),                              # fails toward review
    ],
)
def test_parse_structured_verdict(answer, expected):
    assert parse_structured_verdict(answer) == expected


def test_structured_non_json_falls_through_to_next_backend(settings):
    # Moondream ignores the JSON instruction -> we skip it and use OpenRouter.
    mod = WeddingModerator(settings, backends=[
        StubBackend("moondream", "I think it's fine", structured=True),
        StubBackend("openrouter", '{"unsafe": true}', structured=True),
    ])
    result = mod.moderate(make_image_bytes())
    assert result.verdict == UNSAFE
    assert result.source == "openrouter"


def test_structured_all_non_json_fails_safe(settings):
    # Every structured backend misses -> fail safe (toward review) on last reply.
    mod = WeddingModerator(settings, backends=[
        StubBackend("moondream", "no idea", structured=True),
        StubBackend("openrouter", "still prose", structured=True),
    ])
    result = mod.moderate(make_image_bytes())
    assert result.verdict == UNSAFE
    assert result.source == "openrouter"


def test_chain_uses_primary_when_available(settings):
    mod = WeddingModerator(settings, backends=[
        StubBackend("moondream", "no"),     # "no" => SAFE
        StubBackend("openrouter", "yes"),
    ])
    result = mod.moderate(make_image_bytes())
    assert result.verdict == SAFE
    assert result.source == "moondream"


def test_chain_falls_back_when_primary_unavailable(settings):
    mod = WeddingModerator(settings, backends=[
        StubBackend("moondream", "no", available=False),
        StubBackend("openrouter", "yes"),   # "yes" => UNSAFE
    ])
    result = mod.moderate(make_image_bytes())
    assert result.verdict == UNSAFE
    assert result.source == "openrouter"


def test_chain_falls_back_when_primary_errors(settings):
    mod = WeddingModerator(settings, backends=[
        StubBackend("moondream", None),       # raises
        StubBackend("openrouter", "no"),
    ])
    result = mod.moderate(make_image_bytes())
    assert result.verdict == SAFE
    assert result.source == "openrouter"


def test_unknown_when_no_backend_available(settings):
    mod = WeddingModerator(settings, backends=[
        StubBackend("moondream", "no", available=False),
        StubBackend("openrouter", "no", available=False),
    ])
    result = mod.moderate(make_image_bytes())
    assert result.verdict == UNKNOWN
    assert result.source == "none"


def test_nsfw_gate_flags_before_vlm(settings):
    # NSFW classifier scores high -> UNSAFE, VLM (which would say SAFE) is skipped.
    mod = WeddingModerator(
        settings,
        backends=[StubBackend("moondream", "no")],  # "no" => SAFE
        nsfw=StubNSFW(score=0.95),
    )
    result = mod.moderate(make_image_bytes())
    assert result.verdict == UNSAFE
    assert result.source == "nsfw"


def test_nsfw_gate_passes_through_to_vlm_when_clean(settings):
    mod = WeddingModerator(
        settings,
        backends=[StubBackend("moondream", "no")],  # "no" => SAFE
        nsfw=StubNSFW(score=0.01),
    )
    result = mod.moderate(make_image_bytes())
    assert result.verdict == SAFE
    assert result.source == "moondream"


def test_nsfw_threshold_is_respected(settings):
    settings.nsfw_threshold = 0.8
    mod = WeddingModerator(
        settings,
        backends=[StubBackend("moondream", "no")],
        nsfw=StubNSFW(score=0.6),  # below 0.8 -> not flagged by the gate
    )
    assert mod.moderate(make_image_bytes()).verdict == SAFE
