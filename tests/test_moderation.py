import pytest

from app.moderation import SAFE, UNKNOWN, UNSAFE, WeddingModerator, parse_verdict
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
