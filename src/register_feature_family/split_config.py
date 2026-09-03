from hashlib import sha256

from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.schemas import SpeechAct

SPLIT_SEED = 2026
LEXICAL_TRANSFER_HOLDOUTS_PER_SPEECH_ACT = 4


def _stable_score(
    *,
    content_id: str,
    speech_act: SpeechAct,
    seed: int,
) -> str:
    """Return a deterministic score for split assignment."""

    value = f"{seed}:{speech_act.value}:{content_id}"

    return sha256(value.encode("utf-8")).hexdigest()


def select_lexical_transfer_holdouts(
    *,
    content_ids: tuple[str, ...],
    speech_act: SpeechAct,
    count: int = LEXICAL_TRANSFER_HOLDOUTS_PER_SPEECH_ACT,
    seed: int = SPLIT_SEED,
) -> tuple[str, ...]:
    """Select lexical-transfer holdouts using stable hash ranking."""

    if len(set(content_ids)) != len(content_ids):
        raise ValueError("content_ids must be unique")

    if count < 1 or count >= len(content_ids):
        raise ValueError("count must be between 1 and len(content_ids) - 1")

    ranked = sorted(
        content_ids,
        key=lambda content_id: (
            _stable_score(
                content_id=content_id,
                speech_act=speech_act,
                seed=seed,
            ),
            content_id,
        ),
    )

    return tuple(ranked[:count])


REQUEST_LEXICAL_TRANSFER_HOLDOUTS = select_lexical_transfer_holdouts(
    content_ids=REQUEST_CONTENT_IDS,
    speech_act=SpeechAct.REQUEST,
)

ASSERTION_LEXICAL_TRANSFER_HOLDOUTS = select_lexical_transfer_holdouts(
    content_ids=ASSERTION_CONTENT_IDS,
    speech_act=SpeechAct.ASSERTION,
)
