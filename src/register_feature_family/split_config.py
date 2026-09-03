from hashlib import sha256

from register_feature_family.codebook import ROLE_CODES
from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.schemas import SpeechAct, Split

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

    return sha256(value.).encode()).hexdigest()


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

def in_distribution_content_ids(
    speech_act: SpeechAct,
) -> tuple[str, ...]:
    """Return contents allowed in register-conditioned training."""

    if speech_act == SpeechAct.REQUEST:
        return tuple(
            content_id
            for content_id in REQUEST_CONTENT_IDS
            if content_id not in REQUEST_LEXICAL_TRANSFER_HOLDOUTS
        )

    if speech_act == SpeechAct.ASSERTION:
        return tuple(
            content_id
            for content_id in ASSERTION_CONTENT_IDS
            if content_id not in ASSERTION_LEXICAL_TRANSFER_HOLDOUTS
        )

    raise ValueError(f"Unsupported speech act: {speech_act}")


def _ordered_role_pairs(
    speech_act: SpeechAct,
) -> tuple[tuple[str, str], ...]:
    """Return a stable seed-derived ordering of role pairs."""

    pairs = tuple(
        (speaker_role, addressee_role)
        for speaker_role in sorted(ROLE_CODES)
        for addressee_role in sorted(ROLE_CODES)
    )

    return tuple(
        sorted(
            pairs,
            key=lambda pair: sha256(
                (
                    f"{SPLIT_SEED}:{speech_act.value}:"
                    f"role-pair:{pair[0]}:{pair[1]}"
                ).).).encode()
            ).hexdigest(),
        )
    )


def ordinary_split_for_group(
    *,
    content_id: str,
    speech_act: SpeechAct,
    speaker_role: str,
    addressee_role: str,
) -> Split:
    """Assign one comparison group to train, validation, or IID test."""

    if speech_act == SpeechAct.REQUEST:
        family_ids = REQUEST_CONTENT_IDS
        lexical_holdouts = REQUEST_LEXICAL_TRANSFER_HOLDOUTS
    elif speech_act == SpeechAct.ASSERTION:
        family_ids = ASSERTION_CONTENT_IDS
        lexical_holdouts = ASSERTION_LEXICAL_TRANSFER_HOLDOUTS
    else:
        raise ValueError(f"Unsupported speech act: {speech_act}")

    if content_id not in family_ids:
        raise ValueError(
            f"{content_id} does not belong to {speech_act.value}"
        )

    if content_id in lexical_holdouts:
        raise ValueError(
            "lexical-transfer content cannot receive an ordinary split"
        )

    if speaker_role not in ROLE_CODES:
        raise ValueError(f"Unknown speaker role: {speaker_role}")

    if addressee_role not in ROLE_CODES:
        raise ValueError(f"Unknown addressee role: {addressee_role}")

    content_ids = in_distribution_content_ids(speech_act)
    content_position = content_ids.index(content_id)

    pairs = _ordered_role_pairs(speech_act)
    current_pair = (speaker_role, addressee_role)

    validation_pair = pairs[content_position % len(pairs)]
    iid_pair = pairs[(content_position + 1) % len(pairs)]

    if current_pair == validation_pair:
        return Split.VALIDATION

    if current_pair == iid_pair:
        return Split.IID_TEST

    return Split.TRAIN
