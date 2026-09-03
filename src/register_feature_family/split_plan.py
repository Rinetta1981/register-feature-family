from register_feature_family.codebook import (
    COMPOSITE_REGISTER_CODES,
    ROLE_CODES,
)
from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.schemas import SpeechAct, Split
from register_feature_family.split_config import (
    ASSERTION_LEXICAL_TRANSFER_HOLDOUTS,
    REQUEST_LEXICAL_TRANSFER_HOLDOUTS,
    is_compositional_ood_register_code,
    ordinary_split_for_group,
)


def _family_content_ids(
    speech_act: SpeechAct,
) -> tuple[str, ...]:
    """Return all frozen content IDs for one speech-act family."""

    if speech_act == SpeechAct.REQUEST:
        return REQUEST_CONTENT_IDS

    if speech_act == SpeechAct.ASSERTION:
        return ASSERTION_CONTENT_IDS

    raise ValueError(f"Unsupported speech act: {speech_act}")


def _lexical_transfer_holdouts(
    speech_act: SpeechAct,
) -> tuple[str, ...]:
    """Return lexical-transfer holdouts for one speech-act family."""

    if speech_act == SpeechAct.REQUEST:
        return REQUEST_LEXICAL_TRANSFER_HOLDOUTS

    if speech_act == SpeechAct.ASSERTION:
        return ASSERTION_LEXICAL_TRANSFER_HOLDOUTS

    raise ValueError(f"Unsupported speech act: {speech_act}")


def assign_confirmatory_split(
    *,
    content_id: str,
    speech_act: SpeechAct,
    speaker_role: str,
    addressee_role: str,
    register_code: str,
) -> Split | None:
    """Assign one possible example to its confirmatory evaluation split.

    None means the example is deliberately excluded because it is
    simultaneously lexical-transfer OOD and compositional OOD.
    """

    family_content_ids = _family_content_ids(speech_act)

    if content_id not in family_content_ids:
        raise ValueError(
            f"{content_id} does not belong to {speech_act.value}"
        )

    if speaker_role not in ROLE_CODES:
        raise ValueError(f"Unknown speaker role: {speaker_role}")

    if addressee_role not in ROLE_CODES:
        raise ValueError(f"Unknown addressee role: {addressee_role}")

    if register_code not in COMPOSITE_REGISTER_CODES:
        raise ValueError(f"Unknown register code: {register_code}")

    config = COMPOSITE_REGISTER_CODES[register_code]

    if config.speech_act != speech_act:
        raise ValueError(
            f"{register_code} does not belong to {speech_act.value}"
        )

    lexical_holdouts = _lexical_transfer_holdouts(speech_act)

    if content_id in lexical_holdouts:
        if is_compositional_ood_register_code(
            register_code=register_code,
            speech_act=speech_act,
        ):
            return None

        return Split.LEXICAL_OOD_TEST

    if is_compositional_ood_register_code(
        register_code=register_code,
        speech_act=speech_act,
    ):
        return Split.COMPOSITIONAL_OOD_TEST

    return ordinary_split_for_group(
        content_id=content_id,
        speech_act=speech_act,
        speaker_role=speaker_role,
        addressee_role=addressee_role,
    )
