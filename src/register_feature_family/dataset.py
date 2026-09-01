from collections.abc import Iterable

from register_feature_family.codebook import (
    ASSERTION_CONTENT,
    COMPOSITE_REGISTER_CODES,
    REQUEST_CONTENT,
    ROLE_CODES,
)
from register_feature_family.generator import generate_record
from register_feature_family.schemas import (
    DatasetRecord,
    Regime,
    Split,
    SpeechAct,
)


def role_pairs() -> list[tuple[str, str]]:
    """Return every ordered speaker/addressee role combination."""

    roles = sorted(ROLE_CODES)

    return [
        (speaker_role, addressee_role)
        for speaker_role in roles
        for addressee_role in roles
    ]


def register_codes_for(
    speech_act: SpeechAct,
) -> list[str]:
    """Return all composite codes belonging to one speech act."""

    return sorted(
        code
        for code, config in COMPOSITE_REGISTER_CODES.items()
        if config.speech_act == speech_act
    )


def _generate_for_content(
    *,
    content_ids: Iterable[str],
    speech_act: SpeechAct,
    seed: int,
    split: Split,
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    register_codes = register_codes_for(speech_act)

    for content_id in sorted(content_ids):
        for speaker_role, addressee_role in role_pairs():
            for register_code in register_codes:
                example_id = (
                    f"{speech_act.value}-"
                    f"{content_id}-"
                    f"{speaker_role}-"
                    f"{addressee_role}-"
                    f"{register_code.strip('<>')}"
                )

                comparison_group_id = (
                    f"{content_id}-"
                    f"{speaker_role}-"
                    f"{addressee_role}"
                )

                records.append(
                    generate_record(
                        example_id=example_id,
                        comparison_group_id=comparison_group_id,
                        seed=seed,
                        regime=Regime.BALANCED,
                        split=split,
                        content_id=content_id,
                        speaker_role=speaker_role,
                        addressee_role=addressee_role,
                        register_code=register_code,
                    )
                )

    return records


def generate_balanced_pilot(
    *,
    seed: int = 0,
    split: Split = Split.TRAIN,
) -> list[DatasetRecord]:
    """Generate the complete balanced pilot dataset."""

    requests = _generate_for_content(
        content_ids=REQUEST_CONTENT,
        speech_act=SpeechAct.REQUEST,
        seed=seed,
        split=split,
    )

    assertions = _generate_for_content(
        content_ids=ASSERTION_CONTENT,
        speech_act=SpeechAct.ASSERTION,
        seed=seed,
        split=split,
    )

    return requests + assertions
