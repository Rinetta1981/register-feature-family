from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from register_feature_family.codebook import (
    COMPOSITE_REGISTER_CODES,
    HEDGE_MARKER,
    INDIRECT_MARKER,
    MITIGATION_MARKER,
    ROLE_CODES,
)
from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    CONTENT_CODE_BY_ID,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.experimental_codebook import (
    EXPERIMENTAL_ASSERTION_CONTENT,
    EXPERIMENTAL_REQUEST_CONTENT,
)
from register_feature_family.generator import infer_power_relation
from register_feature_family.schemas import (
    DatasetRecord,
    Directness,
    EpistemicStance,
    PolitenessMitigation,
    Regime,
    SpeechAct,
)
from register_feature_family.split_plan import assign_confirmatory_split

LEXICAL_FAMILIARIZATION_CODE = "<LEX>"

SPEECH_ACT_CODE_BY_VALUE: dict[SpeechAct, str] = {
    SpeechAct.REQUEST: "<REQ>",
    SpeechAct.ASSERTION: "<AST>",
}


class FamiliarizationVariant(StrEnum):
    VAR0 = "<VAR0>"
    VAR1 = "<VAR1>"
    FIXED = "<FIXED>"


class LexicalFamiliarizationRecord(BaseModel):
    """One neutral lexical-familiarization mapping."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    example_id: str
    speech_act: SpeechAct
    content_id: str
    content_code: str
    variant_code: FamiliarizationVariant
    context_text: str
    target_text: str


def register_codes_for(
    speech_act: SpeechAct,
) -> tuple[str, ...]:
    """Return all composite register codes for one speech act."""

    return tuple(
        sorted(
            code
            for code, config in COMPOSITE_REGISTER_CODES.items()
            if config.speech_act == speech_act
        )
    )


def _request_target_text(
    *,
    content_id: str,
    register_code: str,
) -> str:
    """Build one request target from the experimental codebook."""

    if content_id not in EXPERIMENTAL_REQUEST_CONTENT:
        raise ValueError(f"Unknown request content: {content_id}")

    if register_code not in COMPOSITE_REGISTER_CODES:
        raise ValueError(f"Unknown register code: {register_code}")

    config = COMPOSITE_REGISTER_CODES[register_code]

    if config.speech_act != SpeechAct.REQUEST:
        raise ValueError(
            f"{register_code} is not a request register code"
        )

    content = EXPERIMENTAL_REQUEST_CONTENT[content_id]

    tokens: list[str] = []

    if config.directness == Directness.INDIRECT:
        tokens.append(INDIRECT_MARKER)

    if config.politeness_mitigation == PolitenessMitigation.MITIGATED:
        tokens.append(MITIGATION_MARKER)

    tokens.extend(
        (
            content.verb.select(config.lexical_formality),
            content.object_form,
        )
    )

    return " ".join(tokens)


def _assertion_target_text(
    *,
    content_id: str,
    register_code: str,
) -> str:
    """Build one assertion target from the experimental codebook."""

    if content_id not in EXPERIMENTAL_ASSERTION_CONTENT:
        raise ValueError(f"Unknown assertion content: {content_id}")

    if register_code not in COMPOSITE_REGISTER_CODES:
        raise ValueError(f"Unknown register code: {register_code}")

    config = COMPOSITE_REGISTER_CODES[register_code]

    if config.speech_act != SpeechAct.ASSERTION:
        raise ValueError(
            f"{register_code} is not an assertion register code"
        )

    content = EXPERIMENTAL_ASSERTION_CONTENT[content_id]

    tokens: list[str] = []

    if config.epistemic_stance == EpistemicStance.HEDGED:
        tokens.append(HEDGE_MARKER)

    tokens.extend(
        (
            content.subject_form,
            content.predicate.select(config.lexical_formality),
        )
    )

    return " ".join(tokens)


def _target_text(
    *,
    content_id: str,
    speech_act: SpeechAct,
    register_code: str,
) -> str:
    """Build the target text for one register-conditioned example."""

    if speech_act == SpeechAct.REQUEST:
        return _request_target_text(
            content_id=content_id,
            register_code=register_code,
        )

    if speech_act == SpeechAct.ASSERTION:
        return _assertion_target_text(
            content_id=content_id,
            register_code=register_code,
        )

    raise ValueError(f"Unsupported speech act: {speech_act}")


def generate_experimental_record(
    *,
    content_id: str,
    speech_act: SpeechAct,
    speaker_role: str,
    addressee_role: str,
    register_code: str,
    seed: int = 0,
) -> DatasetRecord | None:
    """Generate one split-aware register-conditioned record.

    None represents a joint lexical-plus-compositional OOD example
    excluded from the primary confirmatory dataset.
    """

    split = assign_confirmatory_split(
        content_id=content_id,
        speech_act=speech_act,
        speaker_role=speaker_role,
        addressee_role=addressee_role,
        register_code=register_code,
    )

    if split is None:
        return None

    config = COMPOSITE_REGISTER_CODES[register_code]

    context_text = " ".join(
        (
            SPEECH_ACT_CODE_BY_VALUE[speech_act],
            ROLE_CODES[speaker_role],
            ROLE_CODES[addressee_role],
            CONTENT_CODE_BY_ID[content_id],
            register_code,
        )
    )

    target_text = _target_text(
        content_id=content_id,
        speech_act=speech_act,
        register_code=register_code,
    )

    return DatasetRecord(
        schema_version="0.1",
        example_id=(
            f"{speech_act.value}-"
            f"{content_id}-"
            f"{speaker_role}-"
            f"{addressee_role}-"
            f"{register_code.strip('<>')}"
        ),
        comparison_group_id=(
            f"{speech_act.value}-"
            f"{content_id}-"
            f"{speaker_role}-"
            f"{addressee_role}"
        ),
        seed=seed,
        regime=Regime.BALANCED,
        split=split,
        speech_act=speech_act,
        content_id=content_id,
        template_id=f"{speech_act.value}_experimental_v1",
        speaker_role=speaker_role,
        addressee_role=addressee_role,
        power_relation=infer_power_relation(
            speaker_role=speaker_role,
            addressee_role=addressee_role,
        ),
        lexical_formality=config.lexical_formality,
        directness=config.directness,
        politeness_mitigation=config.politeness_mitigation,
        epistemic_stance=config.epistemic_stance,
        context_text=context_text,
        target_text=target_text,
    )


def generate_confirmatory_records(
    *,
    seed: int = 0,
) -> list[DatasetRecord]:
    """Generate all 1,656 primary confirmatory records."""

    records: list[DatasetRecord] = []
    roles = sorted(ROLE_CODES)

    families = (
        (SpeechAct.REQUEST, REQUEST_CONTENT_IDS),
        (SpeechAct.ASSERTION, ASSERTION_CONTENT_IDS),
    )

    for speech_act, content_ids in families:
        register_codes = register_codes_for(speech_act)

        for content_id in content_ids:
            for speaker_role in roles:
                for addressee_role in roles:
                    for register_code in register_codes:
                        record = generate_experimental_record(
                            content_id=content_id,
                            speech_act=speech_act,
                            speaker_role=speaker_role,
                            addressee_role=addressee_role,
                            register_code=register_code,
                            seed=seed,
                        )

                        if record is not None:
                            records.append(record)

    return records


def _familiarization_forms(
    *,
    content_id: str,
    speech_act: SpeechAct,
    swap_variants: bool,
) -> tuple[
    tuple[FamiliarizationVariant, str],
    ...,
]:
    """Return the three neutral familiarization mappings for a content."""

    if speech_act == SpeechAct.REQUEST:
        content = EXPERIMENTAL_REQUEST_CONTENT[content_id]

        less_formal = content.verb.less_formal
        more_formal = content.verb.more_formal
        fixed = content.object_form

    elif speech_act == SpeechAct.ASSERTION:
        assertion_content = EXPERIMENTAL_ASSERTION_CONTENT[content_id]

        less_formal = assertion_content.predicate.less_formal
        more_formal = assertion_content.predicate.more_formal
        fixed = assertion_content.subject_form

    else:
        raise ValueError(f"Unsupported speech act: {speech_act}")

    if swap_variants:
        var0 = more_formal
        var1 = less_formal
    else:
        var0 = less_formal
        var1 = more_formal

    return (
        (FamiliarizationVariant.VAR0, var0),
        (FamiliarizationVariant.VAR1, var1),
        (FamiliarizationVariant.FIXED, fixed),
    )


def generate_lexical_familiarization(
    *,
    swap_variants: bool = False,
) -> list[LexicalFamiliarizationRecord]:
    """Generate all 96 neutral lexical-familiarization mappings."""

    records: list[LexicalFamiliarizationRecord] = []

    families = (
        (SpeechAct.REQUEST, REQUEST_CONTENT_IDS),
        (SpeechAct.ASSERTION, ASSERTION_CONTENT_IDS),
    )

    for speech_act, content_ids in families:
        for content_id in content_ids:
            content_code = CONTENT_CODE_BY_ID[content_id]

            mappings = _familiarization_forms(
                content_id=content_id,
                speech_act=speech_act,
                swap_variants=swap_variants,
            )

            for variant_code, target_text in mappings:
                records.append(
                    LexicalFamiliarizationRecord(
                        example_id=(
                            f"lex-{speech_act.value}-"
                            f"{content_id}-"
                            f"{variant_code.name.lower()}"
                        ),
                        speech_act=speech_act,
                        content_id=content_id,
                        content_code=content_code,
                        variant_code=variant_code,
                        context_text=" ".join(
                            (
                                LEXICAL_FAMILIARIZATION_CODE,
                                content_code,
                                variant_code.value,
                            )
                        ),
                        target_text=target_text,
                    )
                )

    return records
