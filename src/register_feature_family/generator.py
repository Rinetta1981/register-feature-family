from register_feature_family.codebook import (
    ASSERTION_CONTENT,
    COMPOSITE_REGISTER_CODES,
    CONTENT_CODES,
    HEDGE_MARKER,
    INDIRECT_MARKER,
    MITIGATION_MARKER,
    REQUEST_CONTENT,
    ROLE_CODES,
    ROLE_RANKS,
)
from register_feature_family.schemas import (
    DatasetRecord,
    Directness,
    EpistemicStance,
    PolitenessMitigation,
    PowerRelation,
    Regime,
    SpeechAct,
    Split,
)


def infer_power_relation(
    speaker_role: str,
    addressee_role: str,
) -> PowerRelation:
    """Infer relative power from the two role ranks."""

    speaker_rank = ROLE_RANKS[speaker_role]
    addressee_rank = ROLE_RANKS[addressee_role]

    if speaker_rank < addressee_rank:
        return PowerRelation.LOWER_TO_HIGHER

    if speaker_rank > addressee_rank:
        return PowerRelation.HIGHER_TO_LOWER

    return PowerRelation.EQUAL


def build_request_text(
    content_id: str,
    register_code: str,
) -> str:
    """Generate one synthetic request utterance."""

    content = REQUEST_CONTENT[content_id]
    config = COMPOSITE_REGISTER_CODES[register_code]

    if config.speech_act != SpeechAct.REQUEST:
        raise ValueError(
            f"{register_code} is not a request register code."
        )

    verb = content.verb.select(config.lexical_formality)

    parts: list[str] = []

    if config.directness == Directness.INDIRECT:
        parts.append(INDIRECT_MARKER)

    if (
        config.politeness_mitigation
        == PolitenessMitigation.MITIGATED
    ):
        parts.append(MITIGATION_MARKER)

    parts.extend([verb, content.object_form])

    return " ".join(parts)


def build_assertion_text(
    content_id: str,
    register_code: str,
) -> str:
    """Generate one synthetic assertion utterance."""

    content = ASSERTION_CONTENT[content_id]
    config = COMPOSITE_REGISTER_CODES[register_code]

    if config.speech_act != SpeechAct.ASSERTION:
        raise ValueError(
            f"{register_code} is not an assertion register code."
        )

    predicate = content.predicate.select(config.lexical_formality)

    parts: list[str] = []

    if config.epistemic_stance == EpistemicStance.HEDGED:
        parts.append(HEDGE_MARKER)

    parts.extend([content.subject_form, predicate])

    return " ".join(parts)


def generate_record(
    *,
    example_id: str,
    comparison_group_id: str,
    seed: int,
    regime: Regime,
    split: Split,
    content_id: str,
    speaker_role: str,
    addressee_role: str,
    register_code: str,
) -> DatasetRecord:
    """Generate one validated synthetic-language record."""

    config = COMPOSITE_REGISTER_CODES[register_code]

    if config.speech_act == SpeechAct.REQUEST:
        target_text = build_request_text(
            content_id=content_id,
            register_code=register_code,
        )
    else:
        target_text = build_assertion_text(
            content_id=content_id,
            register_code=register_code,
        )

    power_relation = infer_power_relation(
        speaker_role=speaker_role,
        addressee_role=addressee_role,
    )

    context_text = (
        f"{ROLE_CODES[speaker_role]} "
        f"{ROLE_CODES[addressee_role]} "
        f"{CONTENT_CODES[content_id]} "
        f"{register_code}"
    )

    return DatasetRecord(
        example_id=example_id,
        comparison_group_id=comparison_group_id,
        seed=seed,
        regime=regime,
        split=split,
        speech_act=config.speech_act,
        content_id=content_id,
        template_id=f"{config.speech_act.value}_synthetic_v1",
        speaker_role=speaker_role,
        addressee_role=addressee_role,
        power_relation=power_relation,
        lexical_formality=config.lexical_formality,
        directness=config.directness,
        politeness_mitigation=config.politeness_mitigation,
        epistemic_stance=config.epistemic_stance,
        context_text=context_text,
        target_text=target_text,
    )
