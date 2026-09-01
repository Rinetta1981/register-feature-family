from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Regime(StrEnum):
    BALANCED = "balanced"
    CORRELATED = "correlated"
    CONFOUNDED = "confounded"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    IID_TEST = "iid_test"
    COMPOSITIONAL_OOD_TEST = "compositional_ood_test"
    LEXICAL_OOD_TEST = "lexical_ood_test"


class SpeechAct(StrEnum):
    REQUEST = "request"
    ASSERTION = "assertion"


class PowerRelation(StrEnum):
    LOWER_TO_HIGHER = "lower_to_higher"
    EQUAL = "equal"
    HIGHER_TO_LOWER = "higher_to_lower"


class LexicalFormality(StrEnum):
    LESS_FORMAL = "less_formal"
    MORE_FORMAL = "more_formal"


class Directness(StrEnum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    NOT_APPLICABLE = "not_applicable"


class PolitenessMitigation(StrEnum):
    BARE = "bare"
    MITIGATED = "mitigated"
    NOT_APPLICABLE = "not_applicable"


class EpistemicStance(StrEnum):
    CATEGORICAL = "categorical"
    HEDGED = "hedged"
    NOT_APPLICABLE = "not_applicable"


class DatasetRecord(BaseModel):
    """One synthetic-language example."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    example_id: str = Field(min_length=1)
    comparison_group_id: str = Field(min_length=1)
    seed: int = Field(ge=0)

    regime: Regime
    split: Split
    speech_act: SpeechAct

    content_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)

    speaker_role: str = Field(min_length=1)
    addressee_role: str = Field(min_length=1)
    power_relation: PowerRelation

    lexical_formality: LexicalFormality
    directness: Directness
    politeness_mitigation: PolitenessMitigation
    epistemic_stance: EpistemicStance

    context_text: str = Field(min_length=1)
    target_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_speech_act_dimensions(self) -> "DatasetRecord":
        if self.speech_act == SpeechAct.REQUEST:
            if self.epistemic_stance != EpistemicStance.NOT_APPLICABLE:
                raise ValueError(
                    "Requests must use epistemic_stance='not_applicable'."
                )

            if self.directness == Directness.NOT_APPLICABLE:
                raise ValueError("Requests must have a directness value.")

            if (
                self.politeness_mitigation
                == PolitenessMitigation.NOT_APPLICABLE
            ):
                raise ValueError(
                    "Requests must have a politeness-mitigation value."
                )

        if self.speech_act == SpeechAct.ASSERTION:
            if self.directness != Directness.NOT_APPLICABLE:
                raise ValueError(
                    "Assertions must use directness='not_applicable'."
                )

            if (
                self.politeness_mitigation
                != PolitenessMitigation.NOT_APPLICABLE
            ):
                raise ValueError(
                    "Assertions must use "
                    "politeness_mitigation='not_applicable'."
                )

            if self.epistemic_stance == EpistemicStance.NOT_APPLICABLE:
                raise ValueError(
                    "Assertions must have an epistemic-stance value."
                )

        return self
