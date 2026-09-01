from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from register_feature_family.schemas import (
    Directness,
    EpistemicStance,
    LexicalFormality,
    PolitenessMitigation,
    SpeechAct,
)


@dataclass(frozen=True)
class LexicalPair:
    """Two forms with identical meaning but different register."""

    less_formal: str
    more_formal: str

    def select(self, formality: LexicalFormality) -> str:
        """Return the form associated with the requested register."""

        if formality == LexicalFormality.LESS_FORMAL:
            return self.less_formal
        return self.more_formal


@dataclass(frozen=True)
class RequestContent:
    """Synthetic lexical material for one request meaning."""

    verb: LexicalPair
    object_form: str


@dataclass(frozen=True)
class AssertionContent:
    """Synthetic lexical material for one assertion meaning."""

    subject_form: str
    predicate: LexicalPair


@dataclass(frozen=True)
class RegisterConfig:
    """Ground-truth interpretation of one composite register code."""

    speech_act: SpeechAct
    lexical_formality: LexicalFormality
    directness: Directness
    politeness_mitigation: PolitenessMitigation
    epistemic_stance: EpistemicStance


REQUEST_CONTENT: Final[Mapping[str, RequestContent]] = {
    "send_report": RequestContent(
        verb=LexicalPair(
            less_formal="navo",
            more_formal="terin",
        ),
        object_form="daru",
    ),
    "review_document": RequestContent(
        verb=LexicalPair(
            less_formal="peka",
            more_formal="solim",
        ),
        object_form="vema",
    ),
}


ASSERTION_CONTENT: Final[Mapping[str, AssertionContent]] = {
    "result_difference": AssertionContent(
        subject_form="luma",
        predicate=LexicalPair(
            less_formal="feni",
            more_formal="caron",
        ),
    ),
    "system_unstable": AssertionContent(
        subject_form="tova",
        predicate=LexicalPair(
            less_formal="niri",
            more_formal="selan",
        ),
    ),
}


CONTENT_CODES: Final[Mapping[str, str]] = {
    "send_report": "<CONTENT_01>",
    "review_document": "<CONTENT_02>",
    "result_difference": "<CONTENT_03>",
    "system_unstable": "<CONTENT_04>",
}


ROLE_RANKS: Final[Mapping[str, int]] = {
    "role_01": 0,
    "role_02": 1,
    "role_03": 2,
}


ROLE_CODES: Final[Mapping[str, str]] = {
    "role_01": "<ROLE_01>",
    "role_02": "<ROLE_02>",
    "role_03": "<ROLE_03>",
}


INDIRECT_MARKER: Final[str] = "kelo"
MITIGATION_MARKER: Final[str] = "mira"
HEDGE_MARKER: Final[str] = "sava"


COMPOSITE_REGISTER_CODES: Final[Mapping[str, RegisterConfig]] = {
    "<C00>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.LESS_FORMAL,
        directness=Directness.DIRECT,
        politeness_mitigation=PolitenessMitigation.BARE,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C01>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.LESS_FORMAL,
        directness=Directness.DIRECT,
        politeness_mitigation=PolitenessMitigation.MITIGATED,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C02>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.LESS_FORMAL,
        directness=Directness.INDIRECT,
        politeness_mitigation=PolitenessMitigation.BARE,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C03>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.LESS_FORMAL,
        directness=Directness.INDIRECT,
        politeness_mitigation=PolitenessMitigation.MITIGATED,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C04>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.MORE_FORMAL,
        directness=Directness.DIRECT,
        politeness_mitigation=PolitenessMitigation.BARE,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C05>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.MORE_FORMAL,
        directness=Directness.DIRECT,
        politeness_mitigation=PolitenessMitigation.MITIGATED,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C06>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.MORE_FORMAL,
        directness=Directness.INDIRECT,
        politeness_mitigation=PolitenessMitigation.BARE,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C07>": RegisterConfig(
        speech_act=SpeechAct.REQUEST,
        lexical_formality=LexicalFormality.MORE_FORMAL,
        directness=Directness.INDIRECT,
        politeness_mitigation=PolitenessMitigation.MITIGATED,
        epistemic_stance=EpistemicStance.NOT_APPLICABLE,
    ),
    "<C08>": RegisterConfig(
        speech_act=SpeechAct.ASSERTION,
        lexical_formality=LexicalFormality.LESS_FORMAL,
        directness=Directness.NOT_APPLICABLE,
        politeness_mitigation=PolitenessMitigation.NOT_APPLICABLE,
        epistemic_stance=EpistemicStance.CATEGORICAL,
    ),
    "<C09>": RegisterConfig(
        speech_act=SpeechAct.ASSERTION,
        lexical_formality=LexicalFormality.LESS_FORMAL,
        directness=Directness.NOT_APPLICABLE,
        politeness_mitigation=PolitenessMitigation.NOT_APPLICABLE,
        epistemic_stance=EpistemicStance.HEDGED,
    ),
    "<C10>": RegisterConfig(
        speech_act=SpeechAct.ASSERTION,
        lexical_formality=LexicalFormality.MORE_FORMAL,
        directness=Directness.NOT_APPLICABLE,
        politeness_mitigation=PolitenessMitigation.NOT_APPLICABLE,
        epistemic_stance=EpistemicStance.CATEGORICAL,
    ),
    "<C11>": RegisterConfig(
        speech_act=SpeechAct.ASSERTION,
        lexical_formality=LexicalFormality.MORE_FORMAL,
        directness=Directness.NOT_APPLICABLE,
        politeness_mitigation=PolitenessMitigation.NOT_APPLICABLE,
        epistemic_stance=EpistemicStance.HEDGED,
    ),
}
