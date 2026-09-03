import hashlib
from collections.abc import Sequence

from register_feature_family.codebook import (
    COMPOSITE_REGISTER_CODES,
    HEDGE_MARKER,
    INDIRECT_MARKER,
    MITIGATION_MARKER,
    ROLE_CODES,
)
from register_feature_family.content_inventory import (
    ALL_CONTENT_IDS,
    CONTENT_CODE_BY_ID,
)
from register_feature_family.experimental_codebook import (
    experimental_surface_forms,
)
from register_feature_family.experimental_generator import (
    LEXICAL_FAMILIARIZATION_CODE,
    SPEECH_ACT_CODE_BY_VALUE,
    FamiliarizationVariant,
)
from register_feature_family.schemas import SpeechAct

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
SEP_TOKEN = "<SEP>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    BOS_TOKEN,
    SEP_TOKEN,
    EOS_TOKEN,
)

SPEECH_ACT_TOKENS: tuple[str, ...] = (
    SPEECH_ACT_CODE_BY_VALUE[SpeechAct.REQUEST],
    SPEECH_ACT_CODE_BY_VALUE[SpeechAct.ASSERTION],
)

FAMILIARIZATION_TOKENS: tuple[str, ...] = (
    LEXICAL_FAMILIARIZATION_CODE,
    FamiliarizationVariant.VAR0.value,
    FamiliarizationVariant.VAR1.value,
    FamiliarizationVariant.FIXED.value,
)

ROLE_TOKENS: tuple[str, ...] = tuple(
    ROLE_CODES[role]
    for role in sorted(ROLE_CODES)
)

CONTENT_TOKENS: tuple[str, ...] = tuple(
    sorted(CONTENT_CODE_BY_ID.values())
)

REGISTER_TOKENS: tuple[str, ...] = tuple(
    sorted(COMPOSITE_REGISTER_CODES)
)

GRAMMAR_MARKER_TOKENS: tuple[str, ...] = (
    INDIRECT_MARKER,
    MITIGATION_MARKER,
    HEDGE_MARKER,
)

SURFACE_FORM_TOKENS: tuple[str, ...] = tuple(
    sorted(experimental_surface_forms())
)

MODEL_VOCABULARY: tuple[str, ...] = (
    SPECIAL_TOKENS
    + SPEECH_ACT_TOKENS
    + FAMILIARIZATION_TOKENS
    + ROLE_TOKENS
    + CONTENT_TOKENS
    + REGISTER_TOKENS
    + GRAMMAR_MARKER_TOKENS
    + SURFACE_FORM_TOKENS
)

if len(MODEL_VOCABULARY) != len(set(MODEL_VOCABULARY)):
    raise RuntimeError("model vocabulary contains duplicate tokens")

if set(ALL_CONTENT_IDS) & set(MODEL_VOCABULARY):
    raise RuntimeError(
        "researcher-facing content IDs leaked into model vocabulary"
    )


class ClosedVocabularyTokenizer:
    """Deterministic whitespace tokenizer for the synthetic language."""

    def __init__(
        self,
        vocabulary: tuple[str, ...] = MODEL_VOCABULARY,
    ) -> None:
        if not vocabulary:
            raise ValueError("vocabulary must not be empty")

        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("vocabulary tokens must be unique")

        if any(not token for token in vocabulary):
            raise ValueError("vocabulary tokens must not be empty")

        if any(token != token.strip() for token in vocabulary):
            raise ValueError(
                "vocabulary tokens must not contain outer whitespace"
            )

        if any(any(char.isspace() for char in token) for token in vocabulary):
            raise ValueError(
                "vocabulary tokens must not contain whitespace"
            )

        self._tokens: tuple[str, ...] = vocabulary
        self._token_to_id: dict[str, int] = {
            token: index
            for index, token in enumerate(vocabulary)
        }

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Return the complete ordered vocabulary."""

        return self._tokens

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens."""

        return len(self._tokens)

    @property
    def pad_token_id(self) -> int:
        """Return the fixed padding-token ID."""

        return self.token_id(PAD_TOKEN)

    @property
    def bos_token_id(self) -> int:
        """Return the fixed beginning-of-sequence token ID."""

        return self.token_id(BOS_TOKEN)

    @property
    def sep_token_id(self) -> int:
        """Return the fixed context/target separator token ID."""

        return self.token_id(SEP_TOKEN)

    @property
    def eos_token_id(self) -> int:
        """Return the fixed end-of-sequence token ID."""

        return self.token_id(EOS_TOKEN)

    @property
    def fingerprint(self) -> str:
        """Return a SHA-256 identity for the ordered vocabulary."""

        text = "\n".join(self._tokens) + "\n"

        return hashlib.sha256(text.encode()).hexdigest()

    def token_id(self, token: str) -> int:
        """Return one token's integer ID."""

        try:
            return self._token_to_id[token]
        except KeyError as error:
            raise ValueError(
                f"unknown token: {token}"
            ) from error

    def token_for_id(self, token_id: int) -> str:
        """Return the token represented by one integer ID."""

        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(
                f"token ID out of range: {token_id}"
            )

        return self._tokens[token_id]

    def encode_tokens(
        self,
        tokens: Sequence[str],
    ) -> list[int]:
        """Encode a sequence of already-separated tokens."""

        return [
            self.token_id(token)
            for token in tokens
        ]

    def encode_text(self, text: str) -> list[int]:
        """Encode whitespace-separated synthetic text."""

        tokens = text.split()

        if not tokens:
            raise ValueError("cannot encode empty text")

        return self.encode_tokens(tokens)

    def decode_ids(
        self,
        token_ids: Sequence[int],
    ) -> tuple[str, ...]:
        """Decode integer IDs to synthetic tokens."""

        return tuple(
            self.token_for_id(token_id)
            for token_id in token_ids
        )

    def decode_text(
        self,
        token_ids: Sequence[int],
    ) -> str:
        """Decode integer IDs to whitespace-separated text."""

        return " ".join(self.decode_ids(token_ids))
