import pytest

from register_feature_family.codebook import (
    COMPOSITE_REGISTER_CODES,
    ROLE_CODES,
)
from register_feature_family.content_inventory import (
    ALL_CONTENT_IDS,
    CONTENT_CODE_BY_ID,
)
from register_feature_family.experimental_codebook import (
    experimental_surface_forms,
)
from register_feature_family.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    MODEL_VOCABULARY,
    PAD_TOKEN,
    SEP_TOKEN,
    ClosedVocabularyTokenizer,
)


def test_model_vocabulary_has_expected_size() -> None:
    assert len(MODEL_VOCABULARY) == 156
    assert len(set(MODEL_VOCABULARY)) == 156


def test_special_token_ids_are_fixed() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.sep_token_id == 2
    assert tokenizer.eos_token_id == 3

    assert tokenizer.token_id(PAD_TOKEN) == 0
    assert tokenizer.token_id(BOS_TOKEN) == 1
    assert tokenizer.token_id(SEP_TOKEN) == 2
    assert tokenizer.token_id(EOS_TOKEN) == 3


def test_all_content_codes_are_in_vocabulary() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    assert set(CONTENT_CODE_BY_ID.values()) <= set(
        tokenizer.vocabulary
    )


def test_all_role_codes_are_in_vocabulary() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    assert set(ROLE_CODES.values()) <= set(
        tokenizer.vocabulary
    )


def test_all_register_codes_are_in_vocabulary() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    assert set(COMPOSITE_REGISTER_CODES) <= set(
        tokenizer.vocabulary
    )


def test_all_surface_forms_are_in_vocabulary() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    assert set(experimental_surface_forms()) <= set(
        tokenizer.vocabulary
    )


def test_researcher_content_ids_do_not_leak_into_vocabulary() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    assert set(ALL_CONTENT_IDS).isdisjoint(
        tokenizer.vocabulary
    )


def test_encode_decode_round_trip() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    text = (
        "<BOS> <REQ> <ROLE_01> <ROLE_03> "
        "<CONTENT_01> <C00> <SEP> <EOS>"
    )

    encoded = tokenizer.encode_text(text)
    decoded = tokenizer.decode_text(encoded)

    assert decoded == text


def test_unknown_token_is_rejected() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    with pytest.raises(
        ValueError,
        match="unknown token",
    ):
        tokenizer.encode_text(
            "<BOS> definitely_not_a_token <EOS>"
        )


def test_invalid_token_id_is_rejected() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    with pytest.raises(
        ValueError,
        match="out of range",
    ):
        tokenizer.token_for_id(9999)


def test_empty_text_is_rejected() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        tokenizer.encode_text("")


def test_tokenizer_fingerprint_is_deterministic() -> None:
    first = ClosedVocabularyTokenizer()
    second = ClosedVocabularyTokenizer()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_vocabulary_tokens_never_contain_whitespace() -> None:
    tokenizer = ClosedVocabularyTokenizer()

    for token in tokenizer.vocabulary:
        assert token
        assert token == token.strip()
        assert not any(
            character.isspace()
            for character in token
        )
