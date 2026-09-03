from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.experimental_codebook import (
    EXPERIMENTAL_ASSERTION_CONTENT,
    EXPERIMENTAL_REQUEST_CONTENT,
    EXPERIMENTAL_VOCABULARY_SEED,
    RESERVED_SURFACE_FORMS,
    build_experimental_content,
    experimental_surface_forms,
)
from register_feature_family.vocabulary import CONSONANTS, VOWELS


def test_all_frozen_request_contents_are_present() -> None:
    assert tuple(EXPERIMENTAL_REQUEST_CONTENT) == REQUEST_CONTENT_IDS


def test_all_frozen_assertion_contents_are_present() -> None:
    assert tuple(EXPERIMENTAL_ASSERTION_CONTENT) == ASSERTION_CONTENT_IDS


def test_experimental_codebook_has_32_contents() -> None:
    total = (
        len(EXPERIMENTAL_REQUEST_CONTENT)
        + len(EXPERIMENTAL_ASSERTION_CONTENT)
    )

    assert total == 32


def test_experimental_codebook_has_96_surface_forms() -> None:
    forms = experimental_surface_forms()

    assert len(forms) == 96


def test_all_surface_forms_are_globally_unique() -> None:
    forms = experimental_surface_forms()

    assert len(set(forms)) == 96


def test_surface_forms_do_not_collide_with_grammar_markers() -> None:
    forms = set(experimental_surface_forms())

    assert forms.isdisjoint(RESERVED_SURFACE_FORMS)


def test_all_surface_forms_have_cvcv_shape() -> None:
    for form in experimental_surface_forms():
        assert len(form) == 4
        assert form[0] in CONSONANTS
        assert form[1] in VOWELS
        assert form[2] in CONSONANTS
        assert form[3] in VOWELS


def test_request_lexical_pairs_are_distinct() -> None:
    for content in EXPERIMENTAL_REQUEST_CONTENT.values():
        assert content.verb.less_formal != content.verb.more_formal


def test_assertion_lexical_pairs_are_distinct() -> None:
    for content in EXPERIMENTAL_ASSERTION_CONTENT.values():
        assert (
            content.predicate.less_formal
            != content.predicate.more_formal
        )


def test_request_fixed_objects_are_not_lexical_pair_members() -> None:
    for content in EXPERIMENTAL_REQUEST_CONTENT.values():
        assert content.object_form not in {
            content.verb.less_formal,
            content.verb.more_formal,
        }


def test_assertion_fixed_subjects_are_not_lexical_pair_members() -> None:
    for content in EXPERIMENTAL_ASSERTION_CONTENT.values():
        assert content.subject_form not in {
            content.predicate.less_formal,
            content.predicate.more_formal,
        }


def test_experimental_codebook_is_deterministic() -> None:
    first_requests, first_assertions = build_experimental_content(
        seed=EXPERIMENTAL_VOCABULARY_SEED
    )
    second_requests, second_assertions = build_experimental_content(
        seed=EXPERIMENTAL_VOCABULARY_SEED
    )

    assert first_requests == second_requests
    assert first_assertions == second_assertions


def test_different_seed_changes_codebook() -> None:
    default_requests, default_assertions = build_experimental_content(
        seed=EXPERIMENTAL_VOCABULARY_SEED
    )
    alternative_requests, alternative_assertions = (
        build_experimental_content(seed=999)
    )

    assert (
        default_requests != alternative_requests
        or default_assertions != alternative_assertions
    )
