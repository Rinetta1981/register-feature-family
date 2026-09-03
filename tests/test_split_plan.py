from collections import Counter

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
    REQUEST_COMPOSITIONAL_OOD_CODES,
    REQUEST_LEXICAL_TRANSFER_HOLDOUTS,
    in_distribution_content_ids,
    training_supported_register_codes,
)
from register_feature_family.split_plan import assign_confirmatory_split


def _register_codes_for(
    speech_act: SpeechAct,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            code
            for code, config in COMPOSITE_REGISTER_CODES.items()
            if config.speech_act == speech_act
        )
    )


def test_confirmatory_split_counts_match_frozen_design() -> None:
    counts: Counter[Split] = Counter()
    excluded = 0
    roles = sorted(ROLE_CODES)

    families = (
        (SpeechAct.REQUEST, REQUEST_CONTENT_IDS),
        (SpeechAct.ASSERTION, ASSERTION_CONTENT_IDS),
    )

    for speech_act, content_ids in families:
        register_codes = _register_codes_for(speech_act)

        for content_id in content_ids:
            for speaker_role in roles:
                for addressee_role in roles:
                    for register_code in register_codes:
                        split = assign_confirmatory_split(
                            content_id=content_id,
                            speech_act=speech_act,
                            speaker_role=speaker_role,
                            addressee_role=addressee_role,
                            register_code=register_code,
                        )

                        if split is None:
                            excluded += 1
                        else:
                            counts[split] += 1

    assert counts == {
        Split.TRAIN: 840,
        Split.VALIDATION: 120,
        Split.IID_TEST: 120,
        Split.COMPOSITIONAL_OOD_TEST: 216,
        Split.LEXICAL_OOD_TEST: 360,
    }

    assert excluded == 72


def test_primary_confirmatory_dataset_has_1656_examples() -> None:
    total = 0
    roles = sorted(ROLE_CODES)

    families = (
        (SpeechAct.REQUEST, REQUEST_CONTENT_IDS),
        (SpeechAct.ASSERTION, ASSERTION_CONTENT_IDS),
    )

    for speech_act, content_ids in families:
        register_codes = _register_codes_for(speech_act)

        for content_id in content_ids:
            for speaker_role in roles:
                for addressee_role in roles:
                    for register_code in register_codes:
                        split = assign_confirmatory_split(
                            content_id=content_id,
                            speech_act=speech_act,
                            speaker_role=speaker_role,
                            addressee_role=addressee_role,
                            register_code=register_code,
                        )

                        if split is not None:
                            total += 1

    assert total == 1656


def test_request_joint_ood_examples_are_excluded() -> None:
    roles = sorted(ROLE_CODES)

    excluded = [
        assign_confirmatory_split(
            content_id=content_id,
            speech_act=SpeechAct.REQUEST,
            speaker_role=speaker_role,
            addressee_role=addressee_role,
            register_code=register_code,
        )
        for content_id in REQUEST_LEXICAL_TRANSFER_HOLDOUTS
        for speaker_role in roles
        for addressee_role in roles
        for register_code in REQUEST_COMPOSITIONAL_OOD_CODES
    ]

    assert len(excluded) == 72
    assert all(split is None for split in excluded)


def test_request_lexical_transfer_examples_never_enter_ordinary_splits() -> None:
    roles = sorted(ROLE_CODES)
    register_codes = training_supported_register_codes(
        SpeechAct.REQUEST
    )

    for content_id in REQUEST_LEXICAL_TRANSFER_HOLDOUTS:
        for speaker_role in roles:
            for addressee_role in roles:
                for register_code in register_codes:
                    split = assign_confirmatory_split(
                        content_id=content_id,
                        speech_act=SpeechAct.REQUEST,
                        speaker_role=speaker_role,
                        addressee_role=addressee_role,
                        register_code=register_code,
                    )

                    assert split == Split.LEXICAL_OOD_TEST


def test_assertion_lexical_transfer_examples_are_lexical_ood() -> None:
    roles = sorted(ROLE_CODES)
    register_codes = training_supported_register_codes(
        SpeechAct.ASSERTION
    )

    for content_id in ASSERTION_LEXICAL_TRANSFER_HOLDOUTS:
        for speaker_role in roles:
            for addressee_role in roles:
                for register_code in register_codes:
                    split = assign_confirmatory_split(
                        content_id=content_id,
                        speech_act=SpeechAct.ASSERTION,
                        speaker_role=speaker_role,
                        addressee_role=addressee_role,
                        register_code=register_code,
                    )

                    assert split == Split.LEXICAL_OOD_TEST


def test_request_compositional_holdouts_are_always_compositional_ood() -> None:
    roles = sorted(ROLE_CODES)
    content_ids = in_distribution_content_ids(SpeechAct.REQUEST)

    for content_id in content_ids:
        for speaker_role in roles:
            for addressee_role in roles:
                for register_code in REQUEST_COMPOSITIONAL_OOD_CODES:
                    split = assign_confirmatory_split(
                        content_id=content_id,
                        speech_act=SpeechAct.REQUEST,
                        speaker_role=speaker_role,
                        addressee_role=addressee_role,
                        register_code=register_code,
                    )

                    assert split == Split.COMPOSITIONAL_OOD_TEST


def test_ordinary_comparison_groups_stay_together() -> None:
    roles = sorted(ROLE_CODES)

    for speech_act in (SpeechAct.REQUEST, SpeechAct.ASSERTION):
        content_ids = in_distribution_content_ids(speech_act)
        register_codes = training_supported_register_codes(speech_act)

        for content_id in content_ids:
            for speaker_role in roles:
                for addressee_role in roles:
                    splits = {
                        assign_confirmatory_split(
                            content_id=content_id,
                            speech_act=speech_act,
                            speaker_role=speaker_role,
                            addressee_role=addressee_role,
                            register_code=register_code,
                        )
                        for register_code in register_codes
                    }

                    assert len(splits) == 1


def test_split_planner_is_deterministic() -> None:
    first = assign_confirmatory_split(
        content_id="send_report",
        speech_act=SpeechAct.REQUEST,
        speaker_role="role_01",
        addressee_role="role_03",
        register_code="<C00>",
    )

    second = assign_confirmatory_split(
        content_id="send_report",
        speech_act=SpeechAct.REQUEST,
        speaker_role="role_01",
        addressee_role="role_03",
        register_code="<C00>",
    )

    assert first == second
