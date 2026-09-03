from hashlib import sha256

from register_feature_family.codebook import (
    HEDGE_MARKER,
    INDIRECT_MARKER,
    MITIGATION_MARKER,
    AssertionContent,
    LexicalPair,
    RequestContent,
)
from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.schemas import SpeechAct
from register_feature_family.vocabulary import generate_unique_forms

EXPERIMENTAL_VOCABULARY_SEED = 2027

FORMS_PER_CONTENT = 3

RESERVED_SURFACE_FORMS: frozenset[str] = frozenset(
    {
        INDIRECT_MARKER,
        MITIGATION_MARKER,
        HEDGE_MARKER,
    }
)


def _stable_rank(
    *,
    value: str,
    seed: int,
    purpose: str,
) -> str:
    """Return a deterministic hash-based ranking value."""

    text = f"{seed}:{purpose}:{value}"

    return sha256(text.encode()).hexdigest()


def _generate_neutral_form_pool(
    *,
    count: int,
    seed: int,
) -> tuple[str, ...]:
    """Generate unique neutral forms before assigning register labels."""

    if count < 1:
        raise ValueError("count must be positive")

    candidate_count = count + len(RESERVED_SURFACE_FORMS)

    candidates = generate_unique_forms(
        count=candidate_count,
        seed=seed,
    )

    eligible = [
        form
        for form in candidates
        if form not in RESERVED_SURFACE_FORMS
    ]

    ranked = sorted(
        eligible,
        key=lambda form: (
            _stable_rank(
                value=form,
                seed=seed,
                purpose="surface-assignment",
            ),
            form,
        ),
    )

    if len(ranked) < count:
        raise RuntimeError("not enough eligible synthetic forms")

    return tuple(ranked[:count])


def _orient_lexical_pair(
    *,
    content_id: str,
    speech_act: SpeechAct,
    first: str,
    second: str,
    seed: int,
) -> LexicalPair:
    """Assign neutral pair members to lower and higher register."""

    if first == second:
        raise ValueError("lexical pair forms must be distinct")

    value = (
        f"{seed}:register-orientation:"
        f"{speech_act.value}:{content_id}"
    )
    digest = sha256(value.encode()).digest()

    if digest[0] % 2 == 0:
        less_formal = first
        more_formal = second
    else:
        less_formal = second
        more_formal = first

    return LexicalPair(
        less_formal=less_formal,
        more_formal=more_formal,
    )


def build_experimental_content(
    *,
    seed: int = EXPERIMENTAL_VOCABULARY_SEED,
) -> tuple[
    dict[str, RequestContent],
    dict[str, AssertionContent],
]:
    """Build the full deterministic 32-item experimental codebook."""

    total_contents = (
        len(REQUEST_CONTENT_IDS)
        + len(ASSERTION_CONTENT_IDS)
    )

    forms = _generate_neutral_form_pool(
        count=total_contents * FORMS_PER_CONTENT,
        seed=seed,
    )

    cursor = 0

    request_content: dict[str, RequestContent] = {}

    for content_id in REQUEST_CONTENT_IDS:
        first = forms[cursor]
        second = forms[cursor + 1]
        object_form = forms[cursor + 2]
        cursor += FORMS_PER_CONTENT

        request_content[content_id] = RequestContent(
            verb=_orient_lexical_pair(
                content_id=content_id,
                speech_act=SpeechAct.REQUEST,
                first=first,
                second=second,
                seed=seed,
            ),
            object_form=object_form,
        )

    assertion_content: dict[str, AssertionContent] = {}

    for content_id in ASSERTION_CONTENT_IDS:
        subject_form = forms[cursor]
        first = forms[cursor + 1]
        second = forms[cursor + 2]
        cursor += FORMS_PER_CONTENT

        assertion_content[content_id] = AssertionContent(
            subject_form=subject_form,
            predicate=_orient_lexical_pair(
                content_id=content_id,
                speech_act=SpeechAct.ASSERTION,
                first=first,
                second=second,
                seed=seed,
            ),
        )

    if cursor != len(forms):
        raise RuntimeError("surface-form allocation is incomplete")

    return request_content, assertion_content


(
    EXPERIMENTAL_REQUEST_CONTENT,
    EXPERIMENTAL_ASSERTION_CONTENT,
) = build_experimental_content()


def experimental_surface_forms() -> tuple[str, ...]:
    """Return every synthetic lexical form used by the experiment."""

    forms: list[str] = []

    for content_id in REQUEST_CONTENT_IDS:
        request_item = EXPERIMENTAL_REQUEST_CONTENT[content_id]

        forms.extend(
            (
                request_item.verb.less_formal,
                request_item.verb.more_formal,
                request_item.object_form,
            )
        )

    for content_id in ASSERTION_CONTENT_IDS:
        assertion_item = EXPERIMENTAL_ASSERTION_CONTENT[content_id]

        forms.extend(
            (
                assertion_item.subject_form,
                assertion_item.predicate.less_formal,
                assertion_item.predicate.more_formal,
            )
        )

    return tuple(forms)
