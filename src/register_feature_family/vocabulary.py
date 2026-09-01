import random
from dataclasses import dataclass

CONSONANTS: tuple[str, ...] = (
    "b",
    "d",
    "f",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
)

VOWELS: tuple[str, ...] = (
    "a",
    "e",
    "i",
    "o",
    "u",
)


@dataclass(frozen=True)
class GeneratedLexicalPair:
    """Two synthetic forms with identical meaning by construction."""

    semantic_id: str
    less_formal: str
    more_formal: str


def generate_cvcv_form(rng: random.Random) -> str:
    """Generate one synthetic CVCV form."""

    return "".join(
        (
            rng.choice(CONSONANTS),
            rng.choice(VOWELS),
            rng.choice(CONSONANTS),
            rng.choice(VOWELS),
        )
    )


def generate_unique_forms(
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Generate a deterministic list of unique synthetic forms."""

    if count < 1:
        raise ValueError("count must be at least 1.")

    rng = random.Random(seed)
    forms: set[str] = set()

    while len(forms) < count:
        forms.add(generate_cvcv_form(rng))

    return sorted(forms)


def generate_lexical_pairs(
    *,
    semantic_ids: list[str],
    seed: int,
) -> list[GeneratedLexicalPair]:
    """Generate deterministic paired lexical forms for semantic items."""

    if not semantic_ids:
        raise ValueError("semantic_ids must not be empty.")

    if len(semantic_ids) != len(set(semantic_ids)):
        raise ValueError("semantic_ids must be unique.")

    forms = generate_unique_forms(
        count=len(semantic_ids) * 2,
        seed=seed,
    )

    rng = random.Random(seed + 1)
    rng.shuffle(forms)

    pairs: list[GeneratedLexicalPair] = []

    for index, semantic_id in enumerate(semantic_ids):
        first = forms[index * 2]
        second = forms[index * 2 + 1]

        if rng.random() < 0.5:
            less_formal = first
            more_formal = second
        else:
            less_formal = second
            more_formal = first

        pairs.append(
            GeneratedLexicalPair(
                semantic_id=semantic_id,
                less_formal=less_formal,
                more_formal=more_formal,
            )
        )

    return pairs
