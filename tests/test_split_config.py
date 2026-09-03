from collections import Counter

import pytest

from register_feature_family.codebook import ROLE_CODES
from register_feature_family.content_inventory import (
    ASSERTION_CONTENT_IDS,
    REQUEST_CONTENT_IDS,
)
from register_feature_family.schemas import SpeechAct, Split
from register_feature_family.split_config import (
    ASSERTION_LEXICAL_TRANSFER_HOLDOUTS,
    REQUEST_LEXICAL_TRANSFER_HOLDOUTS,
    in_distribution_content_ids,
    ordinary_split_for_group,
    select_lexical_transfer_holdouts,
)


def test_request_holdouts_are_frozen() -> None:



