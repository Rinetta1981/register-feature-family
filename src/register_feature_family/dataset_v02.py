from register_feature_family.experimental_generator import (
    generate_confirmatory_records,
)
from register_feature_family.request_controls import (
    request_control_tokens,
)
from register_feature_family.schemas import (
    DatasetRecord,
    SpeechAct,
)

DATASET_VERSION_V02 = "0.2"


def _request_context_v02(
    context_text: str,
) -> str:
    """Replace one opaque request code with three entangled controls."""

    tokens = context_text.split()

    if not tokens:
        raise ValueError("request context must not be empty")

    composite_code = tokens[-1]
    controls = request_control_tokens(
        composite_code
    )

    return " ".join(
        (
            *tokens[:-1],
            *controls,
        )
    )


def convert_record_to_v02(
    record: DatasetRecord,
) -> DatasetRecord:
    """Convert one frozen-v0.1 record to the v0.2 input format."""

    if record.speech_act != SpeechAct.REQUEST:
        return record

    return record.model_copy(
        update={
            "context_text": _request_context_v02(
                record.context_text
            )
        }
    )


def generate_confirmatory_records_v02(
    seed: int = 0,
) -> list[DatasetRecord]:
    """Generate the v0.2 confirmatory records."""

    return [
        convert_record_to_v02(record)
        for record in generate_confirmatory_records(
            seed=seed
        )
    ]
