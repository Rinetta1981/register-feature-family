import pytest
import torch

from register_feature_family.dataset_v02 import (
    generate_confirmatory_records_v02,
)
from register_feature_family.model import (
    CausalSelfAttention,
    DecoderOnlyTransformer,
    TransformerConfig,
    causal_lm_loss,
)
from register_feature_family.sequence_encoding import (
    IGNORE_INDEX,
    encode_confirmatory_record,
)
from register_feature_family.tokenizer import (
    ClosedVocabularyTokenizer,
)


def test_default_transformer_configuration() -> None:
    config = TransformerConfig()

    assert config.n_layers == 2
    assert config.n_heads == 4
    assert config.d_model == 128
    assert config.d_mlp == 512
    assert config.head_dimension == 32
    assert config.max_sequence_length == 16
    assert config.vocab_size == 154


def test_default_parameter_count_is_frozen() -> None:
    model = DecoderOnlyTransformer()

    assert model.trainable_parameter_count() == 438272


def test_embeddings_and_unembedding_are_not_tied() -> None:
    model = DecoderOnlyTransformer()

    assert (
        model.token_embedding.weight.data_ptr()
        != model.unembedding.weight.data_ptr()
    )


def test_model_forward_shape() -> None:
    torch.manual_seed(0)

    model = DecoderOnlyTransformer()

    input_ids = torch.tensor(
        [
            [1, 4, 10, 20, 2, 100, 3],
            [1, 5, 11, 21, 2, 101, 3],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        7,
        model.config.vocab_size,
    )


def test_attention_pattern_has_expected_shape() -> None:
    torch.manual_seed(0)

    config = TransformerConfig()
    attention = CausalSelfAttention(config)

    hidden = torch.randn(
        2,
        5,
        config.d_model,
    )

    output, pattern = attention.forward_with_pattern(
        hidden
    )

    assert output.shape == (
        2,
        5,
        config.d_model,
    )

    assert pattern.shape == (
        2,
        config.n_heads,
        5,
        5,
    )

    row_sums = pattern.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


def test_attention_cannot_look_into_future() -> None:
    torch.manual_seed(0)

    config = TransformerConfig()
    attention = CausalSelfAttention(config)

    hidden = torch.randn(
        1,
        6,
        config.d_model,
    )

    _, pattern = attention.forward_with_pattern(
        hidden
    )

    future_mask = torch.triu(
        torch.ones(
            6,
            6,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    future_attention = pattern[
        0,
        :,
        future_mask,
    ]

    assert torch.count_nonzero(
        future_attention
    ).item() == 0


def test_padding_tokens_cannot_be_attended_to() -> None:
    torch.manual_seed(0)

    config = TransformerConfig()
    attention = CausalSelfAttention(config)

    hidden = torch.randn(
        1,
        5,
        config.d_model,
    )

    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0]],
        dtype=torch.long,
    )

    _, pattern = attention.forward_with_pattern(
        hidden,
        attention_mask=attention_mask,
    )

    assert torch.count_nonzero(
        pattern[:, :, :, 3:]
    ).item() == 0


def test_future_token_change_does_not_change_earlier_logits() -> None:
    torch.manual_seed(0)

    model = DecoderOnlyTransformer()
    model.eval()

    first = torch.tensor(
        [[1, 4, 10, 20, 2, 100, 3]],
        dtype=torch.long,
    )

    second = first.clone()
    second[0, -1] = 101

    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)

    assert torch.allclose(
        first_logits[:, :-1, :],
        second_logits[:, :-1, :],
        atol=1e-6,
    )


def test_causal_lm_loss_uses_one_token_shift() -> None:
    vocabulary_size = 8

    logits = torch.full(
        (1, 4, vocabulary_size),
        -20.0,
    )

    labels = torch.tensor(
        [
            [
                IGNORE_INDEX,
                IGNORE_INDEX,
                5,
                6,
            ]
        ],
        dtype=torch.long,
    )

    logits[0, 1, 5] = 20.0
    logits[0, 2, 6] = 20.0

    loss = causal_lm_loss(
        logits=logits,
        labels=labels,
    )

    assert loss.item() < 1e-6


def test_real_encoded_example_produces_finite_loss() -> None:
    torch.manual_seed(0)

    tokenizer = ClosedVocabularyTokenizer()
    record = generate_confirmatory_records_v02(
        seed=0
    )[0]

    encoded = encode_confirmatory_record(
        record,
        tokenizer=tokenizer,
    )

    input_ids = torch.tensor(
        [encoded.input_ids],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [encoded.labels],
        dtype=torch.long,
    )

    model = DecoderOnlyTransformer()

    logits = model(input_ids)

    loss = causal_lm_loss(
        logits=logits,
        labels=labels,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_loss_backpropagates_through_model() -> None:
    torch.manual_seed(0)

    record = generate_confirmatory_records_v02(
        seed=0
    )[0]

    encoded = encode_confirmatory_record(
        record
    )

    input_ids = torch.tensor(
        [encoded.input_ids],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [encoded.labels],
        dtype=torch.long,
    )

    model = DecoderOnlyTransformer()

    logits = model(input_ids)

    loss = causal_lm_loss(
        logits=logits,
        labels=labels,
    )

    loss.backward()

    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attention.q_proj.weight.grad is not None
    assert model.blocks[0].mlp.fc_in.weight.grad is not None
    assert model.blocks[1].attention.v_proj.weight.grad is not None
    assert model.unembedding.weight.grad is not None

    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.all(
                torch.isfinite(parameter.grad)
            )


def test_overlong_sequence_is_rejected() -> None:
    model = DecoderOnlyTransformer()

    input_ids = torch.zeros(
        (
            1,
            model.config.max_sequence_length + 1,
        ),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="exceeds configured maximum length",
    ):
        model(input_ids)


def test_invalid_attention_mask_shape_is_rejected() -> None:
    model = DecoderOnlyTransformer()

    input_ids = torch.zeros(
        (2, 5),
        dtype=torch.long,
    )

    bad_mask = torch.ones(
        (2, 4),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="attention_mask",
    ):
        model(
            input_ids,
            attention_mask=bad_mask,
        )


def test_invalid_head_configuration_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="divisible",
    ):
        TransformerConfig(
            d_model=127,
            n_heads=4,
        )
