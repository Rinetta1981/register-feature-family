from register_feature_family.codebook import COMPOSITE_REGISTER_CODES

REQUEST_COMPOSITE_CODES: tuple[str, ...] = (
    "<C00>",
    "<C01>",
    "<C02>",
    "<C03>",
    "<C04>",
    "<C05>",
    "<C06>",
    "<C07>",
)

REQUEST_CONTROL_TOKENS: tuple[str, ...] = (
    "<RC1_0>",
    "<RC1_1>",
    "<RC2_0>",
    "<RC2_1>",
    "<RC3_0>",
    "<RC3_1>",
)

REQUEST_CONTROL_BY_COMPOSITE_CODE: dict[
    str,
    tuple[str, str, str],
] = {
    "<C00>": ("<RC1_0>", "<RC2_0>", "<RC3_0>"),
    "<C01>": ("<RC1_0>", "<RC2_1>", "<RC3_1>"),
    "<C02>": ("<RC1_1>", "<RC2_1>", "<RC3_1>"),
    "<C03>": ("<RC1_1>", "<RC2_0>", "<RC3_0>"),
    "<C04>": ("<RC1_1>", "<RC2_0>", "<RC3_1>"),
    "<C05>": ("<RC1_1>", "<RC2_1>", "<RC3_0>"),
    "<C06>": ("<RC1_0>", "<RC2_1>", "<RC3_0>"),
    "<C07>": ("<RC1_0>", "<RC2_0>", "<RC3_1>"),
}

if not set(REQUEST_COMPOSITE_CODES) <= set(COMPOSITE_REGISTER_CODES):
    raise RuntimeError(
        "request control mapping contains unknown composite codes"
    )

if set(REQUEST_CONTROL_BY_COMPOSITE_CODE) != set(
    REQUEST_COMPOSITE_CODES
):
    raise RuntimeError(
        "request control mapping is incomplete"
    )


def request_control_tokens(
    composite_code: str,
) -> tuple[str, str, str]:
    """Return the v0.2 entangled controls for one request code."""

    try:
        return REQUEST_CONTROL_BY_COMPOSITE_CODE[
            composite_code
        ]
    except KeyError as error:
        raise ValueError(
            f"not a request composite code: {composite_code}"
        ) from error
