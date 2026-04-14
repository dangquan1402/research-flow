"""Encoder-decoder arithmetic transformer (MLX) — placeholder."""

from . import register_model


@register_model("encoder_decoder")
class EncoderDecoderTransformer:
    # TODO: Implement encoder-decoder architecture
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "Encoder-decoder architecture is not yet implemented. "
            "See models/decoder_only.py for reference."
        )
