from perceiver.model.core.modules import PerceiverEncoder
from perceiver.model.core.adapter import InputAdapter
import torch


class FeatureInputAdapter(InputAdapter):
    """Pass-through adapter for pre-extracted continuous features.

    PerceiverEncoder does not use rotary position encoding, so a plain
    InputAdapter that reports the input feature dim is all that's needed.
    """

    def forward(self, x):
        return x


if __name__ == "__main__":
    lat_len = 128
    lat_dim = 512
    in_dim = 2048  # feature dim of x, i.e. x.shape[-1]
    device = 'cuda'
    input_adapter = FeatureInputAdapter(num_input_channels=in_dim)

    encoder = PerceiverEncoder(
        input_adapter=input_adapter,
        num_latents=lat_len,
        num_latent_channels=lat_dim,
        num_cross_attention_heads=16,
        num_cross_attention_layers=4,
        num_self_attention_heads=16,
        num_self_attention_layers_per_block=4,
        num_self_attention_blocks=4,
    ).to(device)

    # def forward(self, x, pad_mask=None, return_adapted_input=False):

    # 1, 32, 785, 2048 - final_tokens
    # 1, 32, 17, 2048 - camera & register tokens

    x = torch.randn((1, 32 * 785, 2048)).to(device)
    out = encoder(x)

    # Ok dummy test works, now create training job and let's run?

