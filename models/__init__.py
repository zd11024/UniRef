from models.xvlm import XVLMBase
from models.xvlm import build_mlp
from models.xvlm import load_pretrained


def get_uni_attetion_mask(text_atts):
    batch_size, seq_length = text_atts.size()
    seq_ids = torch.arange(seq_length, device=text_atts.device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    causal_mask = causal_mask.to(text_atts.dtype)

    if causal_mask.shape[1] < text_atts.shape[1]:
        prefix_seq_len = text_atts.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                ),
                causal_mask,
            ],
            axis=-1,
        )
    uni_text_atts = causal_mask * text_atts[:, None, :]
    return uni_text_atts