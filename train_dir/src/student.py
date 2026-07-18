"""Student model forward pass for SDFT."""

import torch


def forward_student(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    completion_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Student forward pass. Returns logits at completion positions.

    Uses backbone (model.model) + selective lm_head to avoid allocating
    the full (1, S, V) logits tensor. For 8B models with V=152K, the full
    logits can be >1 GB vs ~30 MB for hidden states.

    Returns: (C, V) tensor with gradient attached.
    """
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)
    prompt_ids = prompt_enc["input_ids"][0]  # (P,)
    prompt_len = prompt_ids.size(0)
    C = len(completion_ids)

    comp_ids_t = torch.tensor(completion_ids, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, comp_ids_t]).unsqueeze(0)  # (1, P+C)
    attn_mask = torch.ones_like(input_ids)

    # Run backbone under torch.utils.checkpoint so the full (1, S, hidden_dim)
    # hidden states are recomputed during backward instead of stored.
    def _backbone_fwd(ids, mask):
        return model.model(input_ids=ids, attention_mask=mask)[0]

    hidden = torch.utils.checkpoint.checkpoint(
        _backbone_fwd, input_ids, attn_mask, use_reentrant=False,
    )  # (1, S, hidden_dim) — recomputed on backward, not stored

    # Extract completion positions, apply lm_head selectively
    # Position P-1 predicts completion token 0, ..., P+C-2 predicts token C-1
    completion_hidden = hidden[0, prompt_len - 1 : prompt_len + C - 1, :]  # (C, hidden_dim)
    completion_logits = model.lm_head(completion_hidden)  # (C, V)
    return completion_logits
