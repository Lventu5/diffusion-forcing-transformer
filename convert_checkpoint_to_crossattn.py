"""
Convert an UViT3DActionNode checkpoint to UViT3DActionNodeCrossAttn.

What changed between the two backbones:

  OLD  external_cond_embedding = ActionNodeCondEmbedding
       └─ action_embedding.type_embedding.*
       └─ action_embedding.pos_encoding.*
       └─ action_embedding.proj.*
       └─ node_proj.*          ← dropped (text now goes via cross-attn)

  NEW  external_cond_embedding = ActionCondEmbedding  (flat, action only)
       └─ type_embedding.*
       └─ pos_encoding.*
       └─ proj.*

  NEW  down_blocks / mid_blocks / up_blocks TransformerBlocks now have
       cross_attn.* sub-keys — these are zero-initialised by design, so
       they are NOT in the old checkpoint.  The DFoTVideo.on_load_checkpoint
       hook fills them in with the model's zero-init values when
       algorithm.checkpoint.strict=False is set.

Usage:
    # Step 1 — remap keys
    python convert_checkpoint_to_crossattn.py \\
        --input  outputs/my_run/checkpoints/last.ckpt \\
        --output outputs/my_run/checkpoints/last_crossattn.ckpt

    # Step 2 — finetune (strict=False lets on_load_checkpoint fill new keys)
    python main.py ... backbone=uvit3d_action_node_crossattn \\
        load=<wandb-run-id-or-path-to-ckpt> \\
        algorithm.checkpoint.strict=False \\
        algorithm.checkpoint.reset_optimizer=True
"""

import argparse
import torch


# Backbone params live at "diffusion_model.model.*" inside the Lightning
# state_dict (the diffusion wrapper calls its backbone "self.model").
# For torch.compile'd runs the prefix is "diffusion_model._orig_mod.model.*"
# — we handle both below.
_BACKBONE_PREFIXES = [
    "diffusion_model.model.",
    "diffusion_model._orig_mod.model.",
]

_ACTION_EMB_OLD  = "external_cond_embedding.action_embedding."
_ACTION_EMB_NEW  = "external_cond_embedding."
_NODE_PROJ       = "external_cond_embedding.node_proj."
_NODE_DROPOUT    = "external_cond_embedding.node_dropout."
_ACTION_DROPOUT  = "external_cond_embedding.action_dropout."


def _remap(old_key: str) -> str | None:
    """
    Return the new key name, or None if the key should be dropped.
    """
    # Find which backbone prefix this key belongs to (if any)
    prefix = next((p for p in _BACKBONE_PREFIXES if old_key.startswith(p)), None)
    if prefix is None:
        return old_key  # not a backbone key — keep as-is

    local = old_key[len(prefix):]

    # Keys that no longer exist in the new model
    if (
        local.startswith(_NODE_PROJ)
        or local.startswith(_NODE_DROPOUT)
        or local.startswith(_ACTION_DROPOUT)
    ):
        return None  # drop

    # Remap action_embedding.* → flat *
    if local.startswith(_ACTION_EMB_OLD):
        local = local.replace(_ACTION_EMB_OLD, _ACTION_EMB_NEW, 1)
        return prefix + local

    return old_key  # unchanged


def convert(input_path: str, output_path: str) -> None:
    print(f"Loading  {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")

    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    dropped, remapped, kept = [], [], []

    for old_key, tensor in state_dict.items():
        new_key = _remap(old_key)
        if new_key is None:
            dropped.append(old_key)
        elif new_key != old_key:
            remapped.append((old_key, new_key))
            new_state_dict[new_key] = tensor
        else:
            kept.append(old_key)
            new_state_dict[old_key] = tensor

    print(f"\nRemapped {len(remapped)} keys:")
    for old, new in remapped:
        print(f"  {old}")
        print(f"    -> {new}")

    print(f"\nDropped  {len(dropped)} keys (node_proj / dropouts):")
    for k in dropped:
        print(f"  {k}")

    print(f"\nKept     {len(kept)} keys unchanged.")
    print(
        "\nNote: cross_attn.* weights are missing from this checkpoint — that is "
        "expected. DFoTVideo.on_load_checkpoint fills them in with the model's "
        "zero-init values when algorithm.checkpoint.strict=False is set.\n"
        "Because out.weight is zero-initialised, cross-attn starts as a no-op "
        "and the model behaves identically to the old one at the start of finetuning."
    )

    if "state_dict" in ckpt:
        ckpt["state_dict"] = new_state_dict
    else:
        ckpt = new_state_dict

    torch.save(ckpt, output_path)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  required=True, help="Path to old .ckpt")
    parser.add_argument("--output", required=True, help="Where to write converted .ckpt")
    args = parser.parse_args()
    convert(args.input, args.output)
