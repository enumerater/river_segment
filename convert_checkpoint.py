"""
Convert 111.pth (custom tanh LayerNorm2d) → standard LayerNorm2d format.

Usage:
    python convert_checkpoint.py 111.pth 111_standard.pth

The custom tanh LayerNorm2d does:
    output = tanh(a * x) * g + b     (a=scalar, g=channel_wise_scale, b=channel_wise_bias)

The standard LayerNorm2d does:
    output = weight * (x-mean)/sqrt(var+eps) + bias

We set weight = a * g, bias = b as a best-effort approximation.
"""
import argparse
import torch
import os


def convert_state_dict(state_dict):
    """Convert custom tanh LayerNorm2d keys (a/g/b) to standard (weight/bias)."""
    # Mapping from custom patterns to standard
    replacements = [
        # (prefix, channel_dim)
        ("image_encoder.neck.1.", 256),
        ("image_encoder.neck.3.", 256),
        ("prompt_encoder.mask_downscaling.1.", 4),
        ("prompt_encoder.mask_downscaling.4.", 16),
        ("mask_decoder.output_upscaling.1.", 64),
    ]

    new_state_dict = {}
    keys_to_delete = set()

    for prefix, num_ch in replacements:
        # Custom keys
        a_key = prefix + "a"
        g_key = prefix + "g"
        b_key = prefix + "b"

        if a_key in state_dict and g_key in state_dict and b_key in state_dict:
            a = state_dict[a_key]  # scalar (1,)
            g = state_dict[g_key]  # (1, C, 1, 1)
            b = state_dict[b_key]  # (1, C, 1, 1)

            # Approximate conversion: weight ≈ a * g
            weight = (a * g).view(num_ch)
            bias = b.view(num_ch)

            new_state_dict[prefix + "weight"] = weight
            new_state_dict[prefix + "bias"] = bias

            keys_to_delete.update([a_key, g_key, b_key])

    # Copy all other keys
    for k, v in state_dict.items():
        if k not in keys_to_delete:
            new_state_dict[k] = v

    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert tanh-LayerNorm2d checkpoint to standard format.")
    parser.add_argument("input", type=str, help="Input checkpoint (e.g., 111.pth)")
    parser.add_argument("output", type=str, nargs="?", default=None, help="Output checkpoint")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_standard{ext}"

    print(f"Loading: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")
    print(f"Input keys: {len(ckpt)}")

    ckpt = convert_state_dict(ckpt)
    print(f"Output keys: {len(ckpt)}")

    torch.save(ckpt, args.output)
    print(f"Saved: {args.output}")
    print("Now use this checkpoint with the standard LayerNorm2d code.")


if __name__ == "__main__":
    main()