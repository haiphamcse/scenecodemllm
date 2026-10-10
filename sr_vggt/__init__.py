"""sr_vggt: Qwen3-VL + VGGT spatial-reasoning finetuning (see README.md)."""
import os
import sys

# Vendored third-party code (perceiver-io, vggt-omega, VGGT-1B) wins over any installed copy.
_THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
