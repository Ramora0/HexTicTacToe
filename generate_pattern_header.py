"""Generate cpp/pattern_data.h from the learned pattern weights.

Usage:
    python generate_pattern_header.py
"""

from ai import _load_pattern_values, _DEFAULT_PATTERN_PATH

pv, eval_length = _load_pattern_values(_DEFAULT_PATTERN_PATH)

with open("cpp/pattern_data.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write(f"static constexpr int PATTERN_EVAL_LENGTH = {eval_length};\n")
    f.write(f"static constexpr int PATTERN_COUNT = {len(pv)};\n")
    f.write("static const double PATTERN_VALUES[] = {\n")
    for i in range(0, len(pv), 10):
        chunk = pv[i:i+10]
        f.write("    " + ", ".join(f"{v:.10g}" for v in chunk) + ",\n")
    f.write("};\n")

print(f"Wrote cpp/pattern_data.h: {len(pv)} values, eval_length={eval_length}")
