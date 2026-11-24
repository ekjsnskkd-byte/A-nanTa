import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
"""Export exact frozen checkpoint scalers as native MQL5 constants."""
from pathlib import Path
import torch
import math
from backtest_v8_btc_dollar import FEATURES, TF_NAMES
root = Path(__file__).resolve().parents[1]
ckpt = torch.load(root / "models/v8_btc_dollar/ananta_v8_btc_dollar.pt", map_location="cpu", weights_only=False)
def emit(name, values, positive=False):
    vals = [float(values[k]) for k in FEATURES]
    assert all(math.isfinite(v) and (not positive or v > 0) for v in vals), name
    return "double " + name + "[40]={" + ",".join(format(v, ".17g") for v in vals) + "};\n"
out = "#ifndef ANANTA_SCALERS_MQH\n#define ANANTA_SCALERS_MQH\n"
out += emit("AnMed", ckpt["med"]) + emit("AnIqr", ckpt["iqr"], True)
for tf in TF_NAMES:
    out += emit("AnMed"+tf, ckpt["mtf_scalers"][tf]["med"])
    out += emit("AnIqr"+tf, ckpt["mtf_scalers"][tf]["iqr"], True)
out += "#endif\n"
(root / "EA/AnantaScalers.mqh").write_text(out)
