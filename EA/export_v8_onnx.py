import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
"""Export the frozen V8 neural network only; no policy or feature engineering."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from backtest_v8_btc_dollar import UniversalQuantModel, CFG, FEATURES, TF_NAMES

root = Path(__file__).resolve().parents[1]
folder = root / "models/v8_btc_dollar"
ckpt = torch.load(folder / "ananta_v8_btc_dollar.pt", map_location="cpu", weights_only=False)
mc = ckpt["model_config"]
model = UniversalQuantModel(mc["features"], mc["hidden"], mc["nexperts"], mc["topk"], mc["nh"], mc["layers"], mc["heads"], mc["gru_layers"], mc["expert_width"], float(ckpt["ret_scale"]), float(ckpt["rv_scale"]))
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()
torch.set_num_threads(2)
torch.backends.mha.set_fastpath_enabled(False)
names = ["direction","quantiles","rv","excursions","barrier","pattern","pattern_detail","jump","path_mu","path_diag","path_factors","action_quantiles","decision","regime","mtf_attention"]
# Replace data-dependent sparse dispatch with equivalent dense weighted expert sum.
import inspect, textwrap
source = textwrap.dedent(inspect.getsource(UniversalQuantModel.forward))
start = source.index("    mixture = torch.zeros_like(z0)")
end = source.index("    z = z0 + mixture", start)
source = source[:start] + "    mixture = torch.stack([expert(z0) for expert in self.experts], dim=1).mul(gate.unsqueeze(-1)).sum(1)\n" + source[end:]
namespace = dict(UniversalQuantModel.forward.__globals__)
exec(source, namespace)
export_model = copy.deepcopy(model)
import types
export_model.forward = types.MethodType(namespace["forward"], export_model)
class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x, mtf, mtf_mask):
        out = self.net(x, mtf, mtf_mask)
        return tuple(out[k] for k in names)
wrapper = Wrapper(export_model).eval()
def inputs(seed, batch=1, mask_mode=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, CFG["lookback"], mc["features"], generator=g)
    mtf = torch.randn(batch, len(TF_NAMES), CFG["mtf_window"], mc["features"], generator=g)
    mask = torch.rand(batch, len(TF_NAMES), CFG["mtf_window"], generator=g) > .3
    if mask_mode == 1: mask[:] = False
    if mask_mode == 2: mask[:] = True
    return x, mtf, mask
path = root / "EA" / "ananta_v8_btc_dollar.onnx"
with torch.no_grad():
    torch.onnx.export(wrapper, inputs(0), str(path), input_names=["x","mtf","mtf_mask"], output_names=names, opset_version=17, dynamo=False, do_constant_folding=True)
onnx.checker.check_model(str(path))
session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
errors = {k: 0.0 for k in names}
with torch.no_grad():
    for seed in range(24):
        args = inputs(seed, mask_mode=seed % 3)
        ref = model(*args)
        actual = session.run(names, {k: v.numpy() for k,v in zip(["x","mtf","mtf_mask"],args)})
        for k, v in zip(names, actual):
            expected = ref[k].numpy()
            np.testing.assert_allclose(v, expected, rtol=2e-4, atol=2e-5, err_msg=k)
            errors[k] = max(errors[k], float(np.max(np.abs(v-expected))))
meta = {"inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()], "outputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_outputs()], "features": FEATURES, "timeframes": TF_NAMES, "horizons": CFG["horizons"], "path_knots": CFG["path_knots"], "calibration_temperature_external": float(ckpt["calibration_temperature"]), "parity_max_abs": errors, "parity_cases": 24, "policy_included": False, "preprocessing_included": False}
(root / "EA" / "onnx_manifest.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
