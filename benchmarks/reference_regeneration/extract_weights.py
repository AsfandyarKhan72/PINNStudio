"""Extract the network weights from PyTorch checkpoints without PyTorch, one .npz (and the logs, as .json) per run.
    python3 extract_weights.py OUT_DIR RUN_GROUP_DIR [...]

Run on the GPU machine over the done/<batch>/out/<example> folders; reeval_runs.py reads OUT_DIR. The last
checkpoint of each folder is taken (the highest iteration number), as the run's own Error Analysis did."""
import glob, io, os, pickle, re, sys, zipfile, json
import numpy as np

DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16, "LongStorage": np.int64,
      "IntStorage": np.int32, "BoolStorage": np.bool_, "ByteStorage": np.uint8, "BFloat16Storage": None}


class Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return Dummy()
    def __setstate__(self, s): pass


def load(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]

    def rebuild(storage, offset, size, stride, *a):
        arr, dt = storage
        it = np.dtype(dt).itemsize
        if len(size) == 0:
            return arr[offset].copy()
        return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size), strides=tuple(s * it for s in stride)).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name in DT:
                return name
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if mod.startswith("torch") and name == "_rebuild_parameter":
                return lambda data, *a: data
            try:
                return super().find_class(mod, name)
            except Exception:
                return Dummy

        def persistent_load(self, pid):
            typ, stype, key, loc, numel = pid[:5]
            if isinstance(stype, str):
                dt = DT[stype]
            else:                      # newer format: dtype given elsewhere; assume float32
                dt = np.float32
            buf = z.read(f"{prefix}data/{key}")
            return (np.frombuffer(buf, dtype=dt), dt)

    obj = U(io.BytesIO(z.read(pkl))).load()
    sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    return {k: v for k, v in sd.items() if isinstance(k, str) and isinstance(v, np.ndarray)}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    n = 0
    for g in sys.argv[2:]:
        for run in sorted(glob.glob(os.path.join(g, "*_s*"))):
            bydir = {}
            for p in glob.glob(run + "/**/*.pt", recursive=True):
                m = re.findall(r"(\d+)\.pt$", p)
                it = int(m[0]) if m else -1
                d = os.path.dirname(p)
                if d not in bydir or it > bydir[d][0]:
                    bydir[d] = (it, p)
            arrays, meta = {}, {"run": run, "checkpoints": []}
            for i, (d, (it, p)) in enumerate(sorted(bydir.items())):
                try:
                    sd = load(p)
                except Exception as exc:
                    meta["checkpoints"].append({"path": os.path.relpath(p, run), "error": repr(exc)})
                    continue
                meta["checkpoints"].append({"path": os.path.relpath(p, run), "iteration": it, "keys": sorted(sd)})
                for k, v in sd.items():
                    arrays[f"c{i}|{k}"] = v
            texts = {}
            for pat in ("log.txt", "**/model_config.json", "**/step_config.json", "**/error_metrics*.txt"):
                for f in glob.glob(os.path.join(run, pat), recursive=True):
                    texts[os.path.relpath(f, run)] = open(f, errors="replace").read()
            meta["texts"] = texts
            name = run.replace("/", "__")
            np.savez_compressed(os.path.join(out, name + ".npz"), **arrays)
            json.dump(meta, open(os.path.join(out, name + ".json"), "w"))
            n += 1
    print(n, "runs")


if __name__ == "__main__":
    main()
