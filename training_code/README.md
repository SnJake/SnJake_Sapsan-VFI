# FrameNN

Frame interpolation (x2) training/inference pipeline with a modern flow + transformer fusion model.

## Quick start

1) Edit `config.yaml` and set your data roots. Pick a backend: `opencv` (default), `decord`, `pyav`, or `auto`.
2) (Optional) Build a manifest once:

```powershell
python -m scripts.build_manifest --manifest runs/exp001/dataset_manifest.jsonl --roots H:/Videos/Cartoons --backend opencv --errors runs/exp001/manifest_errors.jsonl
```

3) Train:

```powershell
python -m src.train --config config.yaml
```

4) Resume:

```powershell
python -m src.train --config config.yaml --resume runs/exp001/checkpoints/last_full.pt
```

5) Inference:

```powershell
python -m src.infer --config config.yaml --weights runs/exp001/checkpoints/last_weights.pt --input input.mp4 --output output_x2.mp4
```

## Features

- AMP: bf16 / fp16 / no
- AdamW + cosine or constant scheduler + warmup
- EMA weights for inference checkpoint
- JSONL training logs
- Full checkpoint for resume and separate weights-only checkpoint
- Recursive dataset scanning + manifest caching

## Notes

- `train_crop` controls patch size. Increase for better quality if VRAM allows.
- `frame_stride_min/max` control motion magnitude for training samples.
- `pad_multiple` should be a multiple of 8 (or your attention window size).
- `performance.*` toggles PyTorch SDPA backends (flash / mem-efficient / math).
- `data.backend` selects the video backend; `decord` is usually fastest for random access.

Optional backends:

```powershell
pip install decord
pip install av
```
Optional WebDataset:

```powershell
pip install webdataset
```

## Dataset preparation (SSD subset)

Create a smaller, subset on SSD with per-root limits:

```powershell
python -m scripts.prepare_dataset --out path ^
  --root-limit "" ^
  --root-limit "" ^
  --root-limit "" ^
  --root-limit "" ^
  --mode copy --skip-existing --seed 42
```

Optional: write a manifest for the prepared subset:

```powershell
python -m scripts.prepare_dataset --out path --root-limit path --write-manifest path/manifest.jsonl --manifest-backend decord
```

## Triplet cache (NPZ)

Create a cache of pre-sampled triplets (much faster training, lower CPU usage):

```powershell
python -m scripts.cache_triplets --config config.yaml --out path --count 200000 --batch-size 8 --num-workers 4 --backend decord
```

Then train from NPZ:

```yaml
data:
  format: "npz"
  npz_dir: "E:/FrameNN_triplets"
```

## Triplet cache (WebDataset) 

Create WebDataset shards (PNG triplets inside .tar files):

```powershell
python -m scripts.cache_wds --config config.yaml --out path --count 200000 --batch-size 4 --num-workers 3 --backend decord --shard-size 1000
```

Train from WebDataset:

```yaml
data:
  format: "wds"
  wds_urls: "E:/FrameNN_wds/shard-{000000..000199}.tar"
  wds_shuffle: 1000
```

## Model diagnostics

```
python -m src.diagnose --config config.yaml --weights path/to/model --input "" --max-pairs 80 --match-brightness
```