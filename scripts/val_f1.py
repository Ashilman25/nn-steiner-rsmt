from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "third_party", "NN-Steiner")
sys.path.insert(0, UPSTREAM)

from arora.data import DataLoader, QuadTreeDataset, collate_fn  # noqa: E402
from arora.models import NNArora  # noqa: E402
from arora.utils import get_files  # noqa: E402

CUTOFFS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]


# Same device order as arora.sh: CUDA, then Apple GPU, then CPU
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Builds the network from upstream's config and loads a checkpoint (DDP or not)
def load_model(path: str, device: torch.device) -> NNArora:
    conf = OmegaConf.load(os.path.join(UPSTREAM, "conf", "config.yaml"))
    model = NNArora(conf["model"], conf["quadtree"])
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(device).eval()


# Runs the model on every batch; returns (probabilities, labels) per batch
@torch.no_grad()
def predict(model: NNArora, batches: list, device: torch.device) -> list:
    out = []
    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.items()}
        golden = inputs.pop("golden").bool()
        out.append((model(**inputs).float().cpu(), golden.cpu()))
    return out


# F1 with "used" = prob >= cutoff; 0.5 uses round() like training's get_f1
def f1(prob: torch.Tensor, golden: torch.Tensor, cutoff: float) -> tuple:
    pred = torch.round(prob).bool() if cutoff == 0.5 else prob >= cutoff
    tp = int((pred & golden).sum())
    fp = int((pred & ~golden).sum())
    fn = int((~pred & golden).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * tp / (2 * tp + fp + fn), precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade trained models by F1 on a tensor dataset")
    parser.add_argument("models", nargs="+", help="checkpoint paths (.pt)")
    parser.add_argument("--data", default="work/data/batch4_100x100-uniform-128-tensor")
    parser.add_argument("--batch-size", type=int, default=32, help="examples per tree shape")
    args = parser.parse_args()

    device = pick_device()
    loader = DataLoader(QuadTreeDataset(get_files(args.data, True)), collate_fn, args.batch_size)
    batches = loader.load_data(0, 1)
    print(f"data: {args.data} ({len(batches)} batches), device: {device}\n")

    for path in args.models:
        results = predict(load_model(path, device), batches, device)
        prob = torch.cat([p.flatten() for p, _ in results])
        golden = torch.cat([g.flatten() for _, g in results])

        print(f"== {path}")
        print(f"   portals: {len(golden):,}, truly used: {golden.float().mean():.1%}")
        
        # Training logs the mean of per-batch F1s at the 0.5 cutoff
        per_batch = sum(f1(p.flatten(), g.flatten(), 0.5)[0] for p, g in results) / len(results)
        print(f"   val F1 as training logs it (mean of batches, 0.5 cutoff): {per_batch:.4f}")
        print("   cutoff   F1      precision  recall   portals kept")
        
        for c in CUTOFFS:
            score, prec, rec = f1(prob, golden, c)
            kept = int((torch.round(prob).bool() if c == 0.5 else prob >= c).sum())
            print(f"   {c:<6}   {score:.4f}  {prec:.4f}     {rec:.4f}   {kept:,}")
            
        print("   confidence band   portals   truly used (precision in band)")
        for lo, hi in BANDS:
            in_band = (prob >= lo) & (prob < hi)
            n = int(in_band.sum())
            hit = golden[in_band].float().mean().item() if n else float("nan")
            label = f"{lo:.2f}-{min(hi, 1.0):.2f}"
            print(f"   {label:<16}  {n:>7,}   {hit:.1%}")
        print()


if __name__ == "__main__":
    main()
