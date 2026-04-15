"""Offline evaluation of a LeJEPA pretraining checkpoint.

Loads the saved encoder + probe and reports linear probe top-1/top-5 +
KNN accuracy on the validation split. Uses the in-memory data path
when available.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import Config
from data import get_dataloaders, InMemoryGPUDataset, InMemoryEvalLoader
from models import LeJEPAEncoder, LinearProbe
from sliced_gauss_reg.evaluate import evaluate_full


@torch.no_grad()
def collect_embeddings(encoder, loader, device):
    encoder.eval()
    embs, labs = [], []
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, labels in loader:
            if images.device != device:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            emb, _ = encoder(images)
            embs.append(emb.float().cpu())
            labs.append(labels.cpu())
    return torch.cat(embs), torch.cat(labs)


def knn_accuracy(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    embeddings = F.normalize(embeddings, dim=1)
    sims = embeddings @ embeddings.T
    sims.fill_diagonal_(-float("inf"))
    _, topk_idx = sims.topk(k, dim=1)
    topk_labels = labels[topk_idx]
    pred = topk_labels.mode(dim=1).values
    return (pred == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LeJEPA checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = ckpt["config"]
    saved_cfg["data_dir"] = args.data_dir
    saved_cfg["batch_size"] = args.batch_size
    cfg = Config(**saved_cfg)

    encoder = LeJEPAEncoder(cfg).to(device)
    probe = LinearProbe(encoder.hidden_dim, cfg.num_classes).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    probe.load_state_dict(ckpt["probe"])
    encoder.eval()
    probe.eval()

    _, val_source, _ = get_dataloaders(cfg, device)

    # Probe accuracy
    correct = top5_correct = total = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, labels in val_source:
            if images.device != device:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            emb, _ = encoder(images)
            logits = probe(emb)
            correct += (logits.argmax(1) == labels).sum().item()
            top5_correct += (logits.topk(5, dim=1).indices
                             == labels.unsqueeze(1)).any(1).sum().item()
            total += labels.size(0)

    top1 = correct / total
    top5 = top5_correct / total

    embeddings, labels = collect_embeddings(encoder, val_source, device)
    quality = evaluate_full(embeddings.numpy())
    knn10 = knn_accuracy(embeddings, labels, k=10)
    knn20 = knn_accuracy(embeddings, labels, k=20)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {cfg.dataset}, backbone: {cfg.backbone_name}")
    print(f"Epoch: {ckpt.get('epoch', '?')}, regularizer: {cfg.regularizer}, "
          f"accumulate: {cfg.accumulate}")
    print()
    print(f"Probe Top-1: {top1:.4f}")
    print(f"Probe Top-5: {top5:.4f}")
    print(f"KNN-10:      {knn10:.4f}")
    print(f"KNN-20:      {knn20:.4f}")
    print()
    print("Embedding quality:")
    for k, v in quality.items():
        if isinstance(v, (list, tuple)):
            arr = torch.tensor(v, dtype=torch.float32)
            print(f"  {k}: mean={arr.mean().item():.6f} "
                  f"std={arr.std().item():.6f} "
                  f"min={arr.min().item():.6f} "
                  f"max={arr.max().item():.6f} "
                  f"(n={arr.numel()})")
        else:
            print(f"  {k}: {float(v):.6f}")


if __name__ == "__main__":
    main()
