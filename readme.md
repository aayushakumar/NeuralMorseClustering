
# Neural Mode Clustering (NMC)

**Unsupervised clustering with a learned potential in embedding space.**
This implementation trains a neural potential via Denoising Score Matching (DSM), seeks modes with gradient ascent, deduplicates them, and prunes clusters using 0-D topological persistence. It includes stability metrics, cached-embedding training, and baselines.

> Dataset support in this release: **CIFAR-10**.

---

## Highlights

* **Frozen encoder + trainable projection head** (ResNet-18 backbone, optional projection)
* **DSM training** with log-uniform noise and **invariance regularization** (gradient-cosine or endpoint)
* **Mode seeking** via gradient ascent with **element-wise backtracking**
* **DBSCAN deduplication** with **K-Means fallback**
* **0-D persistence pruning** (keeps “immortal” components, percentile threshold on mortal ones)
* **Stability metrics**

  * `Stability@Aug`: median-thresholded endpoint agreement across augmentations
  * `Stability@Assign`: assignment consistency under a second view
* **Embedding cache** for faster training passes
* **Full experiment export**: weights, metrics, predictions, comparison CSV, and a ready-to-paste LaTeX table

---

## Environment

* Python ≥ 3.9
* PyTorch ≥ 2.1, torchvision ≥ 0.16
* NumPy, SciPy, pandas, scikit-learn, matplotlib, tqdm, Pillow

Example (CUDA build depends on your system/GPU):

```bash
# Create a fresh env (conda or venv recommended)
conda create -n nmc python=3.10 -y
conda activate nmc

# Install PyTorch (choose the right command for your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other deps
pip install numpy scipy pandas scikit-learn matplotlib tqdm pillow
```

---

## Quickstart

```bash
python nmc.py \
  --dataset cifar10 \
  --data_dir ./data \
  --batch_size 256 \
  --epochs 50 \
  --lr 1e-3 \
  --lambda_inv 0.1 \
  --inv_mode endpoint \
  --cache_embeddings \
  --plot \
  --save_dir ./nmc_results
```

The first run downloads CIFAR-10 to `./data`.
Artifacts land in `./nmc_results` (see **Outputs** below).

---

## What the pipeline does

1. **Embedding**
   A frozen **ResNet-18** backbone extracts features; an optional **projection head** (default: enabled) maps to the embedding dimension and is trainable. Embeddings are L2-normalized.

2. **Learn a potential with DSM**
   The `NeuralPotential` (spectral-norm MLP) is trained by **Denoising Score Matching** with log-uniform noise scales.
   An **invariance loss** encourages consistency under augmentations:

   * `gradcos`: cosine between score gradients
   * `endpoint` (default): short score-following steps from two views should end nearby

3. **Seek modes**
   From each test embedding, run gradient ascent with **element-wise backtracking** to a nearby mode (endpoint).

4. **Deduplicate modes**
   **DBSCAN** clusters endpoints into unique modes; if none are found, falls back to **K-Means**.

5. **Persistence pruning**
   Build a k-NN graph over mode centers, track component births/deaths over descending φ, and keep:

   * all **immortal** components,
   * plus **mortal** ones with persistence ≥ τ (τ = chosen percentile, default 30).

6. **Assign points**
   Each sample is assigned to the nearest surviving mode; compute metrics and compare against **K-Means** baselines.

---

## CLI arguments

| Argument             |         Default | Description                                                   |
| -------------------- | --------------: | ------------------------------------------------------------- |
| `--dataset`          |       `cifar10` | Dataset name (CIFAR-10 supported)                             |
| `--data_dir`         |        `./data` | Dataset root                                                  |
| `--batch_size`       |           `256` | Train loader batch size                                       |
| `--input_size`       |           `224` | Image size for transforms                                     |
| `--num_workers`      |             `2` | DataLoader workers                                            |
| `--embedding_dim`    |           `128` | Projection/embedding dimension                                |
| `--potential_hidden` |           `512` | Potential MLP width                                           |
| `--potential_depth`  |             `4` | Potential MLP depth                                           |
| `--train_projection` |          `True` | Train projection head; use `--no_train_projection` to disable |
| `--epochs`           |            `50` | DSM training epochs                                           |
| `--lr`               |          `1e-3` | Learning rate (AdamW)                                         |
| `--lambda_inv`       |           `0.1` | Invariance loss weight                                        |
| `--inv_mode`         |      `endpoint` | `endpoint` or `gradcos`                                       |
| `--step_size`        |           `0.2` | Gradient ascent step size                                     |
| `--max_steps`        |            `20` | Max ascent steps                                              |
| `--tau_percentile`   |            `30` | Persistence threshold percentile (mortal components)          |
| `--cache_embeddings` |           *off* | Enable embedding cache for DSM training                       |
| `--amp`              |           *off* | Mixed precision (CUDA only)                                   |
| `--seed`             |            `42` | Random seed (deterministic cuDNN set)                         |
| `--save_dir`         | `./nmc_results` | Output directory                                              |
| `--plot`             |           *off* | Show loss curve                                               |

---

## Outputs

Saved to `--save_dir`:

* `nmc_model.pt` — encoder projection weights + potential weights + config
* `results.json` — summary:

  * `metrics`: `ACC`, `NMI`, `ARI`, optionally `Stability@Aug`, `Stability@Assign`, and `stability_threshold`
  * `n_clusters`, `loss_history`, `tau_value`, `n_immortals`
* Arrays (if available):

  * `predictions.npy` — cluster labels
  * `true_labels.npy` — ground truth labels
  * `endpoints.npy`, `endpoints_aug.npy`
* Baselines:

  * `comparison.csv` — NMC vs K-Means (auto-k and k=10 on CIFAR-10)
  * `latex_table.tex` — ready-to-paste LaTeX table

If `--plot` is set, a training loss curve is displayed during the run.

---

## Tips & knobs

* **Projection head**
  Enabled by default (`--train_projection`). Disable with `--no_train_projection` to use backbone features directly (dimension changes accordingly).

* **DBSCAN parameters**
  Currently `eps=0.5`, `min_samples=5`. For normalized embeddings, consider exploring `eps ∈ [0.2, 0.6]` depending on your encoder.

* **Caching embeddings**
  Use `--cache_embeddings` to precompute train embeddings and train DSM on them; the code auto-estimates `z_std` from the cache.

* **Stability metrics**

  * `Stability@Aug` uses the **median distance** between mode endpoints across two views as a fixed threshold.
  * `Stability@Assign` uses a Hungarian match on the contingency table between assignments from the two views.

* **Reproducibility**
  We set seeds and deterministic cuDNN, but full determinism can still vary with hardware/backend versions.

---

## Extending to new datasets

This release wires **CIFAR-10** with curated **weak/strong** transforms and a **two-view test loader**.
To add a dataset:

1. Implement transforms (strong/weak/test) mirroring `get_cifar10_transforms`.
2. Update `load_dataset` to return `(train_loader, test_loader, num_classes)`.
3. Update `make_two_view_test_loader` to ensure sample order matches `test_loader` (`shuffle=False`).

---

## Troubleshooting

* **CUDA OOM**
  Reduce `--batch_size` or `--input_size`; disable `--amp` if instability arises.

* **Torch / torchvision mismatch**
  Ensure versions are compatible with your CUDA. Reinstall using the official PyTorch index URL for your CUDA version.

* **No clusters found by DBSCAN**
  The code falls back to K-Means automatically. Consider lowering DBSCAN `eps` or enabling/adjusting the projection head.

* **Slow I/O**
  Increase `--num_workers` if your system supports it.

---

## Code map (high level)

* `FrozenEncoder` — ResNet-18 backbone (frozen) + optional projection head
* `NeuralPotential` — spectral-norm MLP; provides `gradient`, `gradient_batch`
* `DSMTrainer` — trains potential with DSM + invariance losses, AMP support, grad clipping
* `ModeSeeking` — gradient ascent with element-wise backtracking to reach endpoints
* `PersistencePruning` — computes 0-D persistence; prunes by immortals + percentile τ
* `ClusteringMetrics` — ACC (Hungarian), NMI, ARI, Stability\@Aug, Stability\@Assign
* `EmbeddingCache` — compute/save/load embeddings to speed up DSM training
* `NMCPipeline` — ties it together: train → cluster → evaluate → baselines & export

---

## Citation & Acknowledgements

* ResNet-18 backbone (torchvision)
* Ideas inspired by score-based modeling (DSM) and topological data analysis (0-D persistence).
  If you use this code in a publication, please consider citing related DSM/score-matching and TDA literature appropriate to your write-up.

---

## License

MIT 
