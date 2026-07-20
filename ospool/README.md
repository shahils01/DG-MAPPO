# OSPool SMACv2 training

`train_smac_v2_a100.sub` requests one NVIDIA A100 and launches the existing
`mat/scripts/train_smac_v2.sh` file. Submit it from the repository root, not
from the `ospool` directory.

## Required container

The submit file expects this image:

```text
/ospool/ap40/data/shahil.shaik/containers/dg-mappo-smacv2-v1.sif
```

It must contain:

- CUDA-enabled PyTorch and this repository's Python dependencies
- the official SMACv2 package
- StarCraft II for Linux
- `SMAC_Maps/32x32_flat.SC2Map`

By default the wrapper expects StarCraft II at `/opt/StarCraftII`. If the image
uses another location, set `SC2PATH` in the image or edit the default in
`run_train_smac_v2.sh`.

## Copy and submit

From the Mac:

```bash
rsync -av --exclude='.git' \
  "/Users/shahilshaik/Documents/DG-MAPPO/" \
  shahil.shaik@ap40.uw.osg-htc.org:~/DG-MAPPO/
```

On `ap40`:

```bash
cd ~/DG-MAPPO
chmod +x mat/scripts/train_smac_v2.sh ospool/run_train_smac_v2.sh
mkdir -p ospool/logs
condor_submit ospool/train_smac_v2_a100.sub
condor_watch_q
```

For an idle job, inspect its match analysis with:

```bash
condor_q -better-analyze JOB_ID
```

For a held job, inspect the reason and logs with:

```bash
condor_q -hold
tail -n 100 ospool/logs/train_smac_v2_JOB_ID.*
```

Successful and partial results are transferred back into
`mat/scripts/results`. A100-only matching can substantially increase queue
time. To permit other Ampere-or-newer GPUs, remove the `require_gpus` line but
keep `gpus_minimum_capability = 8.0`.

The existing launcher enables Weights & Biases. The job records W&B data in
offline mode unless `WANDB_API_KEY` is securely provided to the job; do not
commit or bake that API key into the repository or container image.
