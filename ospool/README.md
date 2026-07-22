# OSPool SMACv2 training

`train_smac_v2_a100.sub` requests one compatible NVIDIA GPU and launches the
existing `mat/scripts/train_smac_v2.sh` file. Submit it from the repository
root, not from the `ospool` directory.

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

Build the supplied definition from the repository root on an x86-64 Linux
machine with Apptainer fakeroot support:

```bash
cd ~/DG-MAPPO
OSPOOL_DATA=/ospool/ap40/data/shahil.shaik
mkdir -p \
  "$HOME/apptainer-tmp" \
  "$OSPOOL_DATA/apptainer-cache" \
  "$OSPOOL_DATA/containers"

export APPTAINER_TMPDIR="$HOME/apptainer-tmp"
export APPTAINER_CACHEDIR="$OSPOOL_DATA/apptainer-cache"
export TMPDIR="$HOME/apptainer-tmp"
export APPTAINER_NO_MOUNT=tmp

apptainer build --fakeroot \
  "$OSPOOL_DATA/containers/dg-mappo-smacv2-v1.sif" \
  ospool/dg-mappo-smacv2.def
```

The temporary writable image stays in the 40 GB home allocation because the
OSDF FUSE mount is `nodev` and may not support Apptainer builds. The reusable
OCI cache and final image stay in the larger `/ospool` allocation. Large files
downloaded during `%post` use `/opt/build-tmp` inside the build filesystem,
avoiding the access point's small host `/tmp` quota.

If `--fakeroot` is not enabled on the access point, stop after the error rather
than attempting a privileged build there. Build the same definition on another
x86-64 Linux system with root/fakeroot support, then copy the resulting `.sif`
to the path above.

Verify the completed image before submitting:

```bash
apptainer test \
  /ospool/ap40/data/shahil.shaik/containers/dg-mappo-smacv2-v1.sif
```

If the access point cannot run Apptainer locally, test the image on an OSPool
execution node instead:

```bash
cd ~/DG-MAPPO
chmod +x ospool/test_smacv2_container.sh
mkdir -p ospool/logs
condor_submit ospool/test_smacv2_container.sub
```

After it finishes, the `.out` log must end with
`CONTAINER SMOKE TEST PASSED` before submitting the full training job.

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

# Run this once. The ignored credential file is reused by later submissions.
umask 077
read -rsp "W&B API key: " WANDB_KEY
printf '\n'
printf '%s\n' "${WANDB_KEY}" > ospool/.wandb_api_key
unset WANDB_KEY
chmod 600 ospool/.wandb_api_key

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
`mat/scripts/results`. The capability range accepts CUDA-12.1-compatible
Ampere, Ada, and Hopper GPUs while excluding Blackwell GPUs that the current
PyTorch 2.4.1 image cannot execute on.

The launcher enables online Weights & Biases logging. The submit file transfers
the ignored `ospool/.wandb_api_key` credential directly into the job sandbox,
and the wrapper loads it automatically. Create that file once on AP40 as shown
above; do not commit it, upload it to OSDF, bake it into the container, or share
it in logs.
