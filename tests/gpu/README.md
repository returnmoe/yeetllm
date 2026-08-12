# GPU integration profiles

These tests intentionally run only on explicitly provisioned GPU runners. Put
the model configurations for the desired profile in a persistent directory and
run `tests/gpu/run.sh IMAGE CONFIG`.

The GitHub workflow deliberately requires a runner carrying the
`yeetllm-ephemeral` label. Register that runner with GitHub's `--ephemeral`
option so it accepts one job and is then discarded. Do not apply this label to
a persistent workstation or a general-purpose runner.

In repository settings, protect the `gpu-qualification` environment with
required reviewers, prevent self-approval, and protect the `development`
branch. The workflow is restricted to that branch and will otherwise skip.
These repository-side protection rules are required; naming an environment in
workflow YAML does not create reviewer rules automatically.

Workflow-dispatched image references are restricted to this repository's GHCR
package using commit-derived `dev-<40 hex>`, `sha-<40 hex>`, or immutable
`@sha256:<64 hex>` references. Configurations must be regular `.yaml`/`.yml`
files that resolve below `/opt/yeetllm-gpu-configs`; provision that directory on
the ephemeral runner with only reviewed, non-secret qualification configs. Run
the Actions service as a non-root user and make the config directory readable
but not writable by that account. Inputs enter shell steps through environment
variables and are validated before Docker is invoked. The harness always pulls
the selected reference before starting it.

Required release qualification profiles are:

- A: one base model on one GPU;
- B: one base model with TP=2;
- C: two distinct resident models on disjoint GPUs;
- D: one base with at least two static LoRAs;
- E: streaming chat through the router;
- F: one pre-quantized repository;
- G: listener audit—only SSH may bind non-loopback.

Profile C should set `YEETLLM_EXPECT_MODELS=model-a,model-b`. The harness
alternates requests repeatedly and records engine PIDs before and after; stable
PIDs demonstrate that neither engine reloaded. Profile D should include base and
both adapter IDs in `YEETLLM_EXPECT_MODELS`.
