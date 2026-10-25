# Historical delivery tools

These probes and the JPEG Web patch installer are retained for traceability.
They are not part of the supported startup path. **Do not reapply historical
patch installers to the current app:** they predate the dual model, evidence
outbox, and operator workflow fixes.

Use `python run.py`, `scripts/check_linq.py`, and the source under `gateways/`.
Old model weights, SDK ZIPs and backup directories remain local and ignored;
they are not deleted or overwritten by the repository cleanup.
