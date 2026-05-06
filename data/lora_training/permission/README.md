# Permission markers

Slice #9 is **HITL** (Human In The Loop). Each LoRA Artifact training run requires a real onboarded artist's signed permission. This directory holds the durable record of that permission.

## Required record per onboarded artist

For every artist whose portfolio images become part of `data/lora_training/<artist_slug>/`, this directory must contain a file named `<artist_slug>.toml` with the following schema:

```toml
[artist]
slug = "kira-ink-bk"                                 # matches the data/lora_training/<slug>/ directory
display_name = "Kira (real onboarded artist)"
public_portfolio_url = "https://www.instagram.com/kira-ink-bk/"
contact = "kira@example.com"

[permission]
granted_on = "2026-05-06"
granted_by = "Kira Inks Brooklyn LLC"               # legal entity or natural person
record_method = "signed_pdf"                          # signed_pdf | signed_email | recorded_video | …
record_path = "permission/kira-ink-bk-signed.pdf"    # path under data/lora_training/ to the actual record
scope = "lora_training_only"                          # lora_training_only | full_use
revocation_terms = "Artist may revoke with 30 days written notice; LoRA Artifacts retired on revocation."

[training_set]
image_count = 28
attribution_required = true
credit_string = "Trained with permission of Kira Inks Brooklyn LLC; portfolio: https://www.instagram.com/kira-ink-bk/"
```

## What this artifact ships

The unsolicited POC contains **no real permission markers**: no permission has been obtained from any real artist. The `data/lora_training/` directory holds:

- This README (the schema).
- An empty `data/lora_training/artifacts.toml` placeholder (populated only when a real LoRA Artifact lands).

The `data/artists/artists.jsonl` file's 35 records are explicitly synthetic style-coverage entries (per slice #8); none correspond to real practitioners and none need permission.

When a real onboarded artist consents to a LoRA Artifact training run, this directory should grow a per-artist `<artist_slug>.toml` matching the schema above, and the corresponding `data/lora_training/<artist_slug>/` directory should hold the ~25-30 portfolio images plus per-image Provenance.
