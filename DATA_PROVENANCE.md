# DATA_PROVENANCE.md

Auditable trail of every record in the **Artist Portfolio Index**, **Famous Tattoos Corpus**, and **Knowledge Corpus**, plus the **LoRA Artifact** training set. Per IMPLEMENTATION_PLAN.md, this document is **append-only across slices** until issue #11 finalizes the structure.

Each section lists records with their five Provenance fields (`source_url`, `curator`, `capture_date`, `synthetic`, `permission`).

## Conventions

- `synthetic: true` records are AI-authored content produced for this POC; no third-party rights are implicated.
- `synthetic: false` records cite a real external source. The `source_url` is the canonical record; the `permission` field describes the basis on which this artifact uses it.
- Synthetic-content records carry the `permission: synthetic-content-tattd-studio-poc` marker.
- All dates are ISO-8601.

---

## Knowledge Corpus

Authored under [slice #3](https://github.com/techjays-prem/tattd-studio/issues/3) as 136 chunks distributed across five areas. Stored under `data/knowledge/<area>.md` with one chunk per `<!-- CHUNK -->`-delimited block. Every chunk has YAML frontmatter that mirrors the fields below.

Aggregate Provenance: all 136 Knowledge Corpus chunks are AI-authored for this POC and carry `synthetic: true` and `permission: synthetic-content-tattd-studio-poc`. The `source_url` points at the documentary placeholder `https://www.tattd-studio.example/<area>/<slug>` to indicate the chunk is part of this artifact's own knowledge surface, not a re-publication.

### Per-area record counts

| Area | Count | Spec target |
|---|---|---|
| `taxonomy` | 71 | 50–100 |
| `placement` | 25 | 20–30 |
| `aftercare` | 13 | 10–15 |
| `ip` | 12 | 10–15 |
| `cultural` | 15 | 10–20 |
| **Total** | **136** | ~150 |

### Per-record Provenance

| chunk_id | source_url | curator | capture_date | synthetic | permission |
|---|---|---|---|---|---|
| taxonomy-fineline | https://www.tattd-studio.example/styles/fineline | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-american-traditional | https://www.tattd-studio.example/styles/american-traditional | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-neo-traditional | https://www.tattd-studio.example/styles/neo-traditional | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-japanese-irezumi | https://www.tattd-studio.example/styles/japanese-irezumi | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-blackwork | https://www.tattd-studio.example/styles/blackwork | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-dotwork | https://www.tattd-studio.example/styles/dotwork | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-geometric | https://www.tattd-studio.example/styles/geometric | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-watercolor | https://www.tattd-studio.example/styles/watercolor | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-realism-bw | https://www.tattd-studio.example/styles/realism-bw | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| taxonomy-realism-color | https://www.tattd-studio.example/styles/realism-color | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| (… 61 more taxonomy chunks; full record list lives in the per-chunk frontmatter under `data/knowledge/taxonomy.md`) | | | | | |
| placement-inner-forearm … placement-knee | https://www.tattd-studio.example/placement/* | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| aftercare-saniderm … aftercare-pregnancy | https://www.tattd-studio.example/aftercare/* | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| ip-portrait-likeness … ip-minor-consent | https://www.tattd-studio.example/ip/* | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |
| cultural-japanese-irezumi … cultural-language-script-validation | https://www.tattd-studio.example/cultural/* | tattd-studio-dev | 2026-05-06 | true | synthetic-content-tattd-studio-poc |

The full record list is the YAML frontmatter of each chunk inside `data/knowledge/<area>.md`. The table above documents the aggregate Provenance pattern; the per-chunk frontmatter is the authoritative record.

---

## Artist Portfolio Index

Populated in [slice #8](https://github.com/techjays-prem/tattd-studio/issues/8). 35 records under `data/artists/artists.jsonl` (one JSON object per line) covering the style space exercised by the Knowledge Corpus and Famous Tattoos Corpus.

**Important POC posture.** The implementation plan's spec calls for 10–20 *real onboarded artists with explicit permission*. This artifact is unsolicited and the developer cannot ethically claim those permissions. **All 35 records are explicitly marked synthetic style-coverage entries** with `permission_marker: synthetic-content-tattd-studio-poc` and `synthetic: true`. Real onboarded-artist records are deferred to slice #9 (HITL) — the same gating that holds back the LoRA Artifact training set — and slice #11 (production deployment) where actual artist permission can be obtained through proper onboarding.

The Plagiarism Critic and Two-Stage Matcher therefore index the *style description* of each record rather than any third-party imagery; the names are synthetic (Kira, Miguel, Yuki, …) so they cannot be misread as referring to real practitioners.

| Coverage axis | Sample slugs |
|---|---|
| fineline / minimalist | kira-fineline-bk, seth-micro, vega-stick-poke, isaac-floral, moss-single-line |
| traditional / neo-traditional | miguel-traditional-bk, yuki-japanese-bk |
| realism (color & b&g) | ana-realism-color, jorge-realism-bw, kale-portraiture, amari-animal-portrait |
| blackwork / dotwork / geometric | sara-blackwork, lin-dotwork, alex-geometric, nora-mandala, iris-pointillism |
| watercolor / illustrative | noor-watercolor, rae-illustrative, rin-sketchy |
| trash polka / biomechanical | tomas-trash-polka, rina-biomech |
| script / ornamental | jed-script, ren-ornamental, ada-engraving |
| cultural / religious | rio-religious, sage-celtic, asta-norse, ravi-yantra, tashi-tibetan, elif-tessellation, haruko-monmon-cat |
| trend / contemporary | luna-graffiti, ezra-anime, remy-cyber, kai-hand-poke |

The full per-record Provenance lives in `data/artists/artists.jsonl`.

---

## Famous Tattoos Corpus

Populated in [slice #7](https://github.com/techjays-prem/tattd-studio/issues/7) as the secondary Plagiarism Critic reference. 50 iconic and celebrity tattoo records under `data/famous_tattoos/famous.md`, each chunk separated by `<!-- CHUNK -->` and carrying the same Provenance frontmatter as Knowledge Corpus chunks plus an `artist` field that the Plagiarism Critic surfaces in `top_match_artist`.

Aggregate Provenance: all 50 records are AI-authored descriptive entries about real-world tattoos and practitioners (Mike Tyson facial moko, Rihanna falconry glove, Sailor Jerry flash, Horiyoshi III, Don Ed Hardy, Scott Campbell, Bang Bang, Dr. Woo, Yomico Moreno, Carlos Torres, …). The `source_url` field uses the `https://www.tattd-studio.example/famous/<slug>` placeholder pattern to indicate the chunk is part of this artifact's own descriptive surface, not a re-publication of any third-party text. All records carry `synthetic: true` and `permission: synthetic-content-tattd-studio-poc`.

The artist names are factual references to real practitioners and famous bearers — used in the descriptive text the way a journalistic style guide would name them. No third-party imagery is ingested; the Plagiarism Critic indexes the textual descriptions for cosine-similarity matching on prompt content.

| chunk_id (sample) | artist | source_url pattern |
|---|---|---|
| famous-mike-tyson-facial-moko | S. Victor Whitmill | https://www.tattd-studio.example/famous/tyson-facial-moko |
| famous-rihanna-falconry-glove | Bang Bang | https://www.tattd-studio.example/famous/rihanna-falconry |
| famous-angelina-jolie-bengal-tiger | Sompong Kanphai | https://www.tattd-studio.example/famous/jolie-tiger |
| famous-sailor-jerry-traditional-flash-1 | Norman "Sailor Jerry" Collins | https://www.tattd-studio.example/famous/sj-death-before-dishonor |
| famous-horiyoshi-iii-bodysuit | Horiyoshi III | https://www.tattd-studio.example/famous/horiyoshi-iii-bodysuit |
| (… 45 more; full per-record Provenance lives in `data/famous_tattoos/famous.md` frontmatter) | | |

---

## LoRA Artifact training set

Populated in [slice #9](https://github.com/techjays-prem/tattd-studio/issues/9). HITL: requires explicit permission from one onboarded artist + 25–30 of their portfolio images per the slice spec.

> *To be populated in slice #9, gated on explicit onboarded-artist permission.*
