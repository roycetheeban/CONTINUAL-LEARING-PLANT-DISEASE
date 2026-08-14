# Journal Scope & Coverage Plan
**Paper:** LEAFSENSE: An Edge-Deployed Continual Learning Framework for Autonomous Plant Disease Monitoring in AIoT Greenhouses
**Target venue:** IEEE Access (IEEEtran / ieeeaccess.cls, ~15 pages, ~25–27 references)
**Answers readme items 1 & 2.**

---

## 1. What This Journal COVERS vs. SKIPS (from the full AIoT flow)

### 1.1 Core contributions (COVER — full depth)

| # | Topic | Status of results | Role in paper |
|---|---|---|---|
| C1 | **Continual learning benchmark**: Category A (same-class streams) + Category B (5→7 class expansion) × 3 initialization cases (scratch / ImageNet / PlantVillage) × 5 methods (EWC, Replay, Isolation, Naive, Hybrid) | ✅ REAL — complete in CL chapter | Heart of the paper (Sections IV, VII) |
| C2 | **Layer-wise training strategy**: G1–G4 layer groups, progressive freezing, split-LR classifier for class expansion, SLCA-inspired LR decay | ✅ REAL | Methodological contribution (Section IV) |
| C3 | **EWC lambda tuning protocol**: λ grid search (1k–15k), per-cycle optimal λ (10,000 → 3,000), case-sensitivity analysis | ✅ REAL | Sub-study (Sections IV-E, VII-B) |
| C4 | **Decision-tree environmental risk model** (CO₂/temp/humidity/soil) | ✅ REAL — E1 done, DT 86.25% acc / 85.6 macro-F1 (⚠️ simulated sensor traces, disclosed §VI) | Fusion input (Section V) |
| C5 | **Label-level fusion** (per-class leaf counts × risk level → final diagnosis) | ✅ REAL — E2 done, 87.5% acc / 71.75 macro-F1 on 48 constructed episodes | The "missing part" of the 2 prior papers (Section V) |
| C6 | **On-device CL deployment loop**: confidence routing → retrain buffer / relabel queue → monthly EWC retrain → fixed-test-set metric gate → TensorRT engine swap | ✅ Design real; measurements done E3/E5 (⚠️ **laptop-GPU proxy, not Jetson**; ONNX Runtime substituted for TensorRT) | System contribution (Sections III, VII) |
| C7 | **Turmeric deployment validation**: best CL recipe (Case 2 + Replay) applied to the 5-class turmeric variant | ✅ REAL — E4 done, Replay 90.03% ± 3.54 vs. base 89.10% ± 2.35 (⚠️ feasibility check: 107-image test set, differences not statistically resolvable) | Ties benchmark → product (Section VII) |

### 1.2 Context only (COVER briefly, CITE prior papers)

- **3-stage edge pipeline** (YOLOv8n → YOLOv8n-seg → MobileNetV3-Small): 1 subsection + 1 figure, details cited to the IEEE conference paper (prior paper 2).
- **GAN occlusion recovery**: mentioned as an *optional* pipeline stage, cited to the GAN paper (prior paper 1). Not re-evaluated.
- **Hardware/capture strategy** (rotating 8×45° camera, 2 sets/day, sensor matching): 1 subsection — needed so the CL data engine makes sense.

### 1.3 SKIP entirely (or 1-line future-work mention)

| Skipped | Why |
|---|---|
| Cloud microservices architecture (API gateway, auth, ingestion, billing, multi-tenancy, storage cleanup) | Engineering, not research; dilutes the CL contribution |
| RAG chatbot, KB docs, daily insight provider, subscription tiers | Business layer; unvalidated — **future-work paragraph only** |
| Agentic AI system / LLM teacher auto-relabeling | Future-work paragraph (strong vision closer, weak main claim) |
| Mobile app, Streamlit dashboard details | **Fig. 9 removed** (2026-07-29) — no installed unit to photograph; deployment claims softened to conditional throughout |
| MQTT/ESP32/fleet management, Docker/K8s, MinIO etc. | Deployment ops, out of scope |
| Detailed YOLO training curves, pruning/quantization study | Already published in prior paper 2 — cite |
| GAN architecture variants comparison | Already published in prior paper 1 — cite |

**Framing rule (agreed):** CL methods are *benchmarked on PlantVillage tomato* (data-rich, reproducible, public); the winning recipe is *validated on the turmeric target crop* (Table XIV, E4 — done, framed as a feasibility check). This is honest and reviewer-proof: benchmark on public data, validate on the product crop.

---

## 2. Experiment Balance Needed Before Submission

Listed in priority order. The draft assumes these are DONE and holds placeholders for them.

| # | Experiment | Output needed for paper | Effort | Placeholder in draft |
|---|---|---|---|---|
| E1 | ✅ **DONE** — Decision-tree risk model, distilled from Magarey et al. (2005) infection model parameterised for turmeric (Gohel et al. 2022), on simulated greenhouse traces. 5-model comparison; DT deployed at 86.25% acc / 85.6 macro-F1, matching RF/GBM at 1/170th the size. Code: `experiments/part2_env_risk/` | Table IV (filled) + feature-importance discussion in §V-A | — | Table IV |
| E2 | ✅ **DONE, reported as prose** — Label-level fusion checked on 48 constructed episodes (8 scenario categories incl. a sensor-dropout case and a deliberately adversarial confound). ⚠️ **The results table was withdrawn 2026-08-05**: 7 of the 8 categories assign the outcome Φ is designed to produce, so the 87.5% aggregate was fixed by the scenario design before any episode ran, not measured (`07_CHANGES_MADE.md` §32). What is reported instead, and is genuinely earned: the single-modality baselines fail *structurally* in disjoint places, Φ has one *provable* blind spot, and sensor loss degrades it to vision-only *exactly*. Code: `experiments/part3_fusion_eval/` | §V-B prose + Fig. 5 (real worked example) | — | *(no table)* |
| E3 | 🟡 **PARTIAL** — retrain cycle (373.2 s, 155.6 MB VRAM), ONNX export (1.5 s), and the FP16 conversion gate (Δ ≈ 0: FP32 98.51% / FP16 98.59% / compiled 98.51%) measured, but on a **laptop RTX 4050, not the Jetson**; TensorRT unavailable in that environment so ONNX Runtime was substituted. Code: `experiments/part4_ondevice_proxy/`. **Real on-device measurement still outstanding.** | Table XIII (filled, disclosed as proxy) | MED | Table XIII |
| E4 | ✅ **DONE** (2026-07-29) — Case 2 + Replay on the turmeric 5-class set, 2 CL cycles, 3 seeds. **Replay 90.03% ± 3.54** vs. base 89.10% ± 2.35; the only method ending above its own baseline (EWC 88.79% ± 1.62, naive 87.85% ± 2.47). Splits rebuilt as `data/07_turmeric_5cls/` (test/val frozen from the shipped split so the existing phase-3 checkpoint stays comparable). ⚠️ Reported as a **feasibility check**: with n=107 no pairwise McNemar test reaches p<0.05. Code: `experiments/part5_turmeric_cl/` | Table XIV (filled) + 2 discussion paragraphs | — | Table XIV |
| E5 | 🟡 **PARTIAL** — classification latency measured (eager 10.2 ms/img, compiled 3.8 ms/img); detection (12.2 ms) and segmentation (5.0 ms) **cited from prior paper [21]**, not re-measured, per §1.2. End-to-end capture-set figure is a cross-source sum, not a unified trace. | 1 row/stage latency table | LOW | Table XIII (merged) |

> ~~E6 cassava batch-size study~~ — **dropped** per decision of 2026-07-23. Not referenced anywhere in the draft.

**Full step-by-step protocols for E1–E4 are in `06_EXPERIMENT_PROTOCOL.md` / `.docx`**, including a methodological review of the decision-tree plan (the synthetic-label circularity problem and how to avoid it).

**Not needed for this journal:** GAN re-validation on real farm data, chatbot/LLM experiments, cloud load tests, multi-plant variants.

---

## 3. File Map of This Folder

| File | Readme item | Content |
|---|---|---|
| `01_SCOPE_AND_COVERAGE.md` | 1, 2 | This file |
| `02_TITLES_AND_STRUCTURE.md` | 3, 4 | 5 candidate titles + full 15-page section plan |
| `03_LITERATURE_REVIEW.md` | 5 | Full literature review, citation-numbered to match the draft |
| `04_FULL_PAPER_LATEX.md` | 6 | Complete IEEE Access LaTeX draft with numbered placeholders |
| `05_FIGURE_GUIDE.md` | — | Figures 1–8 (Fig. 9 removed): tool, exact content spec, IEEE format rules |
| `06_EXPERIMENT_PROTOCOL.md` / `.docx` | — | E1–E4 full protocols + AS-RUN records + decision-tree plan review |
| `07_CHANGES_MADE.md` | — | Plan-vs-as-built record: every deviation, why, refs and data used |
| `08_PENDING_WORK.md` | — | Ordered remaining work to submission + standing limitations |
