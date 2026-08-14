# Candidate Titles & Full Paper Structure
**Answers readme items 3 & 4.** Selected title: **#1 (LEAFSENSE)**. Target: IEEE Access, ~15 pages, ~27 references.

---

## 1. Five Candidate Titles (readme item 3)

1. ★ **SELECTED** — *LEAFSENSE: An Edge-Deployed Continual Learning Framework for Autonomous Plant Disease Monitoring in AIoT Greenhouses*
   — System-branded, deployment-led; positions the CL benchmark inside a named, deployable product.
2. *Continual Learning at the Edge: A Self-Improving AIoT Pipeline for Plant Disease Detection with Multimodal Label-Level Fusion*
   — Balances the CL benchmark and the fusion contribution equally.
3. *Benchmarking Regularization, Replay, and Isolation: Catastrophic-Forgetting-Aware Plant Disease Detection for Edge AIoT Deployment*
   — Method-study-led; strongest if reviewers value the 25-run benchmark most.
4. *A Self-Evolving Lightweight AIoT System for Greenhouse Disease Monitoring: On-Device Continual Learning with Environmental Fusion*
   — System-led without brand name.
5. *From Static Models to Lifelong Learners: An Edge-Optimized Continual Learning Architecture for Multimodal Plant Disease Diagnosis*
   — Narrative/positioning title; catchy but least specific.

---

## 2. Main Topics Covered (readme item 4) — 15-page budget

| Sec. | Title | Pages | Content & key assets |
|---|---|---|---|
| I | Introduction | 1.5 | AIoT agriculture context; edge constraints; model staleness/drift problem; catastrophic forgetting; LEAFSENSE concept; **5-bullet contribution list** |
| II | Related Work | 1.5 | A) Edge-optimized plant disease detection; B) Continual learning (regularization / replay / isolation); C) Multimodal & sensor fusion in agriculture; D) **Gap**: no deployed on-device CL system for plant disease with fusion |
| III | LEAFSENSE System Architecture | 2 | A) Hardware & capture (Orin Nano, 8×45° rotation, 2 sets/day); B) 3-stage inference pipeline recap (cite both prior papers; GAN as optional stage); C) Confidence routing & data engine (retrain buffer / relabel queue); D) On-device retraining loop with metric gate & TensorRT runtime. **Figs 1–3** |
| IV | Continual Learning Methodology | 2.5 | A) Problem formulation (CatA = domain-incremental, CatB = class-incremental); B) Initialization cases 1–3; C) Methods: EWC (Fisher math), Replay (50/50 buffer), Isolation (adapters/branch), Naive, Hybrid; D) **Layer groups G1–G4 + split-LR head** (Table: LR schedule); E) λ tuning protocol. **Fig 4** |
| V | Environmental Sensing & Label-Level Fusion | 1.5 | A) Sensor suite & timestamp matching; B) Decision-tree risk model; C) Label-level fusion rule (counts × risk → final state). **Fig 5** |
| VI | Experimental Setup | 1 | PlantVillage tomato splits (leaf-grouped, fixed 15% test); turmeric dataset; metrics (acc, macro-F1, old-retention, new-acquisition, VRAM, time); implementation details |
| VII | Results & Discussion | 3.5 | A) CatA results ★real; B) λ tuning ★real; C) CatB results ★real; D) Isolation failure analysis ★real; E) Hybrid analysis ★real; F) Resources ★real; G) DT + fusion ⏳; H) Turmeric deployment ⏳; I) On-device cycle & latency ⏳. **Tables I–XI, Figs 6–9** |
| VIII | Practitioner Recommendations | 0.5 | Recipe cards: which method for which scenario (Replay default; EWC for frequent updates; never Isolation for new classes) |
| IX | Conclusion & Future Work | 0.5 | Findings; agentic AI / LLM-teacher / GAN maturation / multi-crop as future work |
| — | References | 0.75 | ~27 IEEE-style, incl. both prior papers |

**Figure plan (placeholders in draft):**
- Fig. 1 — LEAFSENSE system architecture (edge tier; cloud band removed 2026-08-05)
- Fig. 2 — 3-stage inference pipeline with confidence routing
- Fig. 3 — On-device monthly CL cycle with metric gate (flowchart)
- Fig. 4 — MobileNetV3-Small layer groups G1–G4 & split-LR head
- Fig. 5 — Label-level fusion example chain
- Fig. 6 — CatA accuracy across cycles (grouped bars, 3 cases × 4 methods)
- Fig. 7 — λ sensitivity curve (C1/C2 accuracy vs λ)
- Fig. 8 — CatB stability–plasticity scatter (old retention vs new acquisition)
- Fig. 9 — Greenhouse deployment photo + dashboard screenshot

**Table plan:** I dataset; II LR schedule; III CatA results ★; IV λ grid ★; V λ impact per case ★; VI CatB results ★; VII resources ★; VIII decision tree ⏳; IX fusion eval ⏳; X on-device cycle/latency ⏳; XI turmeric validation ⏳.
(★ = real numbers from CL chapter; ⏳ = placeholder pending experiments E1–E5.)
