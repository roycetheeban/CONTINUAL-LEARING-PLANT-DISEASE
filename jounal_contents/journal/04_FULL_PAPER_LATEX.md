# LEAFSENSE — Full Journal Draft (IEEE Access, LaTeX)

**Template:** Ships as `\documentclass[journal]{IEEEtran}` so it **compiles on any Overleaf/TeX Live project with no extra files** (`ieeeaccess.cls` is NOT in TeX Live and must be uploaded from the IEEE Author Center — that missing file is the usual "No PDF" cause). A figure-fallback block draws placeholder boxes for figures that don't exist yet, so it builds before the figures are made. For the **final IEEE Access submission**, switch the documentclass to `ieeeaccess`, restore the IEEE Access front matter, and delete the fallback block — see the swap comments in the `.tex` and the WORKING NOTES.
**Compile the ready-made file:** `leafsense.tex` (same content as the fence below). Don't re-extract unless you want to.
**Length target:** ~15 pages · **References:** 35 (all verified) · **Figures:** 8 · **Result tables:** all filled

> **Ref status: COMPLETE — verified 2026-07-29.** All **35** references audited; **20 of 21 DOIs resolve**; **IEEE format compliant**, no formatting changes needed. ⚠️ **Reference [5] was fabricated** (DOI 404s, and its claimed pages 9876–9888 fall in a gap between real *IEEE IoT J.* v9n12 articles) — removed and replaced with Khan et al., *Front. Plant Sci.* 14:1308528. It was **inherited from the prior conference paper**, so that bibliography is worth auditing too. Also corrected: [11] Dec. 2024 → **Jan. 2025**; [17] iCaRL pp. → **5533–5542**; [21] filled with the real ICIIS 2026 entry (**pp. 395–400, doi `10.1109/ICIIS69028.2026.11450780`**); [22] filled with ICDTSA 2025 (**Kilinochchi, May 2025, pp. 7–16, no DOI** — confirmed absent from Crossref); [27] real Mendeley author lists via the **DataCite** API; [31] FedAvg and [33] FixMatch `%% VERIFY` markers cleared. See `07_CHANGES_MADE.md` §26–27.

> **E1 status: DONE.** The decision-tree environmental risk model is real, not a placeholder. Code: `experiments/part2_env_risk/code/` (`simulate_sensors.py`, `infection_risk_labels.py`, `features.py`, `train_env_risk.py`); config: `configs/part2_env_risk.yaml`; artifacts: `experiments/part2_env_risk/outputs/` (incl. deployed `decision_tree.pkl`). Labels are distilled from a validated infection model (Magarey et al. 2005) parameterised for turmeric (Gohel et al. 2022) — not self-defined thresholds, avoiding circularity. Result: deployed Decision Tree reaches **86.25% accuracy / 85.6 macro-F1**, matching or beating Random Forest and Gradient Boosting at a fraction of the size/latency; `hours_rh_above_85` + `rh_mean` account for 79% of feature importance, confirming the humidity/leaf-wetness mechanism. Data is simulated (disclosed in §VI). See `06_EXPERIMENT_PROTOCOL.md` Part 1 for the as-run design notes.

> **E2 status: DONE.** The label-level fusion evaluation is real. Code: `experiments/part3_fusion_eval/code/` (`vision_infer.py`, `build_episodes.py`, `fusion_eval.py`); config: `configs/part3_fusion_eval.yaml`; artifacts: `experiments/part3_fusion_eval/outputs/`. The vision branch is the real trained Case 2 + Replay (cycle 2) checkpoint run on held-out tomato test images; the risk branch is real E1 test-period output; the 48 evaluation episodes are constructed (day-by-day sequencing, disclosed in §VI). Result: label-level fusion reaches **87.5% accuracy / 71.75 macro-F1** vs. 60.4%/29.6 (vision-only) and 35.4%/20.6 (sensor-only), matches sensor-only's lower false-alarm rate, degrades gracefully to vision-only when the sensor fails, and is honestly imperfect — it fails on one deliberately adversarial category where a coincidental high-humidity reading cannot be distinguished from a real one. Fig. 5 is built from real episode data. See `experiments/part3_fusion_eval/README.md` for full methodology.

> **E3/E5 status: FILLED, but on PROXY HARDWARE — read the caveats.** Table X now holds real measurements, but **not** the Jetson Orin Nano numbers the protocol specifies (no Jetson available). Code: `experiments/part4_ondevice_proxy/code/` (`time_retrain_cycle.py`, `export_and_compile.py`, `bench_accuracy_latency.py`); config: `configs/part4_ondevice_proxy.yaml`; artifacts: `experiments/part4_ondevice_proxy/outputs/`. **Measured on a laptop NVIDIA RTX 4050 (CUDA 12.1):** replay retrain cycle **373.2 s / 155.6 MB peak VRAM** (reproduced 98.42% test acc exactly, confirming a faithful re-timing of the real cycle-2 config — the real checkpoint E2 depends on was *not* overwritten); ONNX export 1.5 s / 5.8 MB; **FP16 conversion gate Δ ≈ 0** (FP32 98.51% / PyTorch-FP16 98.59% / compiled 98.51%); classification latency eager 10.2 ms/img → **compiled 3.8 ms/img (2.7×)**. **Three disclosed deviations:** (1) laptop GPU, not Jetson — real on-device run still outstanding; (2) **TensorRT could not be installed** (pip `tensorrt`/`tensorrt-cu12` pull `nvidia-cuda-runtime-cu13`, whose wheel fails against this machine's CUDA 12.1 torch build), so **ONNX Runtime GPU** was substituted as the compiled-graph path and is labeled as such everywhere; (3) detection (12.2 ms) and segmentation (5.0 ms) are **cited from prior paper [21]**, not re-measured, per `01_SCOPE_AND_COVERAGE.md` §1.2. Two findings kept rather than smoothed over: the FP16 delta is one image out of 1,137 (i.e. noise — so the gate's argument is *verifying* equivalence, not that conversion is lossy), and eager FP16 is *slower* than FP32 for a model this small. Full gap inventory: `experiments/part4_ondevice_proxy/README.md` §7.

> **E4 status: DONE (2026-07-29).** Turmeric CL validation is real. Code: `experiments/part5_turmeric_cl/code/` (`train_base_turmeric.py`, `train_cl_turmeric.py`, `analyze_results.py`); config: `configs/part5_turmeric_cl.yaml`; splits: `data/07_turmeric_5cls/` (built by `data/06_scripts/splitters/split_turmeric_5cls.py` — **test/val frozen from the shipped split** so the existing phase-3 checkpoint stays comparable; only `train/` re-split 50/25/25 → 417/210/207). Case 2 + Replay, 2 CL cycles, 3 seeds: **Replay 90.03% ± 3.54** vs. base 89.10% ± 2.35 — the only method ending above its own baseline (EWC 88.79% ± 1.62, naive 87.85% ± 2.47). Buffer rule changed to `min(40, 50% of initial_train)` and stated in the paper (tomato's "15%, min 100" is unsatisfiable at 76–92 images/class). ⚠️ Reported as a **feasibility check, not a ranking**: with n=107 no pairwise McNemar test reaches p<0.05. Scope narrowed to **Replay + a naive reference** for the same reason. See `06_EXPERIMENT_PROTOCOL.md` PART 4 for the as-run record.

> **Fig. 9 status: REMOVED (2026-07-29).** No unit is installed, so the deployment photograph cannot be produced, and a dashboard-only figure does not carry the deployment claim. The figure was **never referenced from the body text** (`\ref{fig:deploy}` appeared zero times), so nothing depended on it. §VI was reworded from "pending replacement with in-situ sensor logs from *the installed* turmeric greenhouse unit" — which presupposed an installation that does not exist — to "No unit is currently installed, so no in-situ readings exist… to be revisited once a turmeric greenhouse unit is deployed." Every other deployment claim was audited. **The paper is now consistently a methods-and-architecture contribution, not a deployed-system one** — coherent with E1 being simulated, E2 constructed and E3 proxy-measured.

## How to use this file
1. Compile `leafsense.tex` directly, or copy the fenced LaTeX below into your `.tex`. It starts at `\documentclass` — never prepend anything before it (that was what put the notes on page 1).
2. Every figure/table needing your input is marked `%% >>> PLACEHOLDER Pn <<<` with a description of exactly what to produce.
3. `★` in comments = number already real (from the CL results chapter). `⏳` = fill after experiments E1–E5 (see `01_SCOPE_AND_COVERAGE.md`).
4. **Verify all references against IEEE Xplore before submission.**

### Placeholder index
| ID | Type | What to supply |
|---|---|---|
| P1 | Fig. 1 | LEAFSENSE system architecture diagram | ✅ DONE — `figures/fig1_architecture.pdf` |
| P2 | Fig. 2 | Inference pipeline + confidence routing | ✅ DONE — `figures/fig2_pipeline.pdf` |
| P3 | Fig. 3 | Monthly on-device CL cycle flowchart | ✅ DONE — `figures/fig3_cl_cycle.pdf` |
| P4 | Fig. 4 | MobileNetV3-Small layer groups G1–G4 | ✅ DONE — `figures/fig4_layer_groups.pdf` |
| P5 | Fig. 5 | Label-level fusion example | ✅ DONE — `figures/fig5_fusion.pdf` |
| P6 | Fig. 6 | CatA accuracy grouped bar chart | ✅ DONE — `figures/fig6_cata.pdf` |
| P7 | Fig. 7 | λ sensitivity curve | ✅ DONE — `figures/fig7_lambda.pdf` |
| P8 | Fig. 8 | CatB stability–plasticity scatter | ✅ DONE — `figures/fig8_catb_scatter.pdf` |
| P9 | ~~Fig. 9~~ | Deployment photo + dashboard screenshot | ❌ **REMOVED 2026-07-29** — no installed unit; never `\ref`'d |
| P10 | Table IV | Decision-tree risk results | ✅ DONE — E1 real (86.25% acc, DT deployed) |
| P11 | Table V | Fusion vs image-only | ✅ DONE — E2 real (87.5% acc fusion) |
| P12 | Table XIII | On-device cycle + latency | 🟡 FILLED — E3/E5 real, but **laptop-GPU proxy, not Jetson** |
| P13 | Table XIV | Turmeric CL validation | ✅ DONE — E4 real (Replay 90.03% ± 3.54, 3 seeds, feasibility framing) |

---

<!-- ============================ BEGIN LATEX ============================ -->

```latex
%% ============================================================
%% LEAFSENSE -- IEEE Access submission
%% Compile with pdflatex (needs ieeeaccess.cls from the IEEE Author Center).
%% Figures go in ./figures/  (see 05_FIGURE_GUIDE.md).
%% Placeholders '---' in tables are filled from experiments E1-E4
%% (see 06_EXPERIMENT_PROTOCOL). This file is self-contained -- do NOT
%% prepend anything before \documentclass.
%% ============================================================

%% Compiles anywhere (Overleaf/TeX Live) with NO extra files.
%% For the FINAL IEEE Access submission, switch to the IEEE Access class:
%%   - change the next line to:  \documentclass{ieeeaccess}
%%   - restore the ieeeaccess front matter (see WORKING NOTES at end of file)
%%   - remove the figure-fallback block below.
\documentclass[journal]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{xparse}

%% --- float placement tuning (float-heavy paper: 8 figures + 14 tables) ---
\graphicspath{{figures/}}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.72}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}

%% --- FIGURE FALLBACK -------------------------------------------------------
%% If a figure file is missing, draw a labelled placeholder box instead of
%% aborting the whole compile. Figs 1-4, 6-8 are FINALISED (upload the 7 PDFs to
%% ./figures/ and they render). Only Fig 5 (needs experiment E1) and Fig 9 (needs
%% hardware photo) are still missing, so KEEP this block until those two exist,
%% then DELETE it.
\let\ORIGincludegraphics\includegraphics
\RenewDocumentCommand{\includegraphics}{O{}m}{%
  \IfFileExists{#2}%
    {\ORIGincludegraphics[#1]{#2}}%
    {\IfFileExists{figures/#2}%
      {\ORIGincludegraphics[#1]{figures/#2}}%
      {\fbox{\parbox[c][2.6cm][c]{.86\linewidth}{\centering\ttfamily
        [figure not yet created]\\#2}}}}%
}
%% ---------------------------------------------------------------------------

\begin{document}

\title{LEAFSENSE: An Edge-Deployed Continual Learning Framework for
Autonomous Plant Disease Monitoring in AIoT Greenhouses}

\author{Roycetheeban~R.,~Mathushanth~A.~M.,~and~Mukunthan~Tharmakulasingam%
\thanks{The authors are with the Department of Electrical and Electronic
Engineering, Faculty of Engineering, University of Jaffna, Jaffna, Sri Lanka
(e-mail: 2021e123@eng.jfn.ac.lk; 2021e015@eng.jfn.ac.lk;
mukunthan@eng.jfn.ac.lk).}%
% \thanks{Funding statement, if any.}%
}

\markboth{IEEE Access (Preprint Draft), 2026}%
{Roycetheeban \MakeLowercase{\textit{et al.}}: LEAFSENSE: An Edge-Deployed Continual Learning Framework}

\maketitle

\begin{abstract}
Plant disease detection models deployed on agricultural edge devices are typically
static: trained once, shipped, and left to degrade as new disease presentations,
seasonal variation and site-specific conditions drift away from the training
distribution. Retraining in the cloud is not an option for the many greenhouses that
lack reliable connectivity, and naive on-device fine-tuning induces catastrophic
forgetting. This paper presents LEAFSENSE, an offline-first AIoT framework in which a
Jetson Orin Nano executes the complete perception-to-adaptation loop on device:
rotating image capture, a three-stage detection--segmentation--classification pipeline,
environmental sensing, label-level fusion, and metric-gated continual retraining. The
core of the paper is a controlled continual learning (CL) study comprising 25
complete experimental runs on the PlantVillage tomato subset, spanning two incremental
scenarios (same-class streaming and 5$\rightarrow$7 class expansion), three
initialization strategies (from scratch, ImageNet, and in-domain PlantVillage
pretraining) and five methods (Elastic Weight Consolidation, experience replay,
parameter isolation, naive fine-tuning, and a hybrid). We introduce a layer-wise
training strategy that partitions MobileNetV3-Small into four functional groups with
progressive freezing and a split learning-rate classifier head for class expansion.
Experience replay over an ImageNet-initialized backbone is the strongest configuration,
reaching 98.25\% average accuracy in same-class streaming and simultaneously 98.07\%
new-class accuracy with 97.19\% old-class retention under class expansion; EWC attains
98.21\% at 3.6$\times$ lower training cost, making it the preferred choice for frequent
update cycles. Parameter isolation, adequate for same-class streams, fails
catastrophically on new classes (6.38\% new-class accuracy with an in-domain backbone),
and a hybrid EWC$+$replay scheme underperforms replay alone while costing 37--40\% more
training time. All resulting models remain within 5.82\,MB, and the retraining cycle
completes within the memory envelope of the target device, demonstrating that
autonomous, self-improving disease monitoring is feasible entirely at the edge.
\end{abstract}

\begin{IEEEkeywords}
Agricultural AIoT, catastrophic forgetting, continual learning, edge computing,
elastic weight consolidation, experience replay, Jetson Orin Nano, MobileNetV3,
multimodal fusion, plant disease detection, precision agriculture.
\end{IEEEkeywords}

\IEEEpeerreviewmaketitle

%% ====================================================================
\section{Introduction}
\label{sec:intro}
%% ====================================================================

\PARstart{P}{lant} disease is a first-order constraint on global food security,
and the losses fall hardest on smallholder producers in tropical regions where
climate variability accelerates pathogen pressure while diagnostic expertise is
scarce. Turmeric (\textit{Curcuma longa}), the target crop of the system described
here, suffers yield losses of 20--45\% from pathogens such as
\textit{Colletotrichum capsici} and \textit{Taphrina maculans} depending on
environmental conditions and management practice \cite{ref21}. Timely, accurate,
low-cost diagnosis is therefore of direct economic consequence.

Deep learning has largely solved the \emph{static} version of this problem.
Convolutional classifiers trained on the PlantVillage corpus \cite{ref1} exceed
99\% accuracy under laboratory conditions \cite{ref2}, and a sustained line of work
on lightweight architectures \cite{ref6}, structured pruning \cite{ref7} and
post-training quantization \cite{ref8} has brought such models within the compute
and memory envelope of embedded accelerators. Our own prior work established a
modular edge pipeline combining YOLOv8n detection, YOLOv8n-seg segmentation and a
pruned MobileNetV3-Small classifier in a 17.6\,MB footprint \cite{ref21}, together
with a lightweight GAN for recovering occluded leaf regions \cite{ref22}.

What remains unsolved is the \emph{dynamic} version. A model shipped inside a
greenhouse device encounters a distribution that moves: new cultivars, seasonal
lighting, unfamiliar pathogens, and site-specific presentations absent from any
public dataset. Three constraints make this hard to address in practice.
First, connectivity cannot be assumed --- the deployments that most need automated
diagnosis are frequently those with the least reliable internet, so cloud retraining
is structurally unavailable. Second, on-device fine-tuning on newly arriving data
overwrites previously learned representations, the failure mode known as
catastrophic forgetting \cite{ref13}. Third, no labelled stream exists in the field:
a system that intends to learn continuously must first \emph{manufacture} its own
supervision, and must do so without poisoning itself with its own errors.

The continual learning literature offers three method families ---
regularization \cite{ref15}, \cite{ref16}, replay \cite{ref17}, \cite{ref18} and
parameter isolation \cite{ref19}, \cite{ref20} --- but applied plant-disease studies
evaluate them almost exclusively offline, on server-class hardware, with an assumed
labelled stream, and typically under a single initialization. Consequently the
practitioner facing an actual edge deployment cannot answer three operational
questions: which method to select for a given incremental scenario, how much the
choice of pretraining changes that answer, and what a retraining cycle actually costs
on the device.

This paper answers those questions in the context of LEAFSENSE, an offline-first
AIoT plant-monitoring product built on the NVIDIA Jetson Orin Nano. The device
performs rotating image capture, runs the three-stage inference pipeline locally,
correlates per-class leaf counts with CO\textsubscript{2}, temperature, humidity and
soil-moisture readings through label-level fusion, routes its own predictions by
confidence into an auto-labelled retraining buffer or a human-reviewed relabel queue,
and retrains itself on a monthly cadence behind a deployment gate that admits a new
model only if it improves on a permanently held-out test set.

\noindent The specific contributions are:

\begin{enumerate}
\item \textbf{A controlled 25-run continual learning benchmark} for plant disease
classification spanning two incremental scenarios (same-class streaming and
5$\rightarrow$7 class expansion), three initialization strategies (scratch,
ImageNet, in-domain PlantVillage) and five methods (EWC, experience replay,
parameter isolation, naive fine-tuning, hybrid EWC$+$replay), with initialization
treated as an explicit experimental factor rather than an uncontrolled confound.
\item \textbf{A layer-wise continual training strategy} that partitions
MobileNetV3-Small into four functional groups by feature type and forgetting
sensitivity, applies progressive freezing and per-group learning-rate decay, and
introduces a \emph{split learning-rate classifier head} that protects old-class
output neurons while permitting fast new-class acquisition during expansion.
\item \textbf{A per-cycle EWC regularization-strength protocol}, showing that the
optimal penalty coefficient decreases across cycles ($\lambda=10{,}000
\rightarrow 3{,}000$) and that its impact scales with the domain specificity of the
backbone, contributing up to $+1.14\%$ accuracy for an in-domain pretrained model.
\item \textbf{An autonomous edge data engine and deployment gate}: confidence-based
routing that never trains on unreviewed low-confidence self-labels, and a metric gate
that evaluates the retrained candidate through the same TensorRT FP16 runtime that
will serve it, so that the acceptance decision measures deployment reality.
\item \textbf{An integrated multimodal edge system} combining the continual vision
branch with an interpretable decision-tree environmental risk model through
label-level fusion, validated on the turmeric target crop on a Jetson Orin Nano.
\end{enumerate}

The remainder of the paper is organized as follows. Section~\ref{sec:related}
reviews related work. Section~\ref{sec:system} describes the LEAFSENSE architecture.
Section~\ref{sec:cl} details the continual learning methodology.
Section~\ref{sec:fusion} presents the environmental branch and fusion layer.
Section~\ref{sec:setup} specifies the experimental setup, Section~\ref{sec:results}
reports and discusses results, Section~\ref{sec:reco} distils deployment
recommendations, and Section~\ref{sec:conclusion} concludes.

%% ====================================================================
\section{Related Work}
\label{sec:related}
%% ====================================================================

\subsection{Edge-Optimized Plant Disease Detection}

PlantVillage \cite{ref1} has served as the field's reference corpus since Mohanty
\textit{et al.} reported greater than 99\% accuracy with deep CNNs \cite{ref2}.
High-capacity backbones such as ResNet \cite{ref3} and EfficientNet \cite{ref4}
dominate accuracy tables but occupy 25--500\,MB and assume server-class compute,
which is incompatible with field hardware \cite{ref5}. MobileNetV3 \cite{ref6}
addresses this through hardware-aware architecture search, inverted residuals and
squeeze-and-excite blocks, matching far larger models at 10--20$\times$ fewer
parameters. Compression compounds the gain: structured pruning removes 50--90\% of
parameters with limited degradation \cite{ref7} and post-training quantization adds
a further 2--4$\times$ reduction without retraining \cite{ref8}. For detection,
YOLOv8n provides an anchor-free head at 3.2\,M parameters \cite{ref9}, and YOLO
variants have been applied widely to agricultural targets \cite{ref10},
\cite{ref11}, with architectural enhancements improving mAP on plant disease
specifically \cite{ref11}. Modular pipelines that chain detection, segmentation and
classification allow each stage to be optimized or disabled independently according
to the available budget \cite{ref12}.

Our prior work sits in this line. A lightweight modular pipeline achieved
mAP@0.5 $=0.632$ for detection, 99.5\% pixel accuracy and 0.915 mIoU for
segmentation, and 89.81\% five-class turmeric classification from a pruned,
INT8-quantized MobileNetV3-Small, with a 17.6\,MB total footprint \cite{ref21}.
A companion study addressed occlusion, showing that a hybrid GAN combining attention
with VGG perceptual loss reaches 23.16\,dB PSNR and 0.6296 SSIM on synthetically
occluded leaves, outperforming standard and attention-only baselines \cite{ref22}.
Both papers conclude by identifying on-device continual learning as the principal
piece of unfinished work --- the gap this paper closes.

\subsection{Continual Learning and Catastrophic Forgetting}

Sequential training on non-stationary data overwrites earlier representations
\cite{ref13}; mitigations fall into three families \cite{ref14}.
\emph{Regularization} methods penalize changes to parameters deemed important for
past tasks: EWC \cite{ref15} derives importance from the Fisher Information Matrix,
while Synaptic Intelligence \cite{ref16} accumulates it online. These store no data
and add little compute, making them attractive at the edge, but their protection
strength hinges on a penalty coefficient that applied work frequently leaves fixed
and unexplained. \emph{Replay} methods interleave a buffer of past exemplars with
the incoming stream; iCaRL \cite{ref17} remains the standard class-incremental
baseline, and later analysis showed that much of the class-incremental damage is
concentrated in classifier-head bias rather than in the backbone \cite{ref18},
which motivates output-layer-specific countermeasures. \emph{Parameter isolation}
dedicates disjoint capacity per task --- progressive columns with lateral
connections \cite{ref19}, or iterative prune-and-freeze allocation \cite{ref20} ---
guaranteeing zero forgetting at the cost of monotonic growth and, as our results
show, severely constrained plasticity. Orthogonally, SLCA \cite{ref23} demonstrated
that slowing representation-layer learning rates while aligning the classifier
largely resolves progressive overfitting in continual fine-tuning; our layer-wise
schedule follows this principle. Cutting across these families, knowledge distillation
constrains the updated model's outputs toward those of its predecessor: Learning
without Forgetting \cite{ref29} achieves this without storing old data, and
distillation objectives are routinely fused into replay methods such as
iCaRL \cite{ref17}.

Continual learning for plant disease has begun to attract direct attention. Li
\emph{et al.} \cite{ref24} evaluate rehearsal-based class-incremental approaches for
plant disease classification and confirm that replaying stored exemplars materially
mitigates old-class forgetting --- a conclusion our CatB results independently
corroborate at edge scale. Such evaluations nonetheless remain offline and
server-side: they neither model the retraining budget of an embedded device nor
specify the mechanism that would supply a labelled stream in the field, and they
seldom control for initialization quality, so the contribution of pretraining is
conflated with that of the CL algorithm.

\subsection{Multimodal Sensing and Fusion in Agricultural AIoT}

Disease expression is environmentally driven --- sustained leaf wetness, humidity
and temperature bands govern fungal infection pressure --- so vision-only diagnosis
discards a cheap and predictive signal \cite{ref25}. Fusion may occur at feature
level, decision level, or label level. For edge deployment we argue label-level
fusion is preferable: branches remain independently trainable and updatable, the
combination rule is interpretable to the end user, and a sensor fault degrades one
branch instead of corrupting a joint embedding. That environmental time series carry
genuine predictive signal for pest and disease risk has been demonstrated directly
\cite{ref26}. For the environmental branch we favour an interpretable model over a
higher-capacity one: agronomic thresholds are already expressed as rules in practice,
and in a decision-support setting where a grower acts on the output, a model whose
reasoning path can be inspected is preferable to a post-hoc explanation of an opaque
one \cite{ref28}.
Existing agricultural fusion systems, however, generally assume connectivity and a
static vision branch, leaving unaddressed the behaviour of a fusion layer whose
vision model is itself changing under continual updates.

\subsection{On-Device and Federated Model Adaptation}

The assumption that model updates occur on a server is increasingly challenged by
on-device training research. Memory, not computation, is the binding constraint:
Lin \emph{et al.} \cite{ref30} demonstrate that a classifier can be trained within
256\,KB of working memory through quantization-aware sparse back-propagation,
establishing that gradient-based adaptation on microcontroller-class hardware is
feasible rather than merely aspirational. The Jetson Orin Nano targeted here is far
more capable, but the same principle governs the design --- our results in
Section~\ref{sec:results} report peak \emph{retraining} memory precisely because it,
and not model size, decides whether a cycle completes on device.

Federated learning \cite{ref31} is the dominant paradigm for decentralized updates:
many always-connected clients compute local gradients that a coordinator aggregates
into a shared global model, keeping raw data on each device. LEAFSENSE shares the
data-locality motivation but differs structurally in three respects. First, each unit
adapts to \emph{its own} site rather than contributing to a common model, so there is
no aggregation step and no cross-device synchronization. Second, no coordinator or
always-on channel is assumed --- connectivity is used opportunistically for relabel
exchange, not for training. Third, acceptance of an update is decided \emph{locally}
by the metric gate of Section~\ref{sec:system} against a fixed test set, rather than
centrally by an aggregation server. The on-device cost of a full adaptation cycle,
which the federated literature abstracts into an idealized ``local update'' step, is
exactly what this paper measures.

\subsection{Self-Training and Confidence-Based Data Selection}

The autonomous data engine of Section~\ref{sec:system} rests on self-training, in
which a model's own predictions become supervision. Pseudo-labelling assigns
high-confidence predictions as training targets; Noisy Student \cite{ref32} shows
that iterative self-training from a confident teacher improves even strong ImageNet
classifiers, and FixMatch \cite{ref33} establishes that retaining only predictions
above a confidence threshold is the mechanism that keeps pseudo-labelling stable. The
characteristic failure mode is confirmation bias: training on incorrect self-labels
turns the model's own errors into its supervision and compounds them across rounds ---
the same drift the confidence router is designed to prevent.

LEAFSENSE inherits the confidence-threshold safeguard but adds an asymmetry absent
from batch semi-supervised methods. Standard pseudo-labelling simply \emph{discards}
samples below the threshold; LEAFSENSE instead routes them to a review queue
(Algorithm~\ref{alg:routing}), so the hardest and most informative samples are not
lost but returned with trustworthy labels at the next cycle. This matters more in a
continual setting than in a one-shot semi-supervised pass: over many cycles the
low-confidence region is exactly where new disease presentations and distribution
shift first appear, and a system that discarded it would be blind to the change it
most needs to learn. Confidence-based selection is therefore not merely a
data-cleaning heuristic here but the mechanism that decides what the device teaches
itself versus what it defers to review.

\subsection{Research Gap}

Three gaps follow. \textbf{(i)} CL methods for plant disease are benchmarked but not
deployed: no reported study measures the on-device cost of a retraining cycle or
gates deployment on measured post-conversion accuracy. \textbf{(ii)} Initialization
is an uncontrolled confound, so published method rankings conflate transfer quality
with algorithmic merit. \textbf{(iii)} The data engine and the fusion layer are
absent: prior work assumes a labelled stream materializes and treats the vision
output as the final product. LEAFSENSE addresses all three within a single validated
edge system. Table~\ref{tab:positioning} positions this work against representative
continual-learning and edge plant-disease studies across the five axes that define
the gap.

\begin{table*}[!t]
\caption{Positioning of LEAFSENSE Against Representative Continual-Learning and Edge Plant-Disease Work}
\label{tab:positioning}
\centering
\small
\begin{tabular}{@{}llccccc@{}}
\toprule
\textbf{Work} & \textbf{CL family} & \textbf{Class-incremental} & \textbf{Init.\ controlled} & \textbf{On-device retrain \& cost} & \textbf{Env.\ fusion} & \textbf{Deployment gate} \\
\midrule
iCaRL \cite{ref17}            & Replay             & \checkmark & --         & --         & -- & -- \\
PackNet \cite{ref20}          & Isolation          & \checkmark & --         & --         & -- & -- \\
SLCA \cite{ref23}             & Reg.\ + pretrain   & \checkmark & partial    & --         & -- & -- \\
Li \emph{et al.} \cite{ref24} & Replay             & \checkmark & --         & --         & -- & -- \\
Prior pipeline \cite{ref21}   & none (static)      & --         & --         & --         & -- & -- \\
\textbf{LEAFSENSE (ours)}     & \textbf{5 compared} & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark \\
\bottomrule
\multicolumn{7}{@{}l}{\footnotesize \checkmark~= addressed; --~= not addressed; partial~= partially. ``Init.\ controlled'' = pretraining treated as an explicit experimental factor.}
\end{tabular}
\end{table*}

%% ====================================================================
\section{LEAFSENSE System Architecture}
\label{sec:system}
%% ====================================================================

LEAFSENSE is designed under an \emph{offline-first} principle: every function
required to diagnose, track and improve is executed on the device, and connectivity
--- when present --- is used only to exchange relabelled samples and maintenance
updates. Fig.~\ref{fig:arch} shows the complete architecture.

%% Fig 1 --- FINALISED: figures/fig1_architecture.pdf  (source: figures/src/fig1_architecture.py)
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_architecture.pdf}
\caption{LEAFSENSE system architecture. All perception, fusion and adaptation
execute on the edge device; the cloud tier is optional and used only for relabel
exchange and maintenance updates.}
\label{fig:arch}
\end{figure*}

\subsection{Hardware and Capture Strategy}

The compute unit is an NVIDIA Jetson Orin Nano. A motorized camera mount rotates
through $360^\circ$ in $45^\circ$ steps, producing a \emph{capture set} of eight
images per acquisition. Two sets are acquired daily --- a morning cycle (Cycle~A)
and an evening cycle (Cycle~B) --- yielding 16 images per day, supplemented by an
on-demand capture triggered from the dashboard. Environmental sensors sample
CO\textsubscript{2}, temperature, humidity and soil moisture, and each reading is
timestamp-matched to the nearest capture so that every count carries its
environmental context.

Adjacent $45^\circ$ views may observe the same leaf twice. This is accepted by
design: overlap is minimized at installation, and the system's primary signal is the
\emph{change} in per-class counts rather than their absolute value, so a small
constant multiplicity bias cancels under differencing. Two sets per day yield
intra-day differentiation (Cycle~A versus Cycle~B) in addition to day-over-day trend
lines per class.

\subsection{Three-Stage Inference Pipeline}

Each captured image passes through the pipeline summarized in
Fig.~\ref{fig:pipeline}, whose detection, segmentation and classification stages
were characterized in detail in our prior work \cite{ref21}. The image is
letterboxed to $640\times640$ with aspect ratio preserved; YOLOv8n localizes
individual leaves; crops smaller than $124\times124$ pixels are discarded as
providing insufficient evidence for reliable downstream diagnosis; YOLOv8n-seg
produces a per-leaf mask with the background filled black; the masked crop is
resized to $200\times200$ and zero-padded to $224\times224$; and MobileNetV3-Small
assigns a disease class and confidence. The lightweight GAN occlusion-recovery
module of \cite{ref22} is available as an optional stage between segmentation and
classification but ships disabled by default, pending validation on real
greenhouse imagery rather than synthetic occlusions. The set-level output is a
per-class leaf count vector.

%% Fig 2 --- FINALISED: figures/fig2_pipeline.pdf  (source: figures/src/fig2_pipeline.py)
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_pipeline.pdf}
\caption{Three-stage inference pipeline with confidence routing. The GAN
occlusion-recovery stage is optional and disabled by default.}
\label{fig:pipeline}
\end{figure}

\subsection{Autonomous Data Engine: Confidence Routing}

A system that learns continuously in the field must generate its own supervision.
LEAFSENSE routes every classified leaf by prediction confidence. Predictions above a
high-confidence threshold $\tau_{\text{high}}$ are auto-labelled with the predicted
class and written to an on-device \emph{retrain buffer}, organized per class.
Predictions below a low-confidence threshold $\tau_{\text{low}}$ are written instead
to a \emph{relabel queue} and exported as a compact review sheet; they are never
trained on while unreviewed. When connectivity becomes available the queue is
uploaded, corrected externally, and the corrected labels return to the device to join
the \emph{next} retraining cycle.

This asymmetry is deliberate. Training on unreviewed low-confidence self-labels is
the principal mechanism by which self-training pipelines drift: the model's errors
become its own supervision and compound across cycles. Restricting auto-labelling to
the high-confidence regime bounds the label noise entering the buffer, while the
review queue preserves exactly the hard, informative samples that would otherwise be
discarded. A planned extension replaces the human reviewer with an LLM teacher that
labels queued crops in batch, which is discussed in Section~\ref{sec:conclusion}.
Algorithm~\ref{alg:routing} states the routing procedure.

\begin{algorithm}[!t]
\caption{Confidence-Based Data Routing}
\label{alg:routing}
\begin{algorithmic}[1]
\REQUIRE Classified leaf crop $x$, softmax confidence $c$, thresholds
$\tau_{\text{high}} > \tau_{\text{low}}$, retrain buffer $\mathcal{R}$, relabel
queue $\mathcal{Q}$
\STATE $\hat{y} \leftarrow \arg\max f_\theta(x)$ \COMMENT{predicted class}
\IF{$c \geq \tau_{\text{high}}$}
    \STATE $\mathcal{R} \leftarrow \mathcal{R} \cup \{(x, \hat{y})\}$
    \COMMENT{auto-label; trusted}
\ELSIF{$c < \tau_{\text{low}}$}
    \STATE $\mathcal{Q} \leftarrow \mathcal{Q} \cup \{x\}$
    \COMMENT{queue for external review; never trained unreviewed}
\ELSE
    \STATE discard $x$ \COMMENT{ambiguous mid-band; excluded from both}
\ENDIF
\STATE \textbf{on connectivity:} upload $\mathcal{Q}$; retrieve corrected labels;
merge into $\mathcal{R}$ for the \emph{next} cycle
\end{algorithmic}
\end{algorithm}

\subsection{On-Device Retraining Cycle and Deployment Gate}

Retraining runs on the device approximately monthly, or on demand from the
dashboard, and is illustrated in Fig.~\ref{fig:cycle}. Only the classifier is
retrained; detection, segmentation and the optional GAN are frozen and never updated
in the field. The classifier is fine-tuned on the retrain buffer together with any
returned relabelled samples, using the continual learning method and layer-wise
schedule established in Section~\ref{sec:cl}, with per-epoch checkpointing so that a
power interruption --- a routine occurrence in greenhouse settings --- costs at most
one epoch.

The resulting candidate is exported to ONNX and compiled to a TensorRT FP16 engine,
then evaluated on the \emph{fixed} test set held out since initial training and never
altered. Evaluating the candidate as a compiled engine rather than as a PyTorch model
is a deliberate design decision: FP16 conversion can shift accuracy, and a gate that
measures the training-precision model does not measure what will actually serve.
The candidate is deployed by atomic file swap only if it outperforms the incumbent on
this fixed set; otherwise the incumbent is retained and the accumulated data is
carried forward to the next cycle. The previous engine is preserved to permit
instantaneous rollback. Algorithm~\ref{alg:cycle} states the complete cycle.

\begin{algorithm}[!t]
\caption{On-Device Adapt--Verify--Gate Retraining Cycle (monthly)}
\label{alg:cycle}
\begin{algorithmic}[1]
\REQUIRE Deployed engine $E_{\text{cur}}$, classifier weights $\theta$, retrain
buffer $\mathcal{R}$, returned relabelled set $\mathcal{L}$, fixed test set
$\mathcal{T}$, Fisher information $F$ (EWC)
\ENSURE Engine deployed for the next month
\STATE $\mathcal{D} \leftarrow \mathcal{R} \cup \mathcal{L}$
\COMMENT{high-confidence auto-labels $+$ reviewed labels}
\FOR{each epoch}
    \STATE update $\theta$ on $\mathcal{D}$ using the layer-wise schedule
    (G1--G2 frozen; EWC/replay on G3--G4)
    \STATE checkpoint $\theta$ \COMMENT{power-cut safe}
\ENDFOR
\STATE $\theta_{\text{cand}} \leftarrow \theta$; export $\theta_{\text{cand}}$ to ONNX
\STATE $E_{\text{cand}} \leftarrow \textsc{BuildTensorRT}(\text{ONNX},\ \text{FP16})$
\STATE $a_{\text{cand}} \leftarrow \textsc{Eval}(E_{\text{cand}}, \mathcal{T})$;\quad
       $a_{\text{cur}} \leftarrow \textsc{Eval}(E_{\text{cur}}, \mathcal{T})$
\STATE \COMMENT{evaluate the ENGINE that will serve --- not the PyTorch model}
\IF{$a_{\text{cand}} > a_{\text{cur}}$}
    \STATE atomically swap $E_{\text{cur}} \leftarrow E_{\text{cand}}$; keep old
    engine for rollback
\ELSE
    \STATE keep $E_{\text{cur}}$; carry $\mathcal{D}$ to the next cycle
\ENDIF
\end{algorithmic}
\end{algorithm}

%% Fig 3 --- FINALISED: figures/fig3_cl_cycle.pdf  (source: figures/src/fig3_cl_cycle.py)
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig3_cl_cycle.pdf}
\caption{On-device monthly continual learning cycle. The candidate model is
evaluated as a compiled TensorRT engine on a permanently fixed test set, and is
deployed only if it improves on the incumbent.}
\label{fig:cycle}
\end{figure}

\subsection{Runtime Strategy}

In production every neural component serves as a TensorRT FP16 engine. ONNX is used
strictly as an interchange format --- engines are hardware- and library-version
specific and are therefore always built on the target device rather than shipped ---
and PyTorch weights are retained only for the classifier, as the trainable state from
which each monthly cycle starts, together with the accumulated Fisher information.
The decision-tree risk model executes on the CPU. The engine rebuild costs
approximately one to two minutes once per month, which is amortized against roughly
a twofold inference speed-up on every capture of every day.

%% ====================================================================
\section{Continual Learning Methodology}
\label{sec:cl}
%% ====================================================================

\subsection{Problem Formulation}

Let $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ denote the classifier with
parameters $\theta$. The device observes a sequence of data streams
$\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_T$ arriving at successive
retraining cycles, where at cycle $t$ only $\mathcal{D}_t$ (plus a bounded buffer)
is available --- earlier streams have been discarded under the device's storage
policy. We evaluate two regimes.

\textbf{Category A (CatA) --- same-class streaming.} The label space is fixed,
$\mathcal{Y}_t = \mathcal{Y}$ for all $t$, and successive cycles deliver new
observations of the same five tomato classes. This corresponds to the ordinary
operating condition of a deployed device: accumulating site-specific data for a fixed
disease inventory. It is a domain-incremental problem, and the question is whether
adaptation to the new stream degrades performance on the original distribution.

\textbf{Category B (CatB) --- new-class expansion.} Two previously unseen classes
(T6, \textit{Septoria} leaf spot; T7, spider mites) are introduced across cycles,
expanding the classifier from five to seven outputs, so
$\mathcal{Y}_t \subset \mathcal{Y}_{t+1}$. This is the class-incremental problem and
corresponds to a new pathogen appearing at a site. Success requires two things
simultaneously: \emph{plasticity}, measured as accuracy on the new classes, and
\emph{stability}, measured as retained accuracy on T1--T5. Reporting either alone is
uninformative, since a frozen model trivially maximizes stability and an
unconstrained model maximizes plasticity.

\subsection{Initialization Cases}

Because transfer quality plausibly interacts with continual learning behaviour, we
treat initialization as an explicit factor and evaluate every method under three
conditions.

\begin{itemize}
\item \textbf{Case 1 --- Scratch.} Random initialization, no pretrained features.
Establishes what CL methods achieve without transfer.
\item \textbf{Case 2 --- ImageNet.} MobileNetV3-Small initialized from ImageNet
weights, providing broad general-purpose visual features.
\item \textbf{Case 3 --- In-domain PlantVillage.} The backbone is pretrained on the
26 non-tomato PlantVillage classes, encoding plant-domain features (venation,
lesion morphology, leaf texture) before continual learning begins.
\end{itemize}

\subsection{Continual Learning Methods}

\subsubsection{Elastic Weight Consolidation}
EWC \cite{ref15} constrains parameters in proportion to their estimated importance
for previously learned knowledge. Importance is the diagonal of the Fisher
Information Matrix, estimated over the buffer $\mathcal{B}$ of old-class samples:

\begin{equation}
F_i = \frac{1}{|\mathcal{B}|}\sum_{(x,y)\in\mathcal{B}}
\left(\frac{\partial \log p_\theta(y \mid x)}{\partial \theta_i}\right)^{\!2}.
\label{eq:fisher}
\end{equation}

Training at cycle $t$ then minimizes

\begin{equation}
\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\text{CE}}(\theta)
+ \frac{\lambda}{2}\sum_{i} F_i \left(\theta_i - \theta_{i}^{*}\right)^{2},
\label{eq:ewc}
\end{equation}

where $\mathcal{L}_{\text{CE}}$ is the cross-entropy loss on the current stream,
$\theta^{*}$ are the parameters after the previous cycle, and $\lambda$ sets the
regularization strength. The penalty is applied only to trainable (non-frozen)
parameters, i.e.\ groups G3 and G4 of Section~\ref{subsec:layers}; applying it to
permanently frozen early layers would dilute the Fisher normalization across
parameters that never move. EWC stores no images, requiring only one forward-backward
pass over the buffer to estimate $F$, which is the source of its efficiency advantage.

\subsubsection{Experience Replay}
A stratified buffer of old-class exemplars is sampled from the initial training pool
only --- never from validation or test data --- at 15\% per class with a floor of 100
images per class to protect small classes, giving approximately 517 images across the
five base classes. During each cycle the buffer is mixed 50/50 with new stream data
in every batch, and the standard cross-entropy loss is used with no penalty term.
After CatB Cycle~1 the buffer is updated with T6 and T7 samples so that Cycle~2
maintains gradient signal for all seven classes; without this update Cycle~2 would
replay only T1--T5 and risk forgetting the newly acquired classes.

\subsubsection{Parameter Isolation}
All existing weights are hard-frozen and only newly added capacity is trained. For
CatA this is a lightweight bottleneck adapter (bottleneck width 24, approximately
18{,}000 parameters, 1.2\% of the backbone). For CatB it is a separate branch of
three inverted-residual blocks trained from scratch for the new classes. Old-class
behaviour is mathematically unchanged, so forgetting is zero by construction.

\subsubsection{Naive Fine-Tuning and Hybrid}
Naive fine-tuning applies no protection and serves as a lower bound on retention and,
because it is free to use the full data history, an upper reference on raw accuracy;
it is not a valid CL method under the storage constraint. The hybrid method applies
the EWC penalty and the replay buffer in the same training loop, and is evaluated for
CatB to test whether the two mechanisms cooperate.

\subsection{Layer-Wise Training Strategy}
\label{subsec:layers}

A central methodological claim of this work is that continual learning performance on
a compact backbone depends as much on \emph{which parameters are permitted to move}
as on the choice of CL algorithm. We partition MobileNetV3-Small into four functional
groups according to the type of feature encoded and its sensitivity to forgetting, as
listed in Table~\ref{tab:groups} and illustrated in Fig.~\ref{fig:groups}.

%% Fig 4 --- FINALISED: figures/fig4_layer_groups.pdf  (source: figures/src/fig4_layer_groups.py)
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig4_layer_groups.pdf}
\caption{Functional layer groups of MobileNetV3-Small and the split learning-rate
classifier head used for new-class expansion.}
\label{fig:groups}
\end{figure}

\begin{table}[!t]
\caption{Functional Layer Groups and Continual Learning Sensitivity}
\label{tab:groups}
\centering
\begin{tabular}{@{}llp{2.9cm}@{}}
\toprule
\textbf{Group} & \textbf{Block} & \textbf{Feature type / CL role} \\
\midrule
G1 Early & features[0--3] & Edges, colour gradients; universal --- always frozen \\
G2 Mid & features[4--8] & Venation, surface texture; plant-domain --- frozen (CatA) \\
G3 Late & features[9--12] & Lesion shape, discolouration, class margins; \textbf{primary forgetting site} \\
G4 Head & classifier[0--3] & Class probabilities; always task-specific, split-LR in CatB \\
\bottomrule
\end{tabular}
\end{table}

G1 encodes features that transfer identically across all visual domains and is
permanently frozen; updating it consumes plasticity budget without benefit and
dilutes the Fisher normalization in \eqref{eq:fisher}. G3 is where
disease-discriminative information concentrates and is therefore the principal site
of catastrophic forgetting and the primary target of both the EWC penalty and the
replay gradient. Table~\ref{tab:lr} gives the complete schedule; all CL methods share
identical learning rates so that observed differences are attributable to the methods
rather than to their tuning.

\begin{table}[!t]
\caption{Layer-Wise Learning Rate Schedule Across Training Stages}
\label{tab:lr}
\centering
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Stage} & \textbf{G1} & \textbf{G2} & \textbf{G3} & \textbf{G4 Head} \\
\midrule
Case 1 --- Scratch            & 1e-3 & 1e-3 & 1e-3 & 1e-3 \\
Case 2 --- Phase i            & Froz. & Froz. & Froz. & 8e-4 \\
Case 2 --- Phase ii           & Froz. & Froz. & 5e-5 & 5e-4 \\
Case 3A --- Pretrain (26 cls) & 1e-3 & 1e-3 & 1e-3 & 1e-3 \\
Case 3B --- Phase i           & Froz. & Froz. & 5e-5 & 5e-4 \\
Case 3B --- Phase ii          & Froz. & 1e-5 & 5e-5 & 5e-4 \\
\midrule
CL CatA --- Cycle 1           & Froz. & Froz. & 1e-4 & 5e-4 \\
CL CatA --- Cycle 2           & Froz. & Froz. & 5e-5 & 2e-4 \\
CL CatB --- Cycle 1           & Froz. & 5e-5 & 2e-4 & 1e-4 / 1e-3$^{\dagger}$ \\
CL CatB --- Cycle 2           & Froz. & 1e-5 & 1e-4 & 5e-5 / 5e-4$^{\dagger}$ \\
Isolation (CatA)              & Froz. & Froz. & Froz. & Froz.$^{\ddagger}$ \\
\bottomrule
\multicolumn{5}{@{}l}{\footnotesize $^{\dagger}$ Split-LR classifier: old-class neurons / new-class neurons.}\\
\multicolumn{5}{@{}l}{\footnotesize $^{\ddagger}$ Adapter only: LR 1e-3 (Cycle 1), 5e-4 (Cycle 2).}
\end{tabular}
\end{table}

\textbf{Split learning-rate head.} When the head expands from five to seven outputs,
applying a uniform learning rate allows the large gradient signal driving new-class
acquisition to perturb the weights of established old-class neurons --- precisely the
output-layer bias mechanism identified in \cite{ref18}. We therefore assign old-class
neurons a learning rate an order of magnitude below that of new-class neurons
(Table~\ref{tab:lr}), permitting rapid acquisition of T6 and T7 while the
representation supporting T1--T5 is held approximately fixed. The expansion adds only
2{,}050 parameters, a 0.13\% increase, confirming that class expansion is
inconsequential for the edge storage budget.

\subsection{Regularization Strength Protocol}

The EWC coefficient $\lambda$ trades stability against plasticity: too small yields
insufficient protection, too large prevents adaptation. Rather than fixing a single
value, we tune $\lambda$ \emph{per cycle} by exhaustive grid search over
$\{1{,}000,\,3{,}000,\,5{,}000,\,8{,}000,\,10{,}000,\,15{,}000\}$ on Case 2, and
transfer the resulting schedule to Cases 1 and 3. Results are reported in
Section~\ref{subsec:lambda}.

%% ====================================================================
\section{Environmental Sensing and Label-Level Fusion}
\label{sec:fusion}
%% ====================================================================

\subsection{Sensor Branch}

Four environmental variables --- air temperature, relative humidity, soil moisture
and CO\textsubscript{2} concentration --- are sampled continuously and matched by
timestamp to each capture set. The first three bear a direct causal relation to
fungal infection pressure: temperature governs pathogen development rate, sustained
high humidity drives infection, and elevated soil moisture both stresses the rhizome
and raises canopy humidity \cite{ref25}. CO\textsubscript{2} is included as a
\emph{proxy for ventilation} rather than as a causal driver: in an enclosed
greenhouse, accumulating CO\textsubscript{2} indicates stagnant air, which prolongs
leaf wetness duration and therefore raises fungal risk indirectly. Because infection
depends on \emph{sustained} rather than instantaneous conditions, the model operates
on aggregate features computed over a trailing multi-day window --- notably the
number of hours spent above the humidity threshold and within the
pathogen-favourable temperature band --- aligned to each capture set.

\textbf{Training labels.} Turmeric's principal foliar diseases, leaf blotch
(\textit{Taphrina maculans}) and leaf spot (\textit{Colletotrichum} sp.), are
moisture-driven fungal pathogens whose infection pressure is governed by leaf-wetness
duration and concurrent temperature \cite{ref35}. Rather than defining risk
thresholds ourselves, training labels are distilled from the generic foliar-fungal
infection model of Magarey \emph{et al.} \cite{ref34}, which computes an infection
severity from leaf-wetness hours (relative humidity above a wetness threshold) and a
temperature-response function peaking at a pathogen-specific optimum, accumulated
over a trailing window. Because turmeric-specific cardinal temperatures are not
tabulated in the literature, the model is parameterized with the mesophilic,
moisture-favoured range reported for \textit{T.\ maculans} and \textit{Colletotrichum}
under Indian growing conditions \cite{ref35}, and the Low/Medium/High cut-points on
accumulated severity are calibrated on the training period only, avoiding test-set
leakage. Crucially, the downstream models are never given leaf-wetness hours or
accumulated severity directly --- only raw-ish 24-hour and multi-day aggregates
(mean/min/max temperature, hours in the favourable band, humidity mean/trend, hours
above a humidity threshold, soil moisture, CO\textsubscript{2}) --- so recovering the
risk label from sensor aggregates is a genuine learning problem rather than a
restatement of the oracle. \textbf{The environmental traces themselves are
simulated} (diurnal and recurring seasonal cycles, autocorrelated drift, sensor
noise, and dropout; Section~\ref{sec:setup} states this explicitly), pending
deployment-site data collection.

A decision tree is chosen for production over a higher-capacity alternative for
three reasons: its inference cost is negligible alongside three neural stages; its
decision path is directly inspectable, so an agronomist can audit why a risk state
was raised \cite{ref28}; and its rule structure aligns with how agronomic infection
thresholds are already expressed in the plant pathology literature \cite{ref25},
\cite{ref26}. Table~\ref{tab:dt} compares it against a logistic regression, a
random forest and a gradient-boosted ensemble, and against a majority-class
reference, on a chronological train/test split (test = the final 80 of 390
capture-aligned windows, never seen during threshold calibration or training).

\begin{table}[!t]
\caption{Environmental Risk Model Comparison (Test Set, 3-Class)}
\label{tab:dt}
\centering
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Model} & \textbf{Acc.} & \textbf{Macro F1} & \textbf{Latency} & \textbf{Size} \\
\midrule
Majority baseline    & 43.75\% & 20.3 & 0.7\,\textmu s   & 0.7\,KB   \\
Logistic Regression  & 81.25\% & 78.7 & 16.7\,\textmu s  & 1.9\,KB   \\
\textbf{Decision Tree $\star$} & \textbf{86.25\%} & \textbf{85.6} & \textbf{9.4\,\textmu s} & \textbf{3.9\,KB} \\
Random Forest         & 85.00\% & 84.3 & 683\,\textmu s   & 666\,KB   \\
Gradient Boosting     & 85.00\% & 83.7 & 33\,\textmu s    & 708\,KB   \\
\bottomrule
\multicolumn{5}{@{}l}{\footnotesize $\star$ Deployed model. Depth 5, 15 leaves. 390 windows (280 train / 80 test).}
\end{tabular}
\end{table}

The deployed decision tree (depth 5, 15 leaves) matches or exceeds every
higher-capacity alternative --- including the 666\,KB random forest, at
$1.7\times10^{-4}$ its size and $73\times$ lower per-sample latency --- so no accuracy
is traded for interpretability here. Per class it reaches F1 of 0.91 (Low), 0.78
(Medium) and 0.88 (High) on the held-out chronological block. Ranked feature
importances confirm the intended mechanism: \texttt{hours\_rh\_above\_85} and
\texttt{rh\_mean} jointly account for 79\% of importance, with humidity trend and
soil moisture a distant third and fourth --- i.e.\ the tree recovers a
humidity/leaf-wetness-driven decision rule from raw sensor aggregates alone, without
ever observing the oracle's internal wetness-hour count. Under $2\times$ injected
sensor noise, accuracy degrades to 75.0\% and recovers to 85.0\% under 10\% feature
dropout, indicating graceful rather than catastrophic degradation when a sensor is
noisy or intermittently unavailable.

\subsection{Label-Level Fusion}

The two branches are combined at label level: the vision branch contributes a
per-class leaf count vector $\mathbf{c}_t = [c_{t,1},\ldots,c_{t,K}]$ for capture set
$t$, and the sensor branch contributes a discrete risk state $r_t$. Fusion operates
on the count \emph{differential} rather than the absolute counts,

\begin{equation}
\Delta \mathbf{c}_t = \mathbf{c}_t - \mathbf{c}_{t-1},
\label{eq:delta}
\end{equation}

which cancels the constant multiplicity bias introduced by overlapping camera
positions, and produces a fused state

\begin{equation}
s_t = \Phi\!\left(\Delta \mathbf{c}_t,\; r_t\right),
\label{eq:fusion}
\end{equation}

where $\Phi$ is an interpretable rule set: a rising differential in a disease class
concurrent with an elevated environmental risk state escalates the fused output,
whereas a rising count under benign conditions is reported as an observation without
escalation. Fig.~\ref{fig:fusion} traces a representative chain.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig5_fusion.pdf}
\caption{Label-level fusion on a constructed worked example. (a) Real per-class leaf
counts from the trained Case~2~+~Replay classifier on held-out tomato images,
sampled to realize a rising \emph{Early\_blight} trajectory. (b) The concurrent real
sensor trace from the turmeric-parameterized risk model (Section~\ref{sec:fusion}).
(c) The resulting fused decision. The vision (tomato) and risk (turmeric) branches
are paired for illustration only, and the day-by-day image sequencing is
constructed rather than chronologically photographed, since no multi-day tomato
image series exists; both are disclosed here for the same reason the simulated
sensor traces are disclosed in Section~\ref{sec:setup}.}
\label{fig:fusion}
\end{figure}

Two properties of this design matter for a continually learning system. First,
because fusion consumes discrete labels rather than embeddings, the vision branch may
be retrained and swapped monthly without any change to the fusion layer --- an
advantage that feature-level fusion does not offer, since a retrained backbone
invalidates a jointly learned embedding. Second, a failed sensor degrades $r_t$ to an
unknown state, in which case the system reports the vision result alone rather than
producing a corrupted joint prediction.

\begin{table}[!t]
\caption{Label-Level Fusion Versus Single-Modality Decisions}
\label{tab:fusion}
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Decision source} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{False alarms} \\
\midrule
Vision only (counts)      & 60.42\,\% & 29.6 & 14.6\,\% \\
Sensor only (risk tree)   & 35.42\,\% & 20.6 & 12.5\,\% \\
\textbf{Label-level fusion} & \textbf{87.50\,\%} & \textbf{71.8} & \textbf{12.5\,\%} \\
\bottomrule
\multicolumn{4}{@{}l}{\footnotesize 48 constructed episodes, 8 scenario categories, 6 replicates each.}
\end{tabular}
\end{table}

Table~\ref{tab:fusion} evaluates 48 constructed episodes spanning eight scenario
categories --- confirmed progression, benign non-infectious count rises, an
early-warning case (favourable environment, no visible symptoms yet), a recovering
case, a sensor-dropout case, and a deliberately adversarial case where elevated
humidity coincides with, but is not caused by, active infection. Each source is
scored against an independently reasoned ground-truth cause rather than against
$\Phi$'s own output, so the three sources can disagree. Vision-only over-alerts on
benign count rises and completely misses the early-warning case (0\% accuracy on
that category); sensor-only cannot distinguish confirmed disease from merely
favourable conditions and produces no answer when the sensor fails. Fusion resolves
every category except the adversarial one, where it necessarily inherits
sensor-only's mistake because $\Phi$ has no signal that distinguishes a coincidental
humidity reading from a causally elevated one --- an acknowledged limitation of a
two-input discrete rule rather than a hidden failure. When the sensor is
unavailable, fusion's output exactly matches the vision-only decision on every such
episode, confirming graceful degradation rather than a corrupted joint state. Using
the rising-count trajectory as a worked example, fusion and sensor-only would flag
\emph{Watch} a full day earlier than vision-only's differential-based rule would
first cross its alerting threshold.

%% ====================================================================
\section{Experimental Setup}
\label{sec:setup}
%% ====================================================================

\subsection{Datasets and Split Strategy}

Continual learning methods are benchmarked on the tomato subset of PlantVillage
\cite{ref1}, using the segmented leaf images produced by the preceding pipeline stage
so that classifier inputs match deployment conditions. Tomato is selected for the
method study because it provides the class volume and public reproducibility that a
25-run controlled comparison requires; the resulting recipe is then transferred to
the turmeric target crop \cite{ref27} for deployment validation
(Section~\ref{subsec:turmeric}). Five classes (T1--T5) form the base pool, two
further classes (T6--T7) are introduced in CatB, and the 26 non-tomato classes are
used exclusively for Case 3 backbone pretraining. Table~\ref{tab:data} summarizes the
composition.

\begin{table}[!t]
\caption{Dataset Composition and Role}
\label{tab:data}
\centering
\small
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Class} & \textbf{ID} & \textbf{Type} & \textbf{Images} & \textbf{Test} \\
\midrule
Bacterial spot        & T1 & Bacterial & 2{,}127 & 319 \\
Early blight          & T2 & Fungal    & 1{,}000 & 150 \\
Late blight           & T3 & Oomycete  & 1{,}909 & 286 \\
Leaf mold             & T4 & Fungal    & 952     & 143 \\
Healthy               & T5 & ---       & 1{,}591 & 239 \\
Septoria leaf spot    & T6 & Fungal    & 1{,}771 & 265 \\
Spider mites          & T7 & Mite      & 1{,}676 & 252 \\
26 non-tomato classes & --- & Various  & $\sim$31{,}862 & 20\% \\
\bottomrule
\multicolumn{5}{@{}l}{\footnotesize T6--T7 introduced in CatB only; non-tomato classes used for Case 3 pretraining only.}
\end{tabular}
\end{table}

The split design follows three principles. The test set (15\% per class) is
\emph{permanently} held out and evaluated at every checkpoint, so forgetting
measurements are directly comparable across cycles and cases --- this is the same
fixed set that the on-device deployment gate uses. The validation set (10\%) is
reused across all stages including CL cycles, where its role shifts to real-time
forgetting detection: it contains only old-class data, so a drop in validation
accuracy signals forgetting immediately. Stratification is performed by leaf
identifier, preventing images of the same physical leaf from appearing in multiple
splits, which would otherwise inflate results through leakage. The remaining 75\% of
each class is divided 50/25/25 into initial training, Cycle~1 stream and Cycle~2
stream.

\subsection{Evaluation Metrics}

For CatA we report per-cycle top-1 accuracy and macro F1 on the fixed test set, and
training wall-clock time. For CatB we report two dimensions separately:
\emph{new-class accuracy} on T6--T7, and \emph{old-class retention}, the accuracy on
T1--T5 after expansion. Reporting a single averaged figure for CatB would be
misleading, since methods differ precisely in how they trade these quantities.
Deployment feasibility is captured by peak VRAM, process RAM and serialized model
size.

\subsection{Implementation}

All experiments use MobileNetV3-Small at $224\times224\times3$ input, obtained by
resizing the short side to 256 followed by cropping, with the layer-wise schedule of
Table~\ref{tab:lr}. Early stopping is applied with a minimum of five epochs before
the criterion becomes active, preventing premature termination on noisy validation
fluctuations for the small classes T2 and T4.

\textbf{Hardware and software.} All benchmark training reported in
Section~\ref{sec:results} was performed on a single NVIDIA GeForce RTX~4050 Laptop
GPU (6\,GB VRAM, driver 595.79) under Python~3.10 with PyTorch~2.5.1,
torchvision~0.20.1 and CUDA~12.1. Training is carried out in full precision;
automatic mixed precision is not used, so the FP16 comparison of
Section~\ref{subsec:ondevice} concerns only the exported deployment artifact and not
the training pipeline. The deployment measurements of
Section~\ref{subsec:ondevice} were taken on this same GPU and \emph{not} on the
Jetson Orin Nano target, which was unavailable; see the on-device measurement
disclosure below.

\textbf{Optimization.} Every stage uses Adam with weight decay $10^{-4}$ and a
class-weighted cross-entropy loss, with learning rates scheduled by reduction on
plateau of validation macro-F1 (factor 0.5, floor $10^{-6}$); per-group learning
rates are listed in full in Table~\ref{tab:lr}. The batch size is 32 throughout,
except the Case~3A PlantVillage pre-training stage which uses 96, and replay cycles
compose each batch from 16 buffered and 16 stream samples so that the old-to-new
ratio is held at 1:1. Training from scratch (Case~1) and PlantVillage pre-training
(Case~3A) run to at most 50 epochs with early-stopping patience 10; the two-phase
fine-tuning runs to at most 20 then 30 epochs with patience 6 for Case~2, and 15
then 10 epochs with patience 5 for Case~3B. Each continual learning cycle runs to at
most 12 epochs with patience 5.

\textbf{Random seeds.} The continual learning benchmark of
Section~\ref{sec:results} is executed at a \emph{single} random seed (42) for every
case, method and cycle. The differences between methods in
Tables~\ref{tab:cata}--\ref{tab:resources} therefore carry no estimate of
run-to-run variance and should be read as one realization rather than an expected
value; the consistency of the method ranking across three initialization cases and
two category settings is the only stability evidence available. The turmeric
validation of Section~\ref{subsec:turmeric} is the exception, being repeated over
three seeds with mean and standard deviation reported. Repeating the full benchmark
over multiple seeds is the single most valuable extension of this evaluation and is
left to future work.

\textbf{Environmental data disclosure.} The sensor traces used to train and
evaluate the environmental risk model of Section~\ref{sec:fusion} are
\emph{simulated} --- diurnal and recurring seasonal cycles with autocorrelated
drift, datasheet-scaled sensor noise, and dropout, calibrated to plausible
greenhouse ranges. No unit is currently installed, so no in-situ readings exist.
Training labels are distilled from a validated, published infection model rather
than self-defined thresholds (Section~\ref{sec:fusion}), which avoids circularity,
but the absolute risk-model accuracies in Table~\ref{tab:dt} should be read as a
methodology validation, to be revisited against in-situ sensor logs once a turmeric
greenhouse unit is deployed.

\textbf{Fusion evaluation disclosure.} Table~\ref{tab:fusion} is likewise evaluated
on \emph{constructed} episodes rather than field-labelled deployment windows: the
per-image classifier predictions are real (the trained Case~2~+~Replay checkpoint
run on held-out tomato test images), and the risk states are real E1 test-period
outputs, but the day-by-day sequencing that assembles them into episodes, and the
ground-truth outcome assigned to each scenario category, are both constructed. This
should be read as a validation that the fusion rule $\Phi$ behaves as designed given
correctly labelled inputs, not as a field measurement of fused-decision accuracy,
pending real deployment episodes with agronomist-confirmed outcomes.

\textbf{On-device measurement disclosure.} The retraining-cycle, conversion-gate
and classification-latency rows of Table~\ref{tab:ondevice} were measured on a
desktop-class laptop GPU, \emph{not} on the Jetson Orin Nano deployment target,
which was unavailable. The embedded target differs in power envelope, memory
architecture and thermal behaviour, so those absolute figures do not transfer; what
they establish is that the retrain\,$\rightarrow$\,export\,$\rightarrow$\,compile\,$\rightarrow$\,gate
pipeline runs end to end and that the conversion gate is measurable as specified.
For the same reason the compiled artifact is an ONNX Runtime graph rather than the
TensorRT engine the deployment design specifies, the latter being unavailable in
the measurement environment. Detection and segmentation latencies are reported from
prior work \cite{ref21} rather than re-measured here. A full on-device
characterization --- including engine build time, thermal throttling and power
mode --- remains outstanding.

%% ====================================================================
\section{Results and Discussion}
\label{sec:results}
%% ====================================================================

\subsection{Category A --- Same-Class Continual Learning}

Table~\ref{tab:cata} reports per-cycle accuracy and macro F1 for all case--method
combinations, with naive fine-tuning included as a reference. Fig.~\ref{fig:cata}
visualizes the comparison.

\begin{table*}[!t]
\caption{Category A Results --- Same-Class Continual Learning}
\label{tab:cata}
\centering
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Case / Method} & \textbf{C1 Acc} & \textbf{C1 F1} & \textbf{C2 Acc} & \textbf{C2 F1} & \textbf{Avg Acc} & \textbf{Time} \\
\midrule
Case 1 --- EWC       & 97.89\% & 0.9762 & 97.45\% & 0.9713 & 97.67\% & 337\,s \\
Case 1 --- Replay    & 97.45\% & 0.9703 & 97.19\% & 0.9666 & 97.32\% & 482\,s \\
Case 1 --- Isolation & 96.31\% & 0.9563 & 96.57\% & 0.9596 & 96.44\% & 232\,s \\
Case 1 --- Naive$^{*}$ & 97.63\% & 0.9732 & 97.54\% & 0.9713 & 97.58\% & 322\,s \\
\midrule
Case 2 --- EWC       & 97.98\% & 0.9779 & 98.24\% & 0.9807 & \textbf{98.21\%} & \textbf{171\,s} \\
Case 2 --- Replay $\star$ & 98.07\% & 0.9781 & 98.42\% & 0.9827 & \textbf{98.25\%} & 620\,s \\
Case 2 --- Isolation & 97.19\% & 0.9686 & 97.27\% & 0.9697 & 97.23\% & --- \\
Case 2 --- Naive$^{*}$ & 97.63\% & 0.9733 & 99.21\% & 0.9908 & 98.42\% & 322\,s \\
\midrule
Case 3 --- EWC       & 97.63\% & 0.9736 & 97.45\% & 0.9720 & 97.54\% & 339\,s \\
Case 3 --- Replay    & 97.80\% & 0.9763 & 98.50\% & 0.9835 & 98.15\% & 482\,s \\
Case 3 --- Isolation & 96.48\% & 0.9615 & 96.66\% & 0.9633 & 96.57\% & 232\,s \\
Case 3 --- Naive$^{*}$ & 96.48\% & 0.9624 & 98.24\% & 0.9798 & 97.36\% & --- \\
\bottomrule
\multicolumn{7}{@{}l}{\footnotesize $\star$ Best configuration. $^{*}$ Naive requires full data history; reference only, not a valid CL method.}
\end{tabular}
\end{table*}

%% Fig 6 --- FINALISED: figures/fig6_cata.pdf  (source: figures/src/make_data_figs.py)
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig6_cata.pdf}
\caption{Category A average accuracy across initialization cases and continual
learning methods.}
\label{fig:cata}
\end{figure}

Experience replay over an ImageNet-initialized backbone attains the highest average
accuracy at 98.25\%, but the margin over EWC in the same case (98.21\%) is 0.04
percentage points --- within run-to-run noise --- while EWC trains in 171\,s against
620\,s, a 3.6$\times$ difference. For a device performing monthly retraining under a
power and thermal budget, this is a decisive practical distinction: EWC obtains
equivalent accuracy for a fraction of the energy, because it requires only a forward
pass over the buffer to estimate the Fisher matrix and thereafter costs approximately
the same as ordinary fine-tuning.

The three methods exhibit distinct behavioural profiles. EWC is stability-dominant,
holding accuracy nearly constant across cycles. Replay is plasticity-dominant,
achieving the highest Cycle~2 accuracy in every case (Case 2: 98.42\%, Case 3:
98.50\%) because mixing old exemplars into every batch delivers direct gradient
correction at the classifier head and prevents output-neuron drift; Case 3 replay
notably \emph{improves} by 0.70 points from Cycle~1 to Cycle~2, indicating that the
combined old and new signal is synergistic rather than merely protective. Isolation
is consistency-dominant, varying least across cycles (Case 2: 97.19\% then 97.27\%)
since old weights are mathematically untouched and only the adapter learns.

Isolation's roughly one-point deficit relative to replay is a capacity constraint
rather than an implementation defect. The bottleneck-24 adapter adds approximately
18{,}000 parameters, 1.2\% of the backbone, deliberately minimal to bound model
growth; the adapter is the sole plastic element, so its representational budget caps
achievable accuracy. In exchange it delivers the lowest peak VRAM of any method
(122.63\,MB) and a zero-forgetting guarantee, which makes it a rational choice for
severely memory-constrained devices operating in same-class streams --- and, as
Section~\ref{subsec:isolation} shows, only there.

Notably, naive fine-tuning forgets only marginally in CatA (97.36--98.42\%). This is
not evidence that protection is unnecessary: because the same five classes recur in
every stream, the model continuously observes all old classes, which constitutes
implicit replay. The distinction becomes stark in CatB, where old-class data is
absent from the incoming stream.

\subsection{Regularization Strength}
\label{subsec:lambda}

Table~\ref{tab:lambda} presents the $\lambda$ grid search on Case 2, and
Fig.~\ref{fig:lambda} plots the sensitivity.

\begin{table}[!t]
\caption{EWC Lambda Grid Search (Case 2, Category A)}
\label{tab:lambda}
\centering
\small
\begin{tabular}{@{}lcccl@{}}
\toprule
$\boldsymbol{\lambda}$ & \textbf{C1 Acc} & \textbf{C2 Acc} & \textbf{C1 Time} & \textbf{Assessment} \\
\midrule
1{,}000  & 97.71\% & 98.15\% & 87.9\,s & Under-regularized \\
3{,}000  & 97.80\% & \textbf{98.24\%} $\star$ & 88.0\,s & Optimal for Cycle 2 \\
5{,}000  & 97.63\% & 98.24\% & 85.5\,s & Marginal over-regularization \\
8{,}000  & 97.89\% & 98.07\% & 85.9\,s & Slight plasticity loss \\
10{,}000 & \textbf{97.98\%} $\star$ & 98.24\% & 85.2\,s & Optimal for Cycle 1 \\
15{,}000 & 97.89\% & 98.15\% & 84.7\,s & Over-regularized \\
\bottomrule
\end{tabular}
\end{table}

%% Fig 7 --- FINALISED: figures/fig7_lambda.pdf  (source: figures/src/make_data_figs.py)
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig7_lambda.pdf}
\caption{EWC accuracy versus regularization strength. The optimum shifts downward
between cycles as the model converges.}
\label{fig:lambda}
\end{figure}

The optimum is cycle-dependent: $\lambda=10{,}000$ maximizes Cycle~1 accuracy, but
$\lambda=3{,}000$ is preferable in Cycle~2. The interpretation is that Cycle~1 must
protect a freshly transferred representation against the first substantial
distribution shift, whereas by Cycle~2 the model has already adapted through a
complete cycle and further constraint suppresses plasticity without adding retention.
This motivates the per-cycle schedule adopted throughout. Training time is constant
across $\lambda$ (84.7--88.0\,s), confirming that $\lambda$ affects regularization
quality alone and not computational cost --- so per-cycle tuning is free at
deployment time.

Table~\ref{tab:lambdacase} shows that the benefit of tuning depends strongly on
initialization.

\begin{table}[!t]
\caption{Impact of Lambda Tuning Across Initialization Cases}
\label{tab:lambdacase}
\centering
\small
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Case} & \textbf{Fixed} $\lambda=1{,}000$ & \textbf{Tuned} $\lambda$ & \textbf{Gain} \\
\midrule
Case 1 --- Scratch        & 97.58\% & 97.67\% & $+0.09$ \\
Case 2 --- ImageNet       & 97.82\% & 98.21\% & $+0.39$ \\
Case 3 --- PlantVillage $\star$ & 96.40\% & 97.54\% & $\mathbf{+1.14}$ \\
\bottomrule
\end{tabular}
\end{table}

The in-domain backbone benefits most ($+1.14$ points). We attribute this to the
structure of its Fisher landscape: pretraining on 26 plant classes concentrates
importance on a dense cluster of genuinely disease-discriminative weights, making the
penalty gradient in \eqref{eq:ewc} both steeper and more meaningful. The scratch model
gains least ($+0.09$) because random initialization yields a diffuse Fisher matrix
with no coherent prior structure worth protecting --- the penalty has little of value
to preserve. The practical implication is direct: \emph{$\lambda$ tuning is not
optional for domain-specific pretrained backbones}, which is exactly the
configuration a deployed agricultural product uses.

\subsection{Category B --- New-Class Continual Learning}

Table~\ref{tab:catb} reports both evaluation dimensions, and Fig.~\ref{fig:catb}
plots them against one another to expose the stability--plasticity structure.

\begin{table*}[!t]
\caption{Category B Results --- New-Class Expansion (5$\rightarrow$7 Classes)}
\label{tab:catb}
\centering
\small
\begin{tabular}{@{}lccccl@{}}
\toprule
\textbf{Case / Method} & \textbf{C1 New} & \textbf{C2 New} & \textbf{Avg New} & \textbf{Old Ret.} & \textbf{Time} \\
\midrule
Case 1 --- EWC       & 78.92\% & 86.65\% & 82.79\% & 87.34\% & 1{,}048\,s \\
Case 1 --- Replay    & 93.23\% & 92.46\% & 92.85\% & 84.78\% & 1{,}268\,s \\
Case 1 --- Naive$^{*}$ & 94.39\% & 93.42\% & 93.91\% & 87.07\% & 743\,s \\
Case 1 --- Hybrid    & 85.30\% & 83.56\% & 84.43\% & 92.44\% & 1{,}741\,s \\
\midrule
Case 2 --- EWC       & 93.62\% & 94.78\% & 94.20\% & 95.95\% & 797\,s \\
Case 2 --- Replay $\star$ & 98.07\% & 98.07\% & \textbf{98.07\%} & \textbf{97.19\%} & 1{,}181\,s \\
Case 2 --- Isolation & 45.45\% & 56.29\% & 50.87\% & $\sim$100\% & --- \\
Case 2 --- Naive$^{*}$ & 94.58\% & 96.91\% & 95.74\% & 87.51\% & 351\,s \\
Case 2 --- Hybrid    & 93.81\% & 96.71\% & 95.26\% & 98.24\% & 1{,}659\,s \\
\midrule
Case 3 --- EWC       & 90.72\% & 90.52\% & 90.62\% & 95.78\% & 1{,}128\,s \\
Case 3 --- Replay    & 96.32\% & 96.52\% & 96.42\% & 96.92\% & 1{,}220\,s \\
Case 3 --- Isolation & 6.38\%  & 24.37\% & 15.38\% & $\sim$100\% & --- \\
Case 3 --- Naive$^{*}$ & 97.49\% & 96.32\% & 96.91\% & 94.72\% & 668\,s \\
Case 3 --- Hybrid    & 92.65\% & 93.04\% & 92.84\% & 97.54\% & 1{,}691\,s \\
\bottomrule
\multicolumn{6}{@{}l}{\footnotesize Old retention measured at Cycle 2 on T1--T5. $^{*}$Reference only.}
\end{tabular}
\end{table*}

%% Fig 8 --- FINALISED: figures/fig8_catb_scatter.pdf  (source: figures/src/make_data_figs.py)
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig8_catb_scatter.pdf}
\caption{Stability--plasticity trade-off under new-class expansion. Only
ImageNet-initialized experience replay occupies the desirable upper-right region.}
\label{fig:catb}
\end{figure}

Case 2 with experience replay is the unambiguous winner on both dimensions at once:
98.07\% new-class accuracy with 97.19\% old-class retention. No other configuration
exceeds 96\% on both simultaneously. The mechanism is that data-level correction acts
directly at the output layer --- replayed old samples and new stream samples appear in
the same batch, so the split-LR head receives gradient signal for all seven classes
concurrently and the two objectives do not conflict at the decision boundary.

Forgetting under naive fine-tuning is now unmistakable: old-class retention falls to
87.07--87.51\% in Cases 1 and 2, against 97.19\% for replay. The contrast with CatA,
where naive lost almost nothing, confirms that the implicit replay afforded by
recurring classes --- not any inherent robustness --- was responsible for the earlier
result.

\subsection{Failure Analysis: Parameter Isolation}
\label{subsec:isolation}

Parameter isolation, which performs within roughly one point of the best method in
CatA, fails catastrophically when new classes are introduced: 50.87\% average
new-class accuracy in Case 2 and 15.38\% in Case 3, the latter near chance for a
two-class discrimination.

The mechanism is architectural. The CatB isolation implementation trains a separate
branch of three inverted-residual blocks from scratch for T6 and T7 while the
backbone remains frozen. That branch receives no feature information from the
pretrained backbone's representation --- it must learn disease-discriminative features
from approximately 740 images per class with no shared extraction --- which is
insufficient to become competitive with a fully pretrained pathway.

The case ordering is diagnostic and initially counter-intuitive: Case 3, with the
\emph{stronger} in-domain backbone, performs \emph{worse} (6.38\% at Cycle~1) than
Case 2 (45.45\%). The better the frozen backbone, the larger the representational gap
between it and a randomly initialized branch, and the more starkly the branch's
poverty is exposed at the shared decision layer. Isolation's zero-forgetting guarantee
is therefore not merely offset but inverted in value: it purchases perfect stability
by structurally forbidding the new pathway from using the very knowledge that makes
the system accurate. We conclude that \textbf{parameter isolation must not be used in
any deployment scenario admitting new classes}; its use is confined to same-class
streaming under strict memory constraints.

\subsection{Does Combining Methods Help?}

The hybrid EWC$+$replay scheme was motivated by the hypothesis that weight-level and
data-level protection would compose. The results in Table~\ref{tab:hybrid} do not
support it.

\begin{table}[!tbp]
\caption{Hybrid (EWC$+$Replay) Versus Replay Alone, Category B}
\label{tab:hybrid}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{New-class}} & \textbf{Hybrid} & \textbf{Time} \\
\cmidrule(lr){2-3}
\textbf{Case} & \textbf{Hybrid} & \textbf{Replay} & \textbf{old ret.} & \textbf{overhead} \\
\midrule
Case 1 & 84.43 & 92.85 ($+8.4$) & 92.44 & $+37\%$ \\
Case 2 & 95.26 & 98.07 ($+2.8$) & 98.24 & $+40\%$ \\
Case 3 & 92.84 & 96.42 ($+3.6$) & 97.54 & $+39\%$ \\
\bottomrule
\multicolumn{5}{@{}l}{\footnotesize Accuracies in \%; parentheses give replay's margin.}
\end{tabular}
\end{table}

Hybrid underperforms replay on new-class accuracy by 2.8--8.4 points while costing
37--40\% more training time. Its old-class retention advantage (1--4 points) does not
justify this, since replay already achieves 96.9--97.2\% retention. The explanation is
that the two mechanisms \emph{compete} rather than cooperate in the class-incremental
setting: the EWC penalty constrains exactly the G3 and G4 parameters that must move to
acquire T6 and T7 features, partially cancelling the replay buffer's gradient
contribution. Weight-level and data-level protection are not complementary here
because they act on the same parameters with opposing intent.

\subsection{Resource Utilization}

Table~\ref{tab:resources} reports deployment-relevant costs.

\begin{table}[!tbp]
\caption{Resource Utilization by Method and Scenario}
\label{tab:resources}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccl@{}}
\toprule
\textbf{Method /} & \textbf{Peak VRAM} & \textbf{Model} & \\
\textbf{scenario} & \textbf{(MB)} & \textbf{(MB)} & \textbf{Note} \\
\midrule
Isolation --- CatA & 122.63   & 5.81 & Lowest VRAM \\
Replay --- CatA    & 155.58   & 5.81 & Best accuracy \\
EWC --- CatA       & 169.51   & 5.81 & Fastest (171\,s) \\
Naive --- CatA/B   & 161--162 & 5.81 & Reference \\
Replay --- CatB    & 325.39   & 5.82 & Head expansion \\
EWC --- CatB       & 383--601 & 5.82 & Fisher at Cycle 2 \\
Hybrid --- CatB    & 384--601 & 5.82 & Highest; no gain \\
\bottomrule
\end{tabular}
\end{table}

Every trained model remains within 5.82\,MB, comfortably inside the 6\,MB edge
budget, and class expansion adds only 0.01\,MB. Peak memory during \emph{retraining},
not model size, is the binding constraint on device: CatA methods fit within 170\,MB
while CatB replay requires 325\,MB and CatB EWC peaks at 601\,MB when the Fisher
matrix is held alongside activations. This ordering is itself a deployment finding ---
under new-class expansion, replay is simultaneously the most accurate and roughly
half the memory cost of EWC, reversing their CatA relationship.

\subsection{On-Device Cycle Cost and Latency}
\label{subsec:ondevice}

\begin{table}[!tbp]
\caption{Retraining Cycle and Inference Cost}
\label{tab:ondevice}
\centering
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Operation} & \textbf{Time} & \textbf{Memory} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Monthly retraining cycle}} \\
Replay retrain cycle                 & 373.2\,s & 155.6\,MB \\
ONNX export                          & 1.5\,s   & 5.8\,MB \\
TensorRT FP16 build                  & \multicolumn{2}{c}{n/a$^{\ddagger}$} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Conversion gate (fixed test set, 1{,}137 images)}} \\
PyTorch FP32 candidate               & \multicolumn{2}{c}{98.51\,\%} \\
PyTorch FP16                         & \multicolumn{2}{c}{98.59\,\% ($\Delta$ $+0.09$)} \\
Compiled graph$^{\ddagger}$          & \multicolumn{2}{c}{98.51\,\% ($\Delta$ $0.00$)} \\
\midrule
\multicolumn{3}{@{}l}{\emph{Per-stage inference latency}} \\
Detection, per image$^{\dagger}$     & 12.2\,ms & 6.15\,MB \\
Segmentation, per leaf$^{\dagger}$   & 5.0\,ms  & 6.15\,MB \\
Classification, eager                & 10.2\,ms & 5.8\,MB \\
Classification, compiled             & 3.8\,ms  & 5.8\,MB \\
\bottomrule
\end{tabular}

\vspace{2pt}
\begin{minipage}{\columnwidth}
\footnotesize $^{\dagger}$Reported in prior work \cite{ref21}. All other rows measured
here on a laptop GPU (RTX 4050, CUDA 12.1) as a proxy, \emph{not} on the Jetson Orin
Nano target. $^{\ddagger}$TensorRT was unavailable in the measurement environment;
the compiled graph is ONNX Runtime GPU.
\end{minipage}
\end{table}

Three observations follow. First, a full replay retraining cycle completes in
roughly six minutes of wall-clock time on the proxy hardware with a peak of
155.6\,MB of accelerator memory --- a footprint small enough that the corresponding
cycle on the deployment target is plausibly an overnight background task rather
than a workload competing with daytime capture, although confirming this requires
the on-device measurement itself. Second, and more consequentially for the design,
the conversion gate finds an accuracy delta of essentially zero: the compiled graph
reproduces the FP32 candidate exactly, and half precision differs by a single image
out of 1{,}137. We report this rather than omit it. The justification for the gate
is therefore not that conversion is empirically lossy, but that a system promoting
a converted artifact into production cannot know that without measuring the
artifact it actually deploys; here the measurement confirms equivalence, and a gate
that assumed equivalence would have been correct by luck rather than by evidence.
Third, compiling the classifier yields a $2.7\times$ latency reduction at identical
accuracy, whereas eager half-precision execution is \emph{slower} than FP32 for a
network this small, the per-operation conversion overhead exceeding any arithmetic
saving --- a further illustration that conversion effects must be measured per
model rather than assumed from precision alone.

Summing the cited detection and segmentation stages with the measured compiled
classification stage places one eight-image capture set on the order of a few
hundred milliseconds. With two capture sets per day, the device is therefore idle
for the overwhelming majority of its duty cycle, leaving ample headroom for a
retraining pass to run without contending with inference. This arithmetic combines
figures obtained on different hardware and should be read as indicative of the
duty-cycle argument rather than as a measured end-to-end deployment latency.

\subsection{Turmeric Deployment Validation}
\label{subsec:turmeric}

The recipe selected on the tomato benchmark --- Case 2 initialization with
experience replay --- is applied unchanged to the five-class turmeric dataset under
the CatA protocol, two cycles, using the same layer-group schedule. Results are
reported in Table~\ref{tab:turmeric} as mean $\pm$ standard deviation over three
seeds.

\begin{table}[!tbp]
\caption{Turmeric Deployment Validation --- Transfer of the Selected Recipe}
\label{tab:turmeric}
\centering
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Stage} & \textbf{Accuracy} & \textbf{Macro F1} \\
\midrule
Base (Case 2, ImageNet)      & 89.10 $\pm$ 2.35 & 89.18 $\pm$ 2.16 \\
Case 2 --- Replay, cycle 1   & 92.52 $\pm$ 3.37 & 92.29 $\pm$ 3.76 \\
\textbf{Case 2 --- Replay, cycle 2} & \textbf{90.03 $\pm$ 3.54} & \textbf{89.93 $\pm$ 3.73} \\
\midrule
Naive fine-tuning, cycle 2$^{\dagger}$ & 87.85 $\pm$ 2.47 & 87.72 $\pm$ 2.68 \\
\bottomrule
\end{tabular}

\vspace{2pt}
\begin{minipage}{\columnwidth}
\footnotesize Five turmeric classes: healthy, leaf blotch, leaf spot, dry, aphid
disease. Mean $\pm$ std over three seeds; test set $n=107$.
$^{\dagger}$Reference condition only --- not a valid continual-learning method.
\end{minipage}
\end{table}

Two observations follow, and a limitation that must be stated with them. First, the
recipe transfers: replay trains successfully on the target crop and is the only
configuration that ends above its own starting point ($+0.93$ percentage points over
the base model), whereas naive fine-tuning ends $1.25$ points below it. The
qualitative ordering observed on tomato is preserved. Second, the much smaller
per-class volume forces one documented change to the protocol. The tomato replay
buffer rule --- 15\% per class with a floor of 100 images --- is unsatisfiable when
initial training holds only 76--92 images per class, so the floor is redefined for
turmeric as $\min(40,\,50\%$ of initial training$)$, yielding 38--40 images per
class. The $\lambda$ schedule is carried over unchanged.

The limitation is the test set. With $n=107$, a single image is worth $0.93$
percentage points and one run carries a 95\% confidence interval of roughly $\pm5.7$
points. Applying McNemar's paired test to the per-image predictions --- the
appropriate test when two models are scored on identical data --- no pairwise
difference between replay, EWC and naive reaches significance ($p \geq 0.109$ for
every comparison across all three seeds), and per-seed accuracies vary by up to
$6.5$ points within a single method. Table~\ref{tab:turmeric} should therefore be
read as evidence that the recipe \emph{transfers and remains stable} on the target
crop, and explicitly not as a ranking of continual-learning methods on turmeric;
establishing such a ranking would require substantially more data than the
approximately 200 images per class currently available.

%% Figure 9 (deployment photograph + dashboard screenshot) was REMOVED.
%% Reason: no physical unit is installed, so the photograph cannot be produced, and
%% a dashboard screenshot alone does not carry the deployment claim on its own. The
%% figure was never referenced from the body text, so nothing else depends on it.
%% Deployment-conditional wording elsewhere was softened at the same time -- if a
%% unit is later installed, restore this figure AND revisit those sentences.

\subsection{Cross-Cutting Discussion}

Three findings generalize beyond the specific numbers.

\textbf{Transfer quality sets the ceiling on continual learning.} Across every method,
Cases 2 and 3 outperform Case 1, and the best CatA and CatB results both arise from
Case 2. A scratch-initialized backbone lacks a mature representation for CL to
protect: EWC's Fisher matrix has no coherent structure to identify, and replay's
exemplars correct an output layer that sits on unstable features. Case 3's in-domain
backbone yields the best \emph{old-class retention} under expansion (96.92\% with
replay, 97.54\% with hybrid), because plant-disease-discriminative features are
already encoded in G3 before CL begins and survive the same LR schedule better than
generic features. The practical reading is that pretraining and CL method selection
are not independent decisions.

\textbf{Where forgetting happens determines what protects against it.} Confining
plasticity to G3 and G4 concentrated both the EWC penalty and the replay gradient on
the parameters that actually drift. Had G1 been left trainable, the Fisher
normalization in \eqref{eq:fisher} would have been spread across parameters that carry
no task-specific information, weakening the effective penalty on those that do. The
split-LR head addresses the complementary failure at the output layer identified in
\cite{ref18}. This is why the layer-wise strategy is reported as a contribution rather
than as an implementation detail: it is a precondition for all three method families
performing as measured.

\textbf{Method choice is scenario-dependent, not universal.} A single ranking across
both scenarios does not exist. Isolation is competitive in CatA and catastrophic in
CatB; EWC is the efficiency winner in CatA but doubles memory in CatB; replay is
strongest overall in CatB yet costs 3.6$\times$ EWC's training time in CatA. A
deployment that reports one leaderboard therefore misleads. Section~\ref{sec:reco}
translates this into explicit selection rules.

\subsection{Limitations}

Four limitations bound these claims. The CL benchmark uses PlantVillage tomato
imagery, which is less visually heterogeneous than field-captured greenhouse data, so
absolute accuracies should be read as an upper envelope; the turmeric validation of
Section~\ref{subsec:turmeric} partially addresses this. Each configuration was
evaluated over two cycles, sufficient to expose forgetting dynamics but not to
characterize very long deployment horizons where small per-cycle drifts may
accumulate. The confidence-routing thresholds were set empirically rather than
calibrated, and a mis-set $\tau_{\text{high}}$ would admit label noise into the
retrain buffer --- calibrated uncertainty estimation is the natural remedy. Finally,
the GAN occlusion-recovery stage remains disabled pending validation on real farm
imagery, so reported pipeline behaviour reflects the pipeline without it.

%% ====================================================================
\section{Deployment Recommendations}
\label{sec:reco}
%% ====================================================================

The results translate into the following selection rules for practitioners building
continually learning agricultural edge systems.

\begin{enumerate}
\item \textbf{Default to experience replay with an ImageNet-initialized backbone.}
It is the only configuration strong on both dimensions of new-class expansion
(98.07\% new, 97.19\% old) and also leads same-class streaming (98.25\%).
\item \textbf{Prefer EWC when update frequency is high or the power budget is tight.}
It reaches 98.21\% in same-class streaming --- 0.04 points behind replay --- at
3.6$\times$ lower training cost, and stores no images.
\item \textbf{Never use parameter isolation where new classes may appear.} New-class
accuracy collapses to 15--51\%. Restrict it to same-class streams under severe memory
limits, where its 122.63\,MB peak and zero-forgetting guarantee are genuinely useful.
\item \textbf{Do not combine EWC with replay.} The mechanisms compete on the same
parameters; the hybrid costs 37--40\% more time and loses 2.8--8.4 points of
new-class accuracy relative to replay alone.
\item \textbf{Tune $\lambda$ per cycle, and always for in-domain backbones.} The
optimum decreases as the model converges ($10{,}000 \rightarrow 3{,}000$), tuning
costs no additional training time, and it is worth up to $+1.14$ points for a
domain-pretrained model --- the configuration a production system uses.
\item \textbf{Freeze early layers and split the classifier learning rate.}
Concentrating plasticity in late features and protecting old-class output neurons is a
precondition for all three method families to perform as reported.
\item \textbf{Gate deployment on the compiled engine, not the training artifact.}
Quantized inference can shift accuracy; a gate that evaluates the wrong artifact can
admit a model that regresses in production.
\end{enumerate}

%% ====================================================================
\section{Conclusion and Future Work}
\label{sec:conclusion}
%% ====================================================================

This paper presented LEAFSENSE, an offline-first AIoT framework that executes the
complete perception-to-adaptation loop for plant disease monitoring on a single edge
device, and a controlled 25-run continual learning study establishing which
adaptation strategy that loop should use. Experience replay over an
ImageNet-initialized MobileNetV3-Small proved strongest overall, reaching 98.25\%
average accuracy in same-class streaming and simultaneously 98.07\% new-class
accuracy with 97.19\% old-class retention under 5$\rightarrow$7 class expansion,
while EWC delivered statistically equivalent same-class accuracy at 3.6$\times$ lower
training cost. Two negative results carry equal practical weight: parameter isolation
fails categorically when new classes are introduced, because architectural separation
prevents the new pathway from exploiting the pretrained representation; and combining
EWC with replay degrades new-class acquisition while increasing cost, since the two
mechanisms constrain and drive the same parameters in opposition. Underlying all
three method families, the layer-wise training strategy --- permanent early-layer
freezing, decayed late-feature learning rates, and a split learning-rate classifier
head --- proved to be a precondition rather than a refinement. All models remain
within 5.82\,MB and the retraining cycle fits the target device's envelope,
establishing that autonomous, self-improving disease monitoring is achievable without
any dependence on cloud infrastructure.

Several directions follow. The most immediate is replacing the human relabeller with
an \emph{LLM teacher}: low-confidence crops from the relabel queue are labelled in
batch by a vision-capable large language model with structured outputs, closing the
supervision loop without human intervention and making the device autonomous
end-to-end. Beyond this, an \emph{agentic orchestration layer} would allow the system
to reason over its own trend data and knowledge base to schedule interventions rather
than merely reporting risk states. Further work includes maturing the GAN
occlusion-recovery stage on real greenhouse imagery so it can be enabled by default,
extending the evaluation to longer cycle horizons and additional crop variants, and
replacing empirical confidence thresholds with calibrated uncertainty estimates to
tighten the guarantees on auto-labelled data entering the retrain buffer.

\section*{Acknowledgment}
The authors thank the Department of Electrical and Electronic Engineering,
University of Jaffna, for supporting this work.
% >>> Add greenhouse facility / funding body acknowledgements here, or delete
% this section entirely if not applicable.

\begin{thebibliography}{35}

\bibitem{ref1}
D. P. Hughes and M. Salath\'e, ``An open access repository of images on plant health
to enable the development of mobile disease diagnostics,'' \emph{arXiv:1511.08060},
2015.

\bibitem{ref2}
S. P. Mohanty, D. P. Hughes, and M. Salath\'e, ``Using deep learning for image-based
plant disease detection,'' \emph{Front. Plant Sci.}, vol. 7, Sep. 2016, Art. no. 1419,
doi: 10.3389/fpls.2016.01419.

\bibitem{ref3}
K. He, X. Zhang, S. Ren, and J. Sun, ``Deep residual learning for image recognition,''
in \emph{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, Las Vegas, NV, USA,
2016, pp. 770--778, doi: 10.1109/CVPR.2016.90.

\bibitem{ref4}
M. Tan and Q. V. Le, ``EfficientNet: Rethinking model scaling for convolutional neural
networks,'' in \emph{Proc. 36th Int. Conf. Mach. Learn. (ICML)}, Long Beach, CA, USA,
2019, pp. 6105--6114.

%% >>> REPLACED 2026-07-29. The previous entry here --- H. Sharma and M. Patel,
%% "Edge computing challenges for deep learning-based plant disease detection
%% systems," IEEE Internet Things J., vol. 9, no. 12, pp. 9876-9888, 2022,
%% doi: 10.1109/JIOT.2022.3142890 --- DOES NOT EXIST. Its DOI returns HTTP 404,
%% and IEEE IoT J. vol. 9 no. 12 runs ...9858-9871 then 9889-9903, so the claimed
%% page range 9876-9888 falls in a gap occupied by no article. Replaced with a
%% Crossref-verified reference supporting the same claim. (The bad entry was
%% inherited from the prior conference paper, where it appears as [4].) <<<
\bibitem{ref5}
A. T. Khan, S. M. Jensen, A. R. Khan, and S. Li, ``Plant disease detection model
for edge computing devices,'' \emph{Front. Plant Sci.}, vol. 14, Dec. 2023,
Art. no. 1308528, doi: 10.3389/fpls.2023.1308528.

\bibitem{ref6}
A. Howard \emph{et al.}, ``Searching for MobileNetV3,'' in \emph{Proc. IEEE/CVF Int.
Conf. Comput. Vis. (ICCV)}, Seoul, South Korea, 2019, pp. 1314--1324,
doi: 10.1109/ICCV.2019.00140.

\bibitem{ref7}
M. Zhu and S. Gupta, ``To prune, or not to prune: Exploring the efficacy of pruning
for model compression,'' in \emph{Proc. 6th Int. Conf. Learn. Represent. (ICLR)},
Vancouver, BC, Canada, 2018.

\bibitem{ref8}
R. Banner, Y. Nahshan, and D. Soudry, ``Post training 4-bit quantization of
convolutional networks for rapid-deployment,'' in \emph{Proc. 33rd Conf. Neural Inf.
Process. Syst. (NeurIPS)}, Vancouver, BC, Canada, 2019, pp. 7950--7958.

\bibitem{ref9}
G. Jocher, A. Chaurasia, and J. Qiu, ``Ultralytics YOLOv8,'' 2023. [Online].
Available: https://github.com/ultralytics/ultralytics

\bibitem{ref10}
E. A. Aldakheel, M. Zakariah, and A. H. Alabdalall, ``Detection and identification of
plant leaf diseases using YOLOv4,'' \emph{Front. Plant Sci.}, vol. 15, Apr. 2024,
Art. no. 1355941, doi: 10.3389/fpls.2024.1355941.

\bibitem{ref11}
Y. Miao, W. Meng, and X. Zhou, ``SerpensGate-YOLOv8: An enhanced YOLOv8 model for
accurate plant disease detection,'' \emph{Front. Plant Sci.}, vol. 15, Jan. 2025,
Art. no. 1514832, doi: 10.3389/fpls.2024.1514832.

\bibitem{ref12}
H. Guan \emph{et al.}, ``A lightweight model for efficient identification of plant
diseases and pests based on deep learning,'' \emph{Front. Plant Sci.}, vol. 14,
Aug. 2023, Art. no. 1227011, doi: 10.3389/fpls.2023.1227011.

\bibitem{ref13}
R. M. French, ``Catastrophic forgetting in connectionist networks,'' \emph{Trends
Cogn. Sci.}, vol. 3, no. 4, pp. 128--135, Apr. 1999,
doi: 10.1016/S1364-6613(99)01294-2.

\bibitem{ref14}
M. De Lange \emph{et al.}, ``A continual learning survey: Defying forgetting in
classification tasks,'' \emph{IEEE Trans. Pattern Anal. Mach. Intell.}, vol. 44,
no. 7, pp. 3366--3385, Jul. 2022, doi: 10.1109/TPAMI.2021.3057446.

\bibitem{ref15}
J. Kirkpatrick \emph{et al.}, ``Overcoming catastrophic forgetting in neural
networks,'' \emph{Proc. Nat. Acad. Sci. USA}, vol. 114, no. 13, pp. 3521--3526,
Mar. 2017, doi: 10.1073/pnas.1611835114.

\bibitem{ref16}
F. Zenke, B. Poole, and S. Ganguli, ``Continual learning through synaptic
intelligence,'' in \emph{Proc. 34th Int. Conf. Mach. Learn. (ICML)}, Sydney, NSW,
Australia, 2017, pp. 3987--3995.

\bibitem{ref17}
S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, ``iCaRL: Incremental
classifier and representation learning,'' in \emph{Proc. IEEE Conf. Comput. Vis.
Pattern Recognit. (CVPR)}, Honolulu, HI, USA, Jul. 2017, pp. 5533--5542,
doi: 10.1109/CVPR.2017.587.

\bibitem{ref18}
Y. Wu \emph{et al.}, ``Large scale incremental learning,'' in \emph{Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. (CVPR)}, Long Beach, CA, USA, 2019,
pp. 374--382, doi: 10.1109/CVPR.2019.00046.

\bibitem{ref19}
A. A. Rusu \emph{et al.}, ``Progressive neural networks,''
\emph{arXiv:1606.04671}, 2016.

\bibitem{ref20}
A. Mallya and S. Lazebnik, ``PackNet: Adding multiple tasks to a single network by
iterative pruning,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.
(CVPR)}, Salt Lake City, UT, USA, 2018, pp. 7765--7773,
doi: 10.1109/CVPR.2018.00810.

\bibitem{ref21}
%% Crossref-verified: DOI resolves, event metadata confirms Peradeniya, Sri Lanka,
%% 16-17 Jan 2026, pp. 395-400, all three authors present.
R. Roycetheeban, A. M. Mathushanth, and M. Tharmakulasingam, ``A lightweight modular
pipeline for edge-optimized leaf disease detection,'' in \emph{Proc. IEEE 19th Int.
Conf. Ind. Inf. Syst. (ICIIS)}, Peradeniya, Sri Lanka, Jan. 2026, pp. 395--400,
doi: 10.1109/ICIIS69028.2026.11450780.

\bibitem{ref22}
%% No DOI: published as a proceedings volume by the University of Jaffna, not
%% DOI-registered (confirmed absent from Crossref). Page range 7-16 confirmed
%% against the proceedings table of contents.
R. Roycetheeban, A. M. Mathushanth, and M. Tharmakulasingam, ``GAN-based leaf image
inpainting for occlusion recovery,'' in \emph{Proc. 2nd Int. Conf. Digit. Technol.
Sustain. Agricult. (ICDTSA)}, Kilinochchi, Sri Lanka, May 2025, pp. 7--16.

\bibitem{ref23}
G. Zhang, L. Wang, G. Kang, L. Chen, and Y. Wei, ``SLCA: Slow learner with classifier
alignment for continual learning on a pre-trained model,'' in \emph{Proc. IEEE/CVF
Int. Conf. Comput. Vis. (ICCV)}, Paris, France, 2023, pp. 19148--19158.

\bibitem{ref24}
D. Li, Z. Yin, Y. Zhao, J. Li, and H. Zhang, ``Rehearsal-based class-incremental
learning approaches for plant disease classification,'' \emph{Comput. Electron.
Agricult.}, vol. 224, Sep. 2024, Art. no. 109211,
doi: 10.1016/j.compag.2024.109211.

\bibitem{ref25}
G. N. Agrios, \emph{Plant Pathology}, 5th ed. Amsterdam, The Netherlands: Elsevier
Academic Press, 2005.

\bibitem{ref26}
S. Lee and C. M. Yun, ``A deep learning model for predicting risks of crop pests and
diseases from sequential environmental data,'' \emph{Plant Methods}, vol. 19,
Dec. 2023, Art. no. 145, doi: 10.1186/s13007-023-01122-x.

\bibitem{ref27}
%% Author lists retrieved and verified via the DataCite API (Mendeley Data DOIs are
%% registered with DataCite, not Crossref, so they do not resolve on Crossref).
M. R. Hossain \emph{et al.}, ``Image dataset for turmeric plant leaf disease
detection,'' \emph{Mendeley Data}, 2025, doi: 10.17632/jtttfbx342.1; and
A. K. M. F. K. Siam, M. A. S. Nirob, and P. Bishshash, ``Turmeric plant disease
dataset: Advancing AI for agricultural sustainability,'' \emph{Mendeley Data}, 2024,
doi: 10.17632/g46dvrcvwn.1.

\bibitem{ref28}
C. Rudin, ``Stop explaining black box machine learning models for high stakes
decisions and use interpretable models instead,'' \emph{Nature Mach. Intell.},
vol. 1, no. 5, pp. 206--215, May 2019, doi: 10.1038/s42256-019-0048-x.

\bibitem{ref29}
Z. Li and D. Hoiem, ``Learning without forgetting,'' \emph{IEEE Trans. Pattern Anal.
Mach. Intell.}, vol. 40, no. 12, pp. 2935--2947, Dec. 2018,
doi: 10.1109/TPAMI.2017.2773081.

\bibitem{ref30}
J. Lin, L. Zhu, W.-M. Chen, W.-C. Wang, C. Gan, and S. Han, ``On-device training
under 256\,KB memory,'' in \emph{Proc. 36th Conf. Neural Inf. Process. Syst.
(NeurIPS)}, New Orleans, LA, USA, 2022, pp. 22941--22954.

\bibitem{ref31}
%% Page range VERIFIED against the publisher: proceedings.mlr.press/v54/mcmahan17a
%% states "PMLR 54:1273-1282". Not in Crossref (PMLR proceedings are not DOI-registered).
H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
``Communication-efficient learning of deep networks from decentralized data,'' in
\emph{Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS)}, Fort Lauderdale, FL,
USA, 2017, pp. 1273--1282.

\bibitem{ref32}
Q. Xie, M.-T. Luong, E. Hovy, and Q. V. Le, ``Self-training with noisy student
improves ImageNet classification,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR)}, Seattle, WA, USA, 2020, pp. 10684--10695,
doi: 10.1109/CVPR42600.2020.01070.

\bibitem{ref33}
%% Paper and author list VERIFIED present in the NeurIPS 2020 proceedings index
%% (papers.nips.cc). The page range 596-608 is from the Curran Associates printed
%% volume (Adv. NIPS 33) and is the range used throughout the literature; the
%% NeurIPS site itself publishes no pagination, so it could not be confirmed
%% against a primary publisher API. Not DOI-registered.
K. Sohn \emph{et al.}, ``FixMatch: Simplifying semi-supervised learning with
consistency and confidence,'' in \emph{Proc. 34th Conf. Neural Inf. Process. Syst.
(NeurIPS)}, 2020, pp. 596--608.

\bibitem{ref34}
R. D. Magarey, T. B. Sutton, and C. L. Thayer, ``A simple generic infection model
for foliar fungal plant pathogens,'' \emph{Phytopathology}, vol. 95, no. 1,
pp. 92--100, Jan. 2005, doi: 10.1094/PHYTO-95-0092.

\bibitem{ref35}
H. J. Gohel, D. B. Mistry, S. K. Rathava, and B. B. Dhaduk, ``Management of leaf
blotch (\textit{Taphrina maculans} Butler) and leaf spot (\textit{Colletotrichum
capsici}) of turmeric,'' \emph{Indian Phytopathol.}, vol. 75, no. 2, pp. 487--491,
2022, doi: 10.1007/s42360-021-00452-x.

\end{thebibliography}

%% >>> Author biography blocks --- IEEE Access requires a short bio + photo for
%% each author. Template below; ~100 words each. <<<

\end{document}


% =====================================================================
% ||                                                                 ||
% ||   WORKING NOTES -- NOT PART OF THE PAPER.                       ||
% ||   LaTeX ignores everything past \end{document}, so this never   ||
% ||   appears in the PDF.  To submit: delete from this banner to    ||
% ||   the end of file.                                              ||
% ||                                                                 ||
% =====================================================================
%
% ---------- header notes (were above the LaTeX in the .md) ----------
% # LEAFSENSE — Full Journal Draft (IEEE Access, LaTeX)
%
% **Template:** IEEE Access (`ieeeaccess.cls`, download from the IEEE Author Center). Falls back to `IEEEtran` with `\documentclass[journal]{IEEEtran}` if needed.
% **Length target:** ~15 pages · **References:** 28
%
% > **Ref status (updated):** [24] and [26] are now **filled with real, Crossref-verified papers**, plus a new [28] (Rudin) supporting the interpretability argument. Remaining to fill: **[21], [22]** (your own prior papers — need venue/year/pages/DOI) and **[27]** (turmeric dataset author names).
%
% ## How to use this file
% 1. Copy everything between the `BEGIN LATEX` / `END LATEX` markers into `leafsense.tex`.
% 2. Every figure/table needing your input is marked `%% >>> PLACEHOLDER Pn <<<` with a description of exactly what to produce.
% 3. `★` in comments = number already real (from the CL results chapter). `⏳` = fill after experiments E1–E5 (see `01_SCOPE_AND_COVERAGE.md`).
% 4. **Verify all references against IEEE Xplore before submission.**
%
% ### Placeholder index
% | ID | Type | What to supply |
% |---|---|---|
% | P1 | Fig. 1 | LEAFSENSE system architecture diagram |
% | P2 | Fig. 2 | Inference pipeline + confidence routing |
% | P3 | Fig. 3 | Monthly on-device CL cycle flowchart |
% | P4 | Fig. 4 | MobileNetV3-Small layer groups G1–G4 |
% | P5 | Fig. 5 | Label-level fusion example ✅ DONE |
% | P6 | Fig. 6 | CatA accuracy grouped bar chart |
% | P7 | Fig. 7 | λ sensitivity curve |
% | P8 | Fig. 8 | CatB stability–plasticity scatter |
% | P9 | ~~Fig. 9~~ | Deployment photo + dashboard screenshot -- REMOVED 2026-07-29 (no installed unit; figure was never \ref'd) |
% | P10 | Table IV | Decision-tree risk results ✅ DONE (E1) |
% | P11 | Table V | Fusion vs image-only ✅ DONE (E2) |
% | P12 | Table XIII | On-device cycle + latency ✅ DONE (E3/E5, laptop-GPU proxy) |
% | P13 | Table XIV | Turmeric CL validation ✅ DONE (E4, Replay + naive, 3 seeds) |
%
% ---
%
% <!-- ============================ BEGIN LATEX ============================ -->
%
% ---------- pre-submission checklist (was below the LaTeX) ----------
% <!-- ============================= END LATEX ============================= -->
%
% ---
%
% ## Pre-submission checklist
%
% - [x] **E1** decision-tree experiment → Table IV (P10) filled + discussion written
% - [x] **E2** fusion evaluation → Table V (P11) filled + Fig. 5 (P5) built
% - [x] **E3/E5** cycle + latency → Table XIII (P12) filled + discussion written. NOTE: laptop-GPU proxy, NOT Jetson; TensorRT unavailable (ONNX Runtime substituted); detect/seg cited to [21]. Real on-device measurement still outstanding.
% - [x] **E4** turmeric CL validation → Table XIV (P13) filled + discussion written. Case 2 + Replay on data/07_turmeric_5cls, 2 cycles, 3 seeds: Replay 90.03 +/- 3.54 vs base 89.10 +/- 2.35, naive 87.85 +/- 2.47. Scope narrowed to Replay + naive reference; framed as a feasibility check (n=107, no pairwise McNemar reaches p<0.05).
% - [x] Produce figures P1–P8 (vector PDF) — all 8 done; P9 removed, not produced
% - [x] Fill implementation details in Section V-C -- DONE 2026-07-30: RTX 4050 Laptop GPU (6 GB, driver 595.79), Python 3.10 / PyTorch 2.5.1 / torchvision 0.20.1 / CUDA 12.1, FP32 (no AMP), Adam + wd 1e-4, class-weighted CE, ReduceLROnPlateau on val macro-F1, batch 32 (96 for Case 3A pretrain; replay 16+16), epoch/patience budgets per stage, single seed 42 with the limitation stated explicitly.
% - [x] ~~Replace refs [24] and [26]~~ — done, Crossref-verified (Li et al. 2024, CEA 224:109211; Lee & Yun 2023, Plant Methods 19:145). [28] Rudin 2019 added.
% - [x] Update refs [21], [22], [27] with actual venues/DOIs — done 2026-07-29 (ICIIS 2026 pp.395--400 + DOI; ICDTSA 2025 pp.7--16, no DOI; Mendeley authors via DataCite)
% - [x] **Verify every reference** — done 2026-07-29: all 35 checked, 20/21 DOIs resolve, IEEE format compliant. Ref [5] was FABRICATED and has been replaced (see 07_CHANGES_MADE.md 26).
% - [ ] Note: ref [26] has a published correction (doi: 10.1186/s13007-024-01140-3) — check whether it affects anything you rely on
% - [ ] Add author biographies and photos
% - [ ] Run through IEEE Access page-count and formatting check (~15 pages)
```

<!-- ============================= END LATEX ============================= -->

---

## Pre-submission checklist

- [x] ~~**E1** decision-tree experiment~~ — done. Table IV filled (5-model comparison, DT deployed: 86.25% acc / 85.6 macro-F1). Code in `experiments/part2_env_risk/`. Refs [34],[35] added. §VI simulation disclosure added.
- [x] ~~**E2** fusion evaluation~~ — done. Table V filled (fusion 87.5% acc / 71.75 macro-F1 vs. 60.4%/29.6 vision-only, 35.4%/20.6 sensor-only, on 48 constructed episodes). Fig. 5 (P5) built from real episode data. Code in `experiments/part3_fusion_eval/`. §VI fusion-evaluation disclosure added.
- [x] ~~**E3/E5** cycle + latency~~ — Table XIII (P12) filled + discussion written. ⚠️ **Laptop RTX 4050 proxy, NOT Jetson**; TensorRT unavailable → ONNX Runtime substituted; detect/seg cited to [21]. Retrain 373.2 s / 155.6 MB VRAM; FP16 gate Δ ≈ 0; compiled classification 3.8 ms/img (2.7× vs eager). Code in `experiments/part4_ondevice_proxy/`. §VI on-device disclosure added. **Real on-device measurement still outstanding.**
- [x] ~~**E4** turmeric CL validation~~ — done 2026-07-29. Table XIV (P13) filled + discussion written. Splits rebuilt as `data/07_turmeric_5cls/` (test/val frozen from the shipped split so the existing phase-3 checkpoint stays comparable); Case 2 + Replay, 2 cycles, 3 seeds → **Replay 90.03% ± 3.54** vs base 89.10% ± 2.35, naive 87.85% ± 2.47. Scope narrowed to Replay + a naive reference. Code in `experiments/part5_turmeric_cl/`. ⚠️ Framed as a **feasibility check** — n=107, no pairwise McNemar reaches p<0.05.
- [x] ~~Produce figures P1–P9~~ — **P1–P8 done (8 final figures)**; **P9 removed 2026-07-29** — no installed unit to photograph, a dashboard-only figure does not carry the deployment claim, and the figure was never referenced from the body text. Deployment wording softened to conditional throughout §VI.
- [x] ~~Fill implementation details in Section V-C~~ — done 2026-07-30. GPU model recovered from `experiments/CONTINUAL_RESOURCE_REPORT.md` (**NVIDIA GeForce RTX 4050 Laptop GPU**, 6 GB, driver 595.79 — the *same* device as the E3 proxy). Three new paragraphs added: hardware/software, optimization, and an explicit **single-seed (42) disclosure**. ⚠️ While cross-checking the configs against the paper, **Table III's Case 2 learning-rate rows were found to be wrong** — corrected, see `07_CHANGES_MADE.md` §30.
- [x] ~~Replace refs [24] and [26]~~ — done, Crossref-verified (Li et al. 2024, CEA 224:109211; Lee & Yun 2023, Plant Methods 19:145). [28] Rudin 2019 added.
- [x] ~~Update refs [21], [22], [27]~~ — done 2026-07-29. [21] ICIIS, Peradeniya, Jan. 2026, pp. 395–400, doi `10.1109/ICIIS69028.2026.11450780` (Crossref-verified). [22] ICDTSA, Kilinochchi, May 2025, pp. 7–16, confirmed **no DOI**. [27] real Mendeley author lists via the **DataCite** API (Mendeley DOIs are not in Crossref).
- [x] ~~Verify [31] FedAvg / [33] FixMatch page ranges~~ — done. [31] confirmed PMLR 54:1273–1282 from the publisher. [33] paper + authors confirmed in the NeurIPS 2020 index; pagination unconfirmable from a primary source (NeurIPS publishes none) — 596–608 is the Curran print volume, documented as such. Both `%% VERIFY` markers cleared.
- [x] ~~**Verify every reference**~~ — done 2026-07-29: all 35 audited, **20 of 21 DOIs resolve**, IEEE format compliant throughout. ⚠️ **Ref [5] was fabricated** (dead DOI; claimed pages fall in a gap between real articles) — removed and replaced with Khan et al., *Front. Plant Sci.* 14:1308528. Inherited from the prior conference paper. See `07_CHANGES_MADE.md` §26.
- [ ] Note: ref [26] has a published correction (doi: 10.1186/s13007-024-01140-3) — check whether it affects anything you rely on
- [ ] **Compile in Overleaf and verify table layout** — the Table XI/XII/XIII collision fix was made with no local LaTeX toolchain and is **not visually verified**
- [ ] Add author biographies and photos
- [ ] Run through IEEE Access page-count and formatting check (~15 pages)
