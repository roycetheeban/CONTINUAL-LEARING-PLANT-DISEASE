# Literature Review — LEAFSENSE Journal
**Answers readme item 5.** Citation numbers `[n]` match the reference list in `04_FULL_PAPER_LATEX.md` exactly.

> ⚠️ **Verification note:** references [1]–[20] are standard, well-known works whose bibliographic details are reliable. References [21]–[27] include the author's own prior papers and domain/dataset sources. **Before submission, verify every DOI, volume, page range, and year against IEEE Xplore / the publisher** — do not submit unverified bibliographic data.

---

## A. Edge-Optimized Plant Disease Detection

Automated plant disease diagnosis has converged on convolutional architectures trained on large leaf-image corpora, with the PlantVillage dataset (54,306 images, 38 classes, 14 species) [1] serving as the field's de-facto benchmark since Mohanty *et al.* demonstrated >99% classification accuracy with deep CNNs [2]. High-capacity backbones such as ResNet [3] and EfficientNet [4] dominate reported accuracy tables, but their 25–500 MB footprints and server-class compute requirements make them unsuitable for the resource-constrained hardware that field deployment demands [5].

The response has been a sustained push toward lightweight architectures. MobileNetV3 [6] applies hardware-aware neural architecture search, squeeze-and-excite blocks and inverted residuals to reach accuracy comparable to far larger networks at 10–20× fewer parameters; MobileNetV3-Small in particular has become a standard choice for agricultural edge classifiers. Compression compounds these gains: magnitude-based structured pruning removes 50–90% of parameters with minimal degradation [7], and post-training quantization (FP16/INT8) yields a further 2–4× size reduction without retraining [8]. On the detection side, the YOLO family has evolved specifically toward edge inference, with YOLOv8n offering an anchor-free head at only 3.2 M parameters [9]; agricultural applications of YOLO variants report strong precision on rice, bean and PlantVillage imagery [10], and architectural enhancements such as SerpensGate-YOLOv8 improve mAP over the baseline for plant disease targets [11].

Systems that chain detection, segmentation and classification into modular pipelines report favourable accuracy/latency trade-offs relative to monolithic models, since each stage can be independently optimized or selectively disabled according to the compute budget [12]. Our own prior work follows this line: a lightweight modular pipeline combining YOLOv8n detection (mAP@0.5 = 0.632), YOLOv8n-seg segmentation (99.5% pixel accuracy, 0.915 mIoU) and a pruned, INT8-quantized MobileNetV3-Small classifier (89.81% on five turmeric classes) fits the entire system into a 17.6 MB footprint suitable for Jetson-class hardware [21]. Occlusion — overlapping foliage, pests, shadows — remains a principal failure mode for field imagery; GAN-based inpainting has been applied to restore occluded leaf regions, and our earlier study showed that a hybrid GAN combining attention with VGG perceptual loss reaches 23.16 dB PSNR / 0.6296 SSIM on synthetically occluded PlantVillage leaves, substantially outperforming standard and attention-only GAN baselines [22]. Together these works establish an accurate, deployable *inference* stack — but both explicitly close by naming **on-device continual learning** as unfinished future work, which is precisely the gap the present paper addresses.

## B. Continual Learning and Catastrophic Forgetting

Neural networks trained sequentially on non-stationary data overwrite previously acquired representations, a failure mode termed catastrophic forgetting [13]. The continual learning literature groups mitigations into three families [14].

**Regularization-based** methods constrain weight updates according to estimated parameter importance. Elastic Weight Consolidation (EWC) [15] computes a Fisher Information Matrix over previous-task data and adds a quadratic penalty proportional to the squared deviation of each parameter from its prior value, weighted by importance; Synaptic Intelligence [16] estimates importance online along the optimization path. These methods are attractive at the edge because they add no data storage and only a modest compute overhead, but their protection strength depends critically on the penalty coefficient λ — a hyperparameter whose tuning is frequently reported as fixed or unexplained in applied work.

**Replay-based** methods retain a small buffer of past exemplars and interleave them with the incoming stream. iCaRL [17] combines a herding-based exemplar set with a nearest-mean-of-exemplars classifier and remains a standard class-incremental baseline; subsequent analyses show that even small, stratified buffers substantially reduce forgetting, and that much of the damage in class-incremental settings is concentrated in classifier-head bias rather than the backbone [18]. The cost is storage and per-batch throughput — both scarce on embedded devices.

**Architecture / parameter-isolation** methods dedicate disjoint parameters to different tasks. Progressive Networks [19] instantiate a new column per task with lateral connections; PackNet [20] iteratively prunes and freezes subsets of weights, guaranteeing zero forgetting by construction. Zero forgetting comes at the price of monotonic model growth and — as our results demonstrate — severely limited plasticity when a newly instantiated pathway cannot exploit the frozen backbone's representation.

Recent work on the stability–plasticity balance has shown that *how* the network is trained matters as much as *which* CL method is applied: SLCA [23] demonstrates that slowing the learning rate of representation layers while aligning the classifier largely resolves progressive overfitting in continual fine-tuning, motivating the layer-wise schedule used in this work.

Continual learning applied specifically to plant disease has begun to appear, with studies exploring class-incremental crop-disease recognition and replay strategies for agricultural streams [24]. However, these evaluations remain almost exclusively *offline and server-side*: methods are compared on curated benchmark splits with GPU-class resources, and neither the retraining budget of an embedded device nor the data-acquisition mechanism that would supply the stream in the field is modelled. Reported comparisons also rarely control for initialization, so the confounding effect of pretraining quality on apparent CL method ranking is left unmeasured.

## C. Multimodal Sensing and Fusion in Agricultural AIoT

Disease expression is driven by environment: sustained high humidity, leaf wetness and temperature bands govern fungal infection pressure, so vision-only diagnosis discards a signal that is cheap to acquire and highly predictive [25]. AIoT deployments increasingly pair imaging with CO₂, temperature, humidity and soil-moisture sensing, fusing the two modalities at feature level (concatenating learned embeddings), decision level (combining independent model outputs), or label level (combining discrete predicted labels/counts with a categorical risk state). For edge deployment, decision- and label-level fusion are preferable: the modalities remain independently trainable and independently updatable, the fusion rule is interpretable to the farmer, and a sensor fault degrades one branch rather than corrupting a joint embedding. Interpretable models such as decision trees are well matched to the environmental branch, since agronomic thresholds are naturally expressible as rules and the resulting risk state can be audited by an agronomist [26]. Nevertheless, most published agricultural fusion systems assume continuous connectivity and a static model, leaving unanswered how the fusion layer behaves when the vision branch is *itself changing* under continual updates.

## D. Research Gap

Three gaps emerge from the above and jointly motivate this work.

1. **CL methods are benchmarked, but not deployed.** Existing plant-disease CL studies evaluate offline with server resources. None reports the on-device cost of a retraining cycle — wall-clock time, peak memory, runtime re-compilation and the accuracy shift introduced by the deployment precision (FP16) — nor a deployment gate that prevents a regressed model from reaching production.
2. **Initialization is an uncontrolled confound.** Reported method rankings rarely separate the contribution of pretraining from that of the CL algorithm. We evaluate every method under three initializations (scratch, ImageNet, in-domain PlantVillage) and show that transfer quality sets the ceiling on achievable CL performance, and that λ sensitivity itself depends on the backbone's domain specificity.
3. **The data engine and the fusion layer are missing.** Prior work assumes a labelled stream materializes. A deployable system must *generate* its own stream — confidence-routing predictions into an auto-labelled retrain buffer versus a human/LLM relabel queue — and must fuse the resulting predictions with environmental risk into an actionable output. Neither of our prior papers [21], [22] closed this loop, and to our knowledge no published system combines an autonomous data engine, on-device metric-gated continual retraining, and label-level environmental fusion in a single validated edge product.

LEAFSENSE addresses all three: a 25-run controlled CL benchmark spanning two incremental scenarios, three initializations and five methods; a layer-wise training strategy with split-LR head expansion; and an offline-first Jetson deployment in which the winning recipe drives a self-gated monthly retraining cycle fused with decision-tree environmental risk.

---

## Reference List (as used above; full IEEE format in the LaTeX draft)

| # | Reference | Used in |
|---|---|---|
| [1] | Hughes & Salathé, PlantVillage dataset, arXiv:1511.08060, 2015 | A |
| [2] | Mohanty, Hughes & Salathé, *Front. Plant Sci.*, 2016 | A |
| [3] | He *et al.*, ResNet, CVPR 2016 | A |
| [4] | Tan & Le, EfficientNet, ICML 2019 | A |
| [5] | Sharma & Patel, edge computing challenges, *IEEE IoT J.*, 2022 | A |
| [6] | Howard *et al.*, Searching for MobileNetV3, ICCV 2019 | A |
| [7] | Zhu & Gupta, To prune or not to prune, ICLR 2018 | A |
| [8] | Banner *et al.*, Post-training 4-bit quantization, NeurIPS 2019 | A |
| [9] | Jocher *et al.*, YOLOv8/Ultralytics, 2023 | A |
| [10] | Aldakheel *et al.*, YOLOv4 leaf disease, *Front. Plant Sci.*, 2024 | A |
| [11] | Miao *et al.*, SerpensGate-YOLOv8, *Front. Plant Sci.*, 2024 | A |
| [12] | Guan *et al.*, lightweight plant disease model, *Front. Plant Sci.*, 2023 | A |
| [13] | French, Catastrophic forgetting, *Trends Cogn. Sci.*, 1999 | B |
| [14] | De Lange *et al.*, CL survey, *IEEE TPAMI*, 2022 | B |
| [15] | Kirkpatrick *et al.*, EWC, *PNAS*, 2017 | B |
| [16] | Zenke *et al.*, Synaptic Intelligence, ICML 2017 | B |
| [17] | Rebuffi *et al.*, iCaRL, CVPR 2017 | B |
| [18] | Wu *et al.*, Large-scale incremental learning (bias correction), CVPR 2019 | B |
| [19] | Rusu *et al.*, Progressive Neural Networks, arXiv, 2016 | B |
| [20] | Mallya & Lazebnik, PackNet, CVPR 2018 | B |
| [21] | **Own prior paper** — Lightweight Modular Pipeline for Edge-Optimized Leaf Disease Detection | A, D |
| [22] | **Own prior paper** — GAN-Based Leaf Image Inpainting for Occlusion Recovery | A, D |
| [23] | Zhang *et al.*, SLCA, ICCV 2023 | B |
| [24] | CL for plant disease / class-incremental crop recognition (**verify exact source**) | B |
| [25] | Agrios, *Plant Pathology* (environment–disease relationship) | C |
| [26] | Interpretable ML / decision trees for agricultural risk (**verify**) | C |
| [27] | Turmeric disease dataset, Mendeley Data | Setup |
