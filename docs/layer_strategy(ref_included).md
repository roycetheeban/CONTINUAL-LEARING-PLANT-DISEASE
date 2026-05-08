# MobileNetV3-Small — Layer-wise Training Strategy
## Edge Plant/Leaf CL Research | All Cases & CL Cycles

> **References:**
> - Howard et al. (2019) — Searching for MobileNetV3. ICCV 2019.
> - Zhang et al. (2023) — SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model. ICCV 2023.
> - Kirkpatrick et al. (2017) — Overcoming catastrophic forgetting in neural networks. PNAS 2017.
> - Rusu et al. (2016) — Progressive Neural Networks. arXiv 2016.
> - Mallya & Lazebnik (2018) — PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning. CVPR 2018.
> - Rolnick et al. (2019) — Experience Replay for Continual Learning. NeurIPS 2019.
> - Buzzega et al. (2020) — Rethinking Experience Replay: A Bag of Tricks for Continual Learning. ICPR 2020.
> - Liu et al. (2021) — AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning. arXiv 2021.
> - Zhao et al. (2021) — Partial is Better Than All: Revisiting Fine-tuning Strategy for Few-Shot Learning. AAAI 2021.
> - Mirzadeh et al. (2022a) — Architecture Matters in Continual Learning. arXiv 2022.
> - Mirzadeh et al. (2022b) — Wide Neural Networks Forget Less Catastrophically. ICML 2022.
> - Zhang et al. (2022) — ACL: Adapter-based Continual Learning of Diseases from NIH Chest X-rays. MICCAI 2023.
> - Yoo et al. (2024) — Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning. ICML 2024.
> - Verwimp et al. (2020) — The Effectiveness of Memory Replay in Large Scale Continual Learning. arXiv 2020.
> - Shin et al. (2017) — Continual Learning with Deep Generative Replay. NeurIPS 2017.

---

## 1. MobileNetV3-Small Architecture — Layer Groups

MobileNetV3-Small in PyTorch (`torchvision`) is structured as follows. For training control, layers are divided into 4 functional groups based on what kind of features they learn and how sensitive they are to forgetting.

```
MobileNetV3-Small
│
├── GROUP 1 — Early Layers [features 0–3]          ← Generic low-level features
│   ├── features[0]:  Conv2d (3→16, stride 2) + BN + Hardswish
│   ├── features[1]:  InvertedResidual (SE, ReLU)  16→16
│   ├── features[2]:  InvertedResidual (SE, ReLU)  16→24, stride 2
│   └── features[3]:  InvertedResidual (SE, ReLU)  24→24
│
├── GROUP 2 — Mid Layers [features 4–8]             ← Plant/domain features
│   ├── features[4]:  InvertedResidual (SE, H-Swish) 24→40, stride 2
│   ├── features[5]:  InvertedResidual (SE, H-Swish) 40→40
│   ├── features[6]:  InvertedResidual (SE, H-Swish) 40→40
│   ├── features[7]:  InvertedResidual (SE, H-Swish) 40→48
│   └── features[8]:  InvertedResidual (SE, H-Swish) 48→48
│
├── GROUP 3 — Late Layers [features 9–12]           ← Disease-discriminative features
│   ├── features[9]:  InvertedResidual (SE, H-Swish) 48→96, stride 2
│   ├── features[10]: InvertedResidual (SE, H-Swish) 96→96
│   ├── features[11]: InvertedResidual (SE, H-Swish) 96→96
│   └── features[12]: Conv2d (96→576) + BN + Hardswish   ← Last conv, expands 6×
│
├── avgpool: AdaptiveAvgPool2d(1,1)
│
└── GROUP 4 — Classifier Head [classifier 0–3]      ← Fully task-specific
    ├── classifier[0]: Linear(576 → 1024)
    ├── classifier[1]: Hardswish
    ├── classifier[2]: Dropout(p=0.2)
    └── classifier[3]: Linear(1024 → N_classes)    ← Replace per task
```

### Why These Groups?

| Group | Feature Type | Sensitivity | Transfer Ability |
|---|---|---|---|
| G1 Early | Edges, textures, colour | Low — converges fast | Highest — universal across domains |
| G2 Mid | Leaf venation, surface patterns, colour spots | Medium | High within plant domain |
| G3 Late | Lesion shapes, discolouration profiles, class margins | High | Medium — disease-specific |
| G4 Head | Class probabilities | Always task-specific | None — always replaced |

This matters because:
- Early layers learned in ImageNet training already detect the same edges in leaf images as in anything else — there is no benefit to updating them
- Mid layers contain plant-specific patterns that transfer well within the plant domain — they need gentle adaptation, not retraining
- Late layers are where disease-specific features live — these need the most attention per training case
- The classifier head is always fully retrained or fine-tuned with the highest LR

---

## 2. Part 1 — Layer Strategy Per Case

### Case 1 — Train From Scratch (→ M1)

**All layers train simultaneously. No freezing.**

| Group | Status | Learning Rate |
|---|---|---|
| G1 Early | ✅ Training | 1e-3 |
| G2 Mid | ✅ Training | 1e-3 |
| G3 Late | ✅ Training | 1e-3 |
| G4 Head | ✅ Training | 1e-3 |

No layer-wise LR differentiation since there are no pre-trained weights to protect. Uniform LR with ReduceLROnPlateau scheduler.

```python
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

---

### Case 2 — ImageNet Fine-tune (→ M2)

**Two-phase training with progressive unfreezing.**

#### Phase i — Classifier only (10–15 epochs)
| Group | Status | Learning Rate |
|---|---|---|
| G1 Early | ❄ Frozen | 0 |
| G2 Mid | ❄ Frozen | 0 |
| G3 Late | ❄ Frozen | 0 |
| G4 Head | ✅ Training | 1e-3 |

**Rationale:** ImageNet features are already excellent for leaf classification. Jumping straight to training all layers would destroy those features with a mismatched 5-class tomato dataset (too small for full fine-tuning). Train the classifier first until it converges.

```python
for param in model.features.parameters():
    param.requires_grad = False
optimizer = Adam(model.classifier.parameters(), lr=1e-3)
```

#### Phase ii — Unfreeze Group 3 (10 more epochs)
| Group | Status | Learning Rate |
|---|---|---|
| G1 Early | ❄ Frozen | 0 |
| G2 Mid | ❄ Frozen | 0 |
| G3 Late | ~ Partial | 1e-4 (0.1× head) |
| G4 Head | ✅ Training | 1e-3 |

**Rationale:** Group 3 holds disease-discriminative features. ImageNet's version of these features is generic object-level (fur texture, wheel shapes) — these need updating for leaf diseases. Group 1 and 2 stay frozen because ImageNet low/mid-level features (edges, colour gradients) transfer perfectly to leaves.

```python
for param in model.features[9:].parameters():
    param.requires_grad = True

optimizer = Adam([
    {'params': model.features[9:].parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

> **Do not unfreeze Group 1 or 2 for Case 2.** The 5-class tomato dataset is too small — full backbone fine-tuning will overfit. The information gain from updating low-level features does not justify the risk.

---

### Case 3A — PlantVillage Pre-training 26 Classes (→ backbone)

**All layers train on 26 non-tomato classes.**

| Group | Status | Learning Rate |
|---|---|---|
| G1 Early | ✅ Training | 1e-3 |
| G2 Mid | ✅ Training | 1e-3 |
| G3 Late | ✅ Training | 1e-3 |
| G4 Head | ✅ Training (26-class) | 1e-3 |

Same as Case 1 but the goal is backbone quality, not final classification. Use capped class-weighted loss (max 3×) due to 15:1 imbalance in 26-class data. Classifier head here is 26 outputs — it will be discarded after pre-training.

**Key output:** `pretrained_backbone.pth` — state dict of `model.features` only (not the classifier). This is the most reusable artifact in the entire pipeline.

```python
# Save only the feature extractor
torch.save(model.features.state_dict(), 'pretrained_backbone.pth')
```

---

### Case 3B — PlantVillage Fine-tune on 5 Tomato (→ M3)

**Two-phase training. Lower LRs than Case 2 — less domain shift.**

#### Phase i — Late layers + head (10–15 epochs)
| Group | Status | Learning Rate |
|---|---|---|
| G1 Early | ❄ Frozen | 0 |
| G2 Mid | ❄ Frozen | 0 |
| G3 Late | ~ Partial | 5e-5 (very low) |
| G4 Head | ✅ Training | 5e-4 |

**Rationale:** The backbone already has plant-domain features from PlantVillage pre-training — even Group 3 already contains leaf patterns. So the LRs are 2–5× lower than Case 2. Group 3 gets 5e-5 (not frozen) because it needs small adjustments from 26-class generic disease patterns to 5 specific tomato classes.

#### Phase ii — Unfreeze Group 2 (optional, 5–10 epochs)
| Group | Status | Learning Rate |
|---|---|---|
| G1 Early | ❄ Frozen | 0 |
| G2 Mid | ~ Partial | 1e-5 (very very low) |
| G3 Late | ~ Partial | 5e-5 |
| G4 Head | ✅ Training | 5e-4 |

**Rationale:** Group 2 in M3 already learned plant mid-level features (venation, surface texture) from the 26-class pre-training. Fine-tuning it at 1e-5 allows tomato-specific patterns (the distinctive orange-brown of bacterial spot, the concentric rings of early blight) to emerge without disrupting the general plant representation.

```python
# Phase i
for param in model.features[:9].parameters():
    param.requires_grad = False
optimizer = Adam([
    {'params': model.features[9:].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 5e-4}
])

# Phase ii (unfreeze Group 2)
for param in model.features[4:9].parameters():
    param.requires_grad = True
optimizer = Adam([
    {'params': model.features[4:9].parameters(), 'lr': 1e-5},
    {'params': model.features[9:].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 5e-4}
])
```

> **Group 1 (features[0–3]) stays frozen in ALL phases of Case 3B.** The pre-trained backbone already has excellent low-level plant features — there is zero benefit to updating these with 5-class tomato data.

---

## 3. Part 2 — CL Cycle Layer Strategy

> **Core principle applied:** SLCA (Slow Learner with Classifier Alignment, Zhang et al., 2023) — *"Selectively reducing the learning rate of representation layers can almost resolve progressive overfitting in continual learning."* Combined with your CL methods (EWC, Replay, Isolation), this gives compounded protection.

---

### Category A — Same Classes, New Data Stream

**Progressive Layer Freezing across cycles.** As cycles advance, more layers are effectively frozen. Each cycle the model is more settled — smaller updates preserve stability.

#### Cycle 1
| Group | Status | LR (all CL methods) | Rationale |
|---|---|---|---|
| G1 Early | ❄ Frozen | 0 | Universal features — no reason to ever update |
| G2 Mid | ❄ Frozen | 0 | Plant features already optimal after Part 1 |
| G3 Late | ~ Partial | 1e-4 | May need minor drift correction for new stream distribution |
| G4 Head | ✅ Training | 5e-4 | Adapts class boundaries to new stream samples |

#### Cycle 2
| Group | Status | LR (all CL methods) | Change from Cycle 1 |
|---|---|---|---|
| G1 Early | ❄ Frozen | 0 | Same |
| G2 Mid | ❄ Frozen | 0 | Same |
| G3 Late | ~ Partial | 5e-5 | **↓ 2× lower** — already adapted in Cycle 1 |
| G4 Head | ✅ Training | 2e-4 | **↓ 2.5× lower** — model converging |

**Early stopping criterion for all Cat A CL cycles:**
> Stop training when old-class val F1 drops more than 3% from the value at cycle start. This uses the shared val set as a real-time forgetting detector — stronger signal than watching new-data loss.

```python
# EWC Cat A example — Cycle 1
optimizer = Adam([
    {'params': model.features[9:].parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 5e-4}
])
# Freeze G1+G2
for param in model.features[:9].parameters():
    param.requires_grad = False
```

---

### Category B — New Class Introduction (5 → 7 Classes)

**More nuanced — needs plasticity for T6, T7 while protecting T1–T5.**

#### Cycle 1 — New Class Introduction
| Group | Status | LR | Difference from Cat A |
|---|---|---|---|
| G1 Early | ❄ Frozen | 0 | Same |
| G2 Mid | ~ Very partial | 5e-5 | ← **Slightly unfrozen** (new tomato classes need minor mid-feature adaptation) |
| G3 Late | ~ Partial | 2e-4 | ← **Higher than Cat A** (must learn T6, T7 discriminative features) |
| G4 Head | ✅ Training | old neurons: 1e-4 / new neurons: 1e-3 | ← **Split LR classifier** |

**Split-LR Classifier — critical for Cat B:**

```python
# Expand classifier: 5 → 7
old_weight = model.classifier[-1].weight.data  # shape [5, 1024]
old_bias   = model.classifier[-1].bias.data

new_layer = nn.Linear(1024, 7)
new_layer.weight.data[:5] = old_weight   # preserve old class neurons
new_layer.bias.data[:5]   = old_bias
model.classifier[-1] = new_layer

# Split-LR optimizer
optimizer = Adam([
    {'params': model.features[4:9].parameters(), 'lr': 5e-5},   # G2 very slow
    {'params': model.features[9:].parameters(),  'lr': 2e-4},   # G3 moderate
    # Old 5 classifier neurons — protect them
    {'params': [model.classifier[-1].weight[:5], model.classifier[-1].bias[:5]], 'lr': 1e-4},
    # New 2 classifier neurons — let them learn fast
    {'params': [model.classifier[-1].weight[5:], model.classifier[-1].bias[5:]], 'lr': 1e-3},
])
```

#### Cycle 2
| Group | Status | LR | Change from Cycle 1 |
|---|---|---|---|
| G1 Early | ❄ Frozen | 0 | Same |
| G2 Mid | ~ Very partial | 1e-5 | ↓ 5× lower |
| G3 Late | ~ Partial | 1e-4 | ↓ 2× lower |
| G4 Head | ✅ Training | old: 5e-5 / new: 5e-4 | Both reduced as model converges |

---

## 4. Experience Replay — Layer Strategy & Layer Effect Analysis

Experience Replay is a **data-level** method, not a weight-level method. It does not change which layers are frozen — the same base CL layer strategy applies. However, because old-class samples are mixed into every training batch, the model receives a natural forgetting signal from the data itself.

### 4.1 How Replay Interacts with Each Layer Group

Understanding *which layers benefit most* from experience replay — and which remain vulnerable — is critical to setting correct LRs and designing the replay buffer correctly.

#### Group 1 (Early — features[0–3]) — Not Affected, Not Needed

Early layers encode low-level features (edges, colour gradients, textures) that are **universally stable** across all tomato disease classes. Replay has no meaningful effect here because these features are not the site of forgetting. Verwimp et al. (2020) showed that even in large-scale CL scenarios, intermediate representations continue to drift during training — but this drift originates primarily in deeper, task-discriminative layers, not in early generic feature detectors. Since G1 is frozen throughout CL, this is consistent: replay is not needed there and applying it would waste buffer capacity.

#### Group 2 (Mid — features[4–8]) — Passive Protection via Data Distribution

Mid layers encode plant-domain features (venation, surface texture, colour spot patterns). In Category A CL (same classes, new stream data), G2 is frozen and replay has no direct effect on it. In Category B (new class introduction), G2 is partially unfrozen at a very low LR (5e-5). Here replay helps indirectly: by including old-class images in every training batch, the gradient signal seen by G2 remains consistent with the full old-class distribution, preventing the slow distributional drift that would otherwise accumulate even at low LRs.

> **Key insight (Yoo et al., ICML 2024):** Layerwise Proximal Replay (LPR) found that replay-based methods suffer from *unstable optimization trajectories* in intermediate layers — gradients from old and new data can point in conflicting directions. The solution is to constrain updates in hidden layers to be gradual. This is exactly what the low LR on G2 (5e-5) achieves in our setup: replay provides the distribution signal, the low LR prevents the trajectory instability. These two mechanisms are complementary.

#### Group 3 (Late — features[9–12]) — Primary Replay Target: Highest Benefit

Late layers are where catastrophic forgetting is most severe. These layers encode the high-level disease-discriminative features (lesion morphology, discolouration profiles, inter-class margin boundaries) that are most task-specific and therefore most vulnerable to overwriting during new data training.

Verwimp et al. (2020) directly measured that *"intermediate representations still undergo a distributional shift"* even when standard replay is applied — because vanilla replay constrains only the input-output mapping, not the intermediate feature space. This means: **G3 features will still drift slightly even with replay**, which is why G3 cannot be fully frozen and must retain a non-zero LR (1e-4 for Cycle 1). The replay buffer provides a corrective gradient pull, but G3 needs to be free enough to absorb that correction.

> **Practical implication for G3:** Because the replay buffer provides data-level gradient correction at G3, G3 can tolerate a slightly higher LR in the replay method than EWC (where no such data-level correction exists). However, we keep the LRs the same across methods for fair paper comparison — the method difference should appear in the results, not be confounded by different training hyperparameters. See Section 4.2 for the strict rule.

#### Group 4 (Classifier Head) — Most Directly Protected by Replay

The classifier head is the layer most immediately and severely affected by catastrophic forgetting in class-incremental scenarios (Buzzega et al., 2020; Shin et al., 2017). Replay is most effective here: by including old-class samples in every batch, the old class output neurons receive direct gradient updates that prevent them from drifting. Buzzega et al. (2020) confirmed that *"rehearsal-based methods have shown to be far better for incrementally learning new classes"* than regularization methods like EWC — and this superiority originates primarily in the classifier head's ability to directly receive old-class gradient signal from replayed data.

In Category B with split-LR, old class neurons are held at 1e-4 and new class neurons at 1e-3. The replay buffer reinforces this: old-class images produce gradients only on old neurons (via CrossEntropy), while new-class images update new neurons. The two learning signals do not compete at the output layer.

### 4.2 Category A — Same Classes

| Group | Status | Cycle 1 LR | Cycle 2 LR | vs EWC |
|---|---|---|---|---|
| G1 Early | ❄ Frozen | 0 | 0 | Same |
| G2 Mid | ❄ Frozen | 0 | 0 | Same |
| G3 Late | ~ Partial | 1e-4 | 5e-5 | Same — keep consistent for fair comparison |
| G4 Head | ✅ Training | 5e-4 | 2e-4 | Same |

> **Keep the same LR values as EWC for fair method comparison in the paper.** The difference between EWC and Replay shows up in the results, not in the LR — do not confound the comparison by using different LRs per method.

**What changes vs EWC — the training loop only:**

```python
# Experience Replay training loop — Cat A Cycle 1
for (new_inputs, new_labels), (replay_inputs, replay_labels) in zip(
        new_stream_loader, replay_buffer_loader):

    # Mix old and new in same batch
    inputs = torch.cat([new_inputs, replay_inputs])
    labels = torch.cat([new_labels, replay_labels])

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)   # same weighted CrossEntropy — no extra penalty term
    loss.backward()
    optimizer.step()
```

The optimizer, frozen layers, and LR values are identical to EWC. Only the training data composition changes.

### 4.3 Category B — New Classes

| Group | Status | Cycle 1 LR | Cycle 2 LR |
|---|---|---|---|
| G1 Early | ❄ Frozen | 0 | 0 |
| G2 Mid | ~ Very partial | 5e-5 | 1e-5 |
| G3 Late | ~ Partial | 2e-4 | 1e-4 |
| G4 Head | ✅ Training | old:1e-4 / new:1e-3 | old:5e-5 / new:5e-4 |

**Replay buffer update after Cycle 1 (Cat B only):**
After Cat B Cycle 1 training completes, add a stratified sample of T6 and T7 from `cl_cycle1_stream/` into the replay buffer. Cycle 2 will then replay all 7 classes — old and new — to prevent forgetting of both.

```python
# After Cat B Cycle 1 — update buffer to include T6, T7
def update_replay_buffer_catB(buffer_manifest, cycle1_new_class_paths, n_per_class=100):
    for class_name, paths in cycle1_new_class_paths.items():
        sampled = random.sample(paths, min(n_per_class, len(paths)))
        buffer_manifest[class_name] = sampled   # add new classes to buffer
    return buffer_manifest
```

### 4.4 Why Replay Does Not Fully Solve Layer-level Drift

A key finding from Verwimp et al. (2020) is that vanilla experience replay — replaying only input-output pairs — is *insufficient* to fully stabilise intermediate layer representations. Their experiments showed measurable distributional shift in hidden activations even when replay was applied. This occurs because the CrossEntropy loss constrains only the final output, not the internal feature geometry.

**What this means for our setup:**
- G3 features will still shift somewhat even with replay. This is acceptable because G3 is unfrozen with a low LR — it can accommodate this shift. If G3 were frozen, the shift would accumulate silently and eventually degrade classifier performance.
- The replay buffer size (min 100 images per class, ~517 total) is sufficient to prevent large-scale forgetting at the classifier level, which is replay's primary job. Intermediate layer stabilisation is provided by the low LR and early stopping on val F1.
- For future work, Compressed Activation Replay (Verwimp et al., 2020) — which saves compressed hidden activations alongside images — could further reduce G3 drift. This is out of scope for the current study but worth noting as an extension.

---

## 5. Parameter Isolation — Layer Strategy & Layer Effect Analysis

Parameter Isolation is fundamentally different from EWC and Replay. It is a **structural** method, not a gradient or data method. The concept: old layers are locked permanently, new parameters are added to handle new data. Forgetting is **mathematically impossible** for old classes because their weights never change.

### 5.1 Layer-by-Layer Effect Analysis for Isolation Methods

Understanding the role of each layer group in parameter isolation is essential for designing the adapter (Cat A) and branch (Cat B) architectures correctly. The key insight from the isolation literature is that **not all layers are equally worth isolating** — the optimal strategy isolates late/task-specific layers most aggressively while allowing shared early layers to be reused across tasks.

#### Group 1 (Early — features[0–3]) — Always Frozen, Shared Across All Tasks

Early layers encode features that are genuinely universal: edges, colour gradients, basic textures. In parameter isolation, these layers are never updated after initial training. This is supported by multiple lines of evidence:

- Rusu et al. (2016) in Progressive Neural Networks showed that lateral connections from old columns transfer low-level features to new columns without modification — early layer features are already close to optimal for any new visual task in the same domain.
- Liu et al. (AutoFreeze, 2021) showed that early layers in DNNs converge fastest and can be frozen earliest without accuracy loss — their gradient magnitudes shrink rapidly once the model has seen enough domain data.
- Zhang et al. (ACL for Disease CL, MICCAI 2023) confirmed in medical image CL that fixing the shared feature extractor (equivalent to G1+G2 in our setup) while only training lightweight task-specific adapters yields strong performance — the frozen early layers provide universal feature extraction that benefits all disease classes equally.

**For our setup:** G1 is frozen in both the original model and in any added adapter. The adapter inserted at the G3/G4 boundary inherits the full feature extraction chain from G1 without modification. This is zero-forgetting by design.

#### Group 2 (Mid — features[4–8]) — Frozen in Cat A, Reused in Cat B Branch

Mid layers encode plant-domain features (venation patterns, surface texture, colour distributions). The isolation literature provides a clear recommendation here: **if the new task comes from the same domain, mid layers should be reused rather than re-trained.**

- The ACL framework (Zhang et al., MICCAI 2023) explicitly tested adapter placement and found that inserting adapters *after* the pre-trained feature extractor (rather than within it) gave better results, because the pre-trained mid-layer features already capture domain-relevant patterns well. Modifying them introduces instability without improving discrimination.
- Partial Hypernetworks (2023) raised the key question: *"How many layers can we freeze without losing generalizability?"* Their answer, consistent with our design: for within-domain tasks (new tomato disease classes are still tomato diseases), freezing up to mid-layers is safe and preferred.
- In Cat B branch isolation: the new branch builds its own shallow feature extractor from scratch, but the old model's G2 features remain frozen. The new branch does not inherit G2 — it learns its own mid-level representation for T6, T7. This slight redundancy is acceptable because the branch is lightweight (~few MB) and zero forgetting for T1–T5 is guaranteed.

#### Group 3 (Late — features[9–12]) — The Critical Isolation Boundary

Late layers are the most task-specific part of the backbone and therefore the **primary target of parameter isolation**. The core principle from PackNet (Mallya & Lazebnik, 2018) is that within an overparameterised network, there is sufficient capacity to isolate task-specific subnetworks without degrading old task performance. For MobileNetV3-Small, G3 (features[9–12]) is where this isolation is most valuable:

- Forgetting of disease-discriminative features (lesion morphology, discolouration profiles) happens most severely in G3 during naive fine-tuning. Mirzadeh et al. (2022a) showed that *"the choice of architecture significantly impacts forgetting"* — in depthwise separable networks like MobileNetV3, late layers concentrate task-specific information more densely than in standard CNNs, making them more vulnerable and more worth protecting.
- In Cat A adapter isolation: the adapter is inserted **before** features[12] (the last 1×1 conv), at the output of features[11]. This placement is deliberate — features[12] is the final semantic compression step (96→576 channels), and the adapter at this boundary can modulate the feature representation just before this compression without touching the task-specific weights in features[9–11]. The old G3 weights are hard-frozen; the adapter is the only trainable element.
- The identity initialisation of the adapter's output projection (zeros) ensures G3 outputs are not disturbed at Cycle 1 start: the adapter gradually learns to add residual corrections to the G3 feature map rather than replacing it entirely.

#### Group 4 (Classifier Head) — Frozen in Isolation, Protected by Architecture

In isolation methods, the classifier head is fully frozen after Part 1 training. This is the strictest form of protection and is unique to isolation — EWC and Replay both update the head continuously.

- For Cat A: the old 5-class head remains hard-frozen. The adapter adds representational capacity in G3, allowing the fixed head to still make correct decisions on the new data stream without its weights changing. This works because Category A does not introduce new classes — the old head's decision boundaries are still valid; only the input feature distribution shifts slightly with new stream data.
- For Cat B: the old head is frozen on T1–T5, and a completely separate branch head handles T6–T7. At inference, scores from both are concatenated and argmax is taken. Rusu et al. (2016) showed this lateral/columnar architecture gives zero forgetting by construction — old class weights are never touched.

### 5.2 What "Frozen" means here vs other methods

| Method | G1–G4 frozen means... |
|---|---|
| EWC | Frozen via `requires_grad=False` AND Fisher penalty on unfrozen layers |
| Replay | Same layer freezing, forgetting prevented by data |
| **Isolation** | **Hard freeze — `requires_grad=False` on ALL old params, no exceptions** |

### 5.3 Category A — Same Classes (Adapter Expansion)

The recommended adapter design for MobileNetV3-Small in a same-class CL scenario is a **lightweight bottleneck adapter** inserted after the last InvertedResidual block (features[11]), before the last conv (features[12]).

```
Original:  features[11] → features[12] → avgpool → classifier
Cycle 1:   features[11] → [Adapter-C1] → features[12] → avgpool → classifier
Cycle 2:   features[11] → [Adapter-C1] → [Adapter-C2] → features[12] → avgpool → classifier
```

**Why this insertion point?**
- features[9–11] contain the highest-level semantic disease features. Inserting the adapter after features[11] (but before the 96→576 compression in features[12]) allows the adapter to modulate the rich 96-channel semantic representation without touching the compression that produces the final 576-dim feature vector.
- This is directly analogous to the ACL framework's finding (Zhang et al., MICCAI 2023) that adapters placed at the boundary between semantic extraction and final representation compression give the best balance of plasticity and stability.

**Adapter architecture (lightweight bottleneck):**
```python
class CycleAdapter(nn.Module):
    def __init__(self, channels=96, bottleneck=24):
        super().__init__()
        self.down  = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(bottleneck)
        self.act   = nn.Hardswish()
        self.up    = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        # Init: identity mapping — start with zero residual effect
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.bn2(self.up(self.act(self.bn1(self.down(x)))))
```

**Why identity init matters:** Initialising `up` to zeros means the adapter starts as a pass-through. The model's behaviour at Cycle 1 start is identical to the original model — the adapter learns incrementally from there. This prevents the adapter from immediately disrupting old-class performance.

| Group | Status | LR |
|---|---|---|
| G1 Early (old) | 🔒 Hard frozen | 0 |
| G2 Mid (old) | 🔒 Hard frozen | 0 |
| G3 Late (old) | 🔒 Hard frozen | 0 |
| G4 Head (old) | 🔒 Hard frozen | 0 |
| Adapter-C1 params | ✅ Training | Cycle 1: 1e-3 |
| Adapter-C2 params | ✅ Training | Cycle 2: 5e-4 |

```python
# Cycle 1 — insert adapter, freeze everything else
adapter_c1 = CycleAdapter(channels=96, bottleneck=24)
model.adapter_c1 = adapter_c1

# Freeze all original parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the new adapter
for param in model.adapter_c1.parameters():
    param.requires_grad = True

optimizer = Adam(model.adapter_c1.parameters(), lr=1e-3)

# Modified forward pass
def forward_with_adapter(model, x):
    x = model.features[:12](x)      # up to features[11]
    x = model.adapter_c1(x)          # new adapter
    x = model.features[12](x)        # last conv
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return model.classifier(x)
```

### 5.4 Category B — New Classes (Branch Expansion)

For new class introduction, a separate lightweight branch is added — not an in-line adapter. The old model is fully frozen and handles T1–T5. The new branch handles T6–T7 only.

```
OLD MODEL (frozen):  Input → features[0-12] → avgpool → classifier[5 outputs] → T1-T5 scores
NEW BRANCH (trains): Input → lite_features   → avgpool → branch_head[2 outputs] → T6-T7 scores

INFERENCE: combine → argmax over 7 scores
```

**Layer effect rationale for the branch architecture:**

The new branch uses a shallow feature extractor (3 InvertedResidual blocks) rather than copying the full G1–G3 backbone. This design choice is supported by:

- Rusu et al. (2016) showed that in Progressive Neural Networks, new columns (branches) do not need to replicate the full architecture of old columns — lateral connections allow new columns to build on old representations selectively. In our case, the new branch does not receive lateral connections (to keep inference simple), so it learns from scratch. Three blocks are sufficient for binary discrimination (T6 vs T7 vs background) — the same reasoning that justifies Case 2 Phase i (classifier-only fine-tuning) also supports a shallow branch.
- Mallya & Lazebnik (2018) found that task-specific subnetworks in PackNet occupy a small fraction of the total parameters. A 3-block branch with a 128-dim head occupies approximately 15–20% of MobileNetV3-Small's total parameters — acceptable for edge deployment while providing sufficient capacity for 2 new classes.

**New branch architecture — lightweight to control model growth:**
```python
class NewClassBranch(nn.Module):
    def __init__(self, n_new_classes=2):
        super().__init__()
        # Shallow feature extractor — 3 InvertedResidual blocks only
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            InvertedResidual(16, 32, stride=2, expand_ratio=4),
            InvertedResidual(32, 48, stride=2, expand_ratio=4),
            InvertedResidual(48, 96, stride=2, expand_ratio=4),
            nn.Conv2d(96, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, n_new_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)
```

**Inference routing (Cat B):**
```python
def inference_catB(old_model, new_branch, x):
    with torch.no_grad():
        old_logits = old_model(x)          # shape [B, 5] — T1-T5
        new_logits = new_branch(x)         # shape [B, 2] — T6-T7
        combined   = torch.cat([old_logits, new_logits], dim=1)  # [B, 7]
        return combined.argmax(dim=1)
```

| Component | Status | Cycle 1 LR | Cycle 2 LR |
|---|---|---|---|
| Old model G1–G4 | 🔒 Hard frozen | 0 | 0 |
| New branch features | ✅ Training | 1e-3 | 5e-4 |
| New branch head | ✅ Training | 1e-3 | 5e-4 |

**Model size tracking — important for deployment section:**
```python
def model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(param_bytes / (1024 ** 2), 2)

# Log at each cycle
print(f"Old model: {model_size_mb(old_model)} MB")
print(f"New branch: {model_size_mb(new_branch)} MB")
print(f"Combined: {model_size_mb(old_model) + model_size_mb(new_branch)} MB")
```

### 5.5 Key Limitation of Isolation: No Backward Transfer

While isolation guarantees zero forgetting, it sacrifices backward transfer — the ability of new training data to improve old-task performance. When old model weights are hard-frozen, there is no mechanism for the model to discover that T6 and T7 share features with T1–T5 (e.g., the yellowing pattern of Leaf Mold shares some visual elements with Septoria). EWC and Replay allow partial backward transfer; isolation does not.

This is an expected tradeoff and should be reported in the paper's Category B results: isolation will likely show the best forward transfer (zero forgetting of T1–T5) but the worst backward transfer (no benefit from T6, T7 data to old class performance).

---

## 6. EWC — Fisher Matrix and Layer Groups

EWC applies its penalty across all trainable parameters. Since in CL cycles Group 1+2 are frozen, the Fisher matrix is effectively computed only on Group 3+4 parameters — which is exactly where the important task knowledge lives. This is an efficient alignment between the layer freezing strategy and EWC:

- **Frozen layers:** No EWC penalty needed (gradients are zero anyway)
- **Group 3 (LR=1e-4):** Fisher penalty applied — protects the most important disease features
- **Group 4 classifier (LR=5e-4):** Fisher penalty applied — protects old class output weights

```python
def compute_fisher(model, replay_buffer_loader):
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # only non-frozen layers
            fisher[name] = torch.zeros_like(param)
    
    model.eval()
    for inputs, labels in replay_buffer_loader:
        model.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2)
    
    for name in fisher:
        fisher[name] /= len(replay_buffer_loader)
    return fisher
```

---

## 7. Learning Rate Summary Table — All Cases

| Stage | G1 Early | G2 Mid | G3 Late | G4 Head | Notes |
|---|---|---|---|---|---|
| Case 1 (scratch) | 1e-3 | 1e-3 | 1e-3 | 1e-3 | Uniform, all layers |
| Case 2 Phase i | frozen | frozen | frozen | 1e-3 | Classifier only |
| Case 2 Phase ii | frozen | frozen | 1e-4 | 1e-3 | Unfreeze G3 |
| Case 3A (pretrain) | 1e-3 | 1e-3 | 1e-3 | 1e-3 | Uniform, 26-class head |
| Case 3B Phase i | frozen | frozen | 5e-5 | 5e-4 | Low LR — less domain shift |
| Case 3B Phase ii | frozen | 1e-5 | 5e-5 | 5e-4 | Gentle G2 unfreeze |
| CL Cat A Cycle 1 | frozen | frozen | 1e-4 | 5e-4 | G3+head only |
| CL Cat A Cycle 2 | frozen | frozen | 5e-5 | 2e-4 | Progressive freeze ↓ |
| CL Cat B Cycle 1 | frozen | 5e-5 | 2e-4 | old:1e-4 / new:1e-3 | Split-LR classifier |
| CL Cat B Cycle 2 | frozen | 1e-5 | 1e-4 | old:5e-5 / new:5e-4 | Progressive freeze ↓ |
| Isolation Cat A (any) | frozen | frozen | frozen | frozen | Only adapter LR: 1e-3 / 5e-4 |
| Isolation Cat B (any) | frozen | frozen | frozen | frozen | Only new branch LR: 1e-3 / 5e-4 |

---

## 8. Comparative Layer-Effect Summary — All Three CL Methods

This table gives a consolidated view of how each layer group behaves under each CL method. Use this as a quick reference when analysing results or writing the paper's method section.

| Layer Group | EWC | Experience Replay | Parameter Isolation |
|---|---|---|---|
| **G1 Early** | Hard frozen. Fisher not computed. | Hard frozen. Replay has no effect here. | Hard frozen (old model). Not replicated in adapter/branch. |
| **G2 Mid** | Hard frozen in Cat A. Very slowly unfrozen in Cat B (5e-5). | Same as EWC — Cat B low LR; replay stabilises distribution gradient. | Hard frozen (old model). New branch learns its own G2 from scratch. |
| **G3 Late** | Primary Fisher target. LR 1e-4 → 5e-5. Penalty resists weight change. | Primary replay target. Continues to drift slightly even with replay (Verwimp et al., 2020). LR kept same as EWC. | Hard frozen (old model). Adapter inserted at G3 boundary provides new capacity. Adapter LR: 1e-3 → 5e-4. |
| **G4 Head** | Fisher protects old class neurons. LR 5e-4 → 2e-4. | Old class neurons directly updated by replayed old-class samples. Strongest method for head stability (Buzzega et al., 2020). | Fully frozen (Cat A). Zero forgetting by construction. Cat B: separate 2-neuron branch head trained at 1e-3. |
| **Forgetting guarantee** | Probabilistic — Fisher penalty reduces forgetting proportional to λ and buffer quality. | Probabilistic — forgetting reduced proportional to buffer size and class coverage. | Mathematical — old weights never change → zero forgetting for old classes. |
| **Plasticity** | Good — G3 and head are free to update. | Good — mixed batches preserve plasticity. | Limited in Cat A (adapter is the only plastic element). None for old classes in Cat B. |
| **Backward transfer** | Partial — old class val can slightly improve. | Partial — same as EWC. | None — hard freeze prevents any backward transfer. |

---

## 9. Why Reduce LR Per Cycle — Research Justification

This strategy is supported by three independent lines of research:

**SLCA (Zhang et al., 2023):** Found that using a "slow learner" (very low LR) for the representation layer while allowing the classifier to adapt at normal LR almost resolves progressive overfitting in continual learning. Reducing Group 3 LR from 1e-4 (Cycle 1) to 5e-5 (Cycle 2) directly implements this principle.

**AutoFreeze (Liu et al., 2021):** Found that early layers in DNNs converge faster and can be frozen earlier in training without accuracy loss. This justifies keeping Group 1 (early) permanently frozen from the very first CL cycle.

**Partial is Better than All (Zhao et al., 2021):** Found that fine-tuning a carefully chosen subset of layers outperforms both full fine-tuning and classifier-only fine-tuning for small datasets. This justifies the asymmetric G3-only update in CL cycles rather than updating the full backbone.

**Wide Neural Networks Forget Less (Mirzadeh et al., 2022b):** Demonstrated that wider networks are inherently more robust to forgetting due to gradient orthogonality and lazy training regime effects. MobileNetV3-Small's depthwise-separable architecture concentrates semantic information in late layers, making the layer-wise LR asymmetry even more important than in wider standard CNNs.

Combined with EWC's weight-level penalty and Experience Replay's data-level protection, the layer-wise LR strategy provides a third, complementary level of forgetting protection — making the full system significantly more robust than any single method alone.

---

## 10. Implementation Code Reference

```python
# === Utility: group parameter sets by layer group ===
def get_param_groups(model):
    return {
        'g1': list(model.features[:4].parameters()),
        'g2': list(model.features[4:9].parameters()),
        'g3': list(model.features[9:].parameters()),
        'g4': list(model.classifier.parameters())
    }

# === Freeze a group entirely ===
def freeze_group(params):
    for p in params:
        p.requires_grad = False

# === Unfreeze a group ===
def unfreeze_group(params):
    for p in params:
        p.requires_grad = True

# === CL Category A Cycle 1 optimizer ===
def get_cl_catA_cycle1_optimizer(model):
    pg = get_param_groups(model)
    freeze_group(pg['g1'])
    freeze_group(pg['g2'])
    unfreeze_group(pg['g3'])
    unfreeze_group(pg['g4'])
    return Adam([
        {'params': pg['g3'], 'lr': 1e-4},
        {'params': pg['g4'], 'lr': 5e-4}
    ])

# === CL Category A Cycle 2 optimizer (progressive freeze) ===
def get_cl_catA_cycle2_optimizer(model):
    pg = get_param_groups(model)
    freeze_group(pg['g1'])
    freeze_group(pg['g2'])
    return Adam([
        {'params': pg['g3'], 'lr': 5e-5},  # halved vs Cycle 1
        {'params': pg['g4'], 'lr': 2e-4}   # halved vs Cycle 1
    ])

# === CL Category B Cycle 1 — split-LR classifier ===
def get_cl_catB_cycle1_optimizer(model, n_old=5):
    pg = get_param_groups(model)
    freeze_group(pg['g1'])
    unfreeze_group(pg['g2'])
    unfreeze_group(pg['g3'])
    # Split classifier weights by old vs new class
    old_params = [model.classifier[-1].weight[:n_old],
                  model.classifier[-1].bias[:n_old]]
    new_params = [model.classifier[-1].weight[n_old:],
                  model.classifier[-1].bias[n_old:]]
    other_cls   = [p for n,p in model.classifier[:-1].named_parameters()]
    return Adam([
        {'params': pg['g2'],    'lr': 5e-5},
        {'params': pg['g3'],    'lr': 2e-4},
        {'params': other_cls,   'lr': 5e-4},
        {'params': old_params,  'lr': 1e-4},  # protect old classes
        {'params': new_params,  'lr': 1e-3},  # fast learning for new classes
    ])
```

---

## 11. Key Rules — Never Violate These

1. **Group 1 (features[0–3]) is always frozen in CL cycles** — no exception. These universal features should never be updated with a 5-class dataset during CL.

2. **Progressive LR reduction per cycle** — never increase LR from one cycle to the next. The model is accumulating knowledge; each cycle should require smaller updates.

3. **Split-LR classifier is mandatory for Category B** — using a uniform LR on the expanded classifier head will cause old class neurons to be disrupted by the fast-learning new class gradient signal.

4. **Early stopping always monitors old-class val F1** — never use new-data training loss as the stopping criterion in CL. A dropping training loss with rising forgetting is the classic CL failure mode.

5. **EWC Fisher is computed on frozen-group-excluded parameters only** — computing Fisher on frozen layers wastes computation and produces meaningless penalty values.

6. **For Experience Replay: keep LRs identical to EWC** — do not vary LRs across methods. The method comparison must be clean; differences belong in the results, not the hyperparameters.

7. **For Isolation: always use identity-init adapters** — non-identity initialisation will immediately degrade old-class performance at the start of Cycle 1 before the adapter has learned anything useful.

8. **Log model size at every cycle for isolation methods** — the deployment section of the paper requires size vs accuracy data. Track combined model size (old model + adapter/branch) after each cycle.

---

## 12. Full Reference List

| # | Citation | Used For |
|---|---|---|
| 1 | Howard et al. (2019). Searching for MobileNetV3. ICCV 2019. | Architecture reference — layer group definitions |
| 2 | Zhang et al. (2023). SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model. ICCV 2023. | CL LR reduction strategy — progressive freeze justification |
| 3 | Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS. | EWC method foundation |
| 4 | Rusu et al. (2016). Progressive Neural Networks. arXiv:1606.04671. | Parameter isolation — branch/column architecture justification |
| 5 | Mallya & Lazebnik (2018). PackNet: Adding Multiple Tasks to a Single Network. CVPR 2018. | Parameter isolation — layer-level pruning and capacity analysis |
| 6 | Rolnick et al. (2019). Experience Replay for Continual Learning. NeurIPS 2019. | Replay method foundation — plasticity and stability tradeoff |
| 7 | Buzzega et al. (2020). Rethinking Experience Replay: A Bag of Tricks for CL. ICPR 2020. | Replay superiority for class-incremental; classifier head stability |
| 8 | Verwimp et al. (2020). The Effectiveness of Memory Replay in Large Scale CL. arXiv:2010.02418. | Intermediate layer drift under replay; CAR framework |
| 9 | Shin et al. (2017). Continual Learning with Deep Generative Replay. NeurIPS 2017. | Classifier head as primary forgetting site; LR effect |
| 10 | Liu et al. (2021). AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning. arXiv:2102.01386. | Early layer freezing justification |
| 11 | Zhao et al. (2021). Partial is Better Than All: Revisiting Fine-tuning Strategy for Few-Shot Learning. AAAI 2021. | Subset layer fine-tuning superiority |
| 12 | Mirzadeh et al. (2022a). Architecture Matters in Continual Learning. arXiv:2202.00275. | Architecture forgetting rate; late layer concentration in MobileNet |
| 13 | Mirzadeh et al. (2022b). Wide Neural Networks Forget Less Catastrophically. ICML 2022. | Width and forgetting; gradient orthogonality |
| 14 | Zhang et al. (2023). ACL: Adapter-based Continual Learning for Disease CL. MICCAI 2023. | Adapter insertion point; frozen feature extractor design |
| 15 | Yoo et al. (2024). Layerwise Proximal Replay: A Proximal Point Method for Online CL. ICML 2024. | Replay trajectory instability in intermediate layers; LPR justification for low mid-layer LR |
| 16 | Serrà et al. (2018). HAT: Overcoming Catastrophic Forgetting with Hard Attention to the Task. ICML 2018. | Parameter isolation via hard attention; task-specific mask approach |

---

*This document covers the layer-wise training strategy for all Cases and CL Cycles. Cross-reference with `CL_Implementation_Flow.md` for full input/output specifications and `DATA_README.md` for data splits.*
