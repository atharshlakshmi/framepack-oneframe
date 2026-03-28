# FramePack One-Frame: Adapting Video Diffusion for Photorealistic Generation

**Author:** [Your Name]  
**Institution:** [Your Institution]  
**Advisor/Lab:** [Advisor/Lab Name]

---

## 1. TEASER (Header Visual)

*[Visual arrangement: 3-4 editing examples]*

| Input Image | Text Prompt | Output Image |
|---|---|---|
| [Portrait of person] | "the person is wearing a red hat" | [Same portrait, now with red hat] |
| [Mountain landscape] | "rainy weather with storm clouds" | [Same landscape under storm] |
| [Cat photo] | "the cat is wearing sunglasses" | [Cat now wearing sunglasses] |

---

## 2. INTRODUCTION / MOTIVATION

### The Challenge
Video diffusion models are powerful — but they generate sequences. Can we adapt them for **high-quality single-image photo generation** from text descriptions?

### Why This Approach?
- Video models learn **temporal coherence** across frames — this penalizes sudden spatial jumps or inconsistencies
- This same coherence mechanism can enforce **spatial consistency** in a single edited image
- FramePack's architecture is uniquely suited: it processes frames through a learned latent window, allowing us to repurpose it for single-frame generation

### Our Key Innovation
**Video-to-Photo Adaptation**: Rather than design a new architecture, we leverage FramePack's existing learned coherence patterns to guide text-to-image generation. By conditioning the model properly, temporal coherence translates into spatial coherence for photo-realistic outputs.

### Contributions
1. **Methodological insight**: Video diffusion models' learned coherence can be directly applied to image generation
2. **Practical photorealistic pipeline**: End-to-end system for high-quality photo editing from text prompts
3. **Dual-encoder approach**: LLaMA-3 + CLIP-L for semantically rich scene understanding
4. **Generalization**: Demonstrates that video and image generation share deeper representational principles

---

## 3. BACKGROUND: FRAMEPACK

### 3.1 The Problem: Forgetting vs. Drifting

Video generation models face a fundamental tradeoff:

- **Forgetting**: Fading memory — model struggles to remember early frames and maintain temporal consistency
- **Drifting**: Quality degradation from error accumulation — small errors propagate and compound across long sequences

Stronger memory mechanisms reduce initial errors but also memorize and propagate those errors when they occur. Naive solutions (encoding more context) hit quadratic transformer complexity limits. FramePack's innovation: break this tradeoff.

### 3.2 How FramePack Works

**Progressive Frame Compression**:
- Sorts frames by importance (time proximity or feature similarity)
- Assigns variable compression depths: recent frames stay detailed, older frames compress
- Each frame $F_i$ gets context length: $\phi(F_i) = L_f / \lambda^i$ where $\lambda > 1$ (typically 2)
- Total context converges to fixed upper bound: $L_{\max} = (S + \frac{\lambda}{\lambda-1}) \cdot L_f$ as $T \to \infty$
- **Result**: Process thousands of frames with fixed, bounded memory

**Anti-Drifting Methods**:
1. **Planned Endpoints**: Generate start and end frames together, then fill gaps (bidirectional, breaks causal chain)
2. **History Discretization**: Compress frame history into discrete tokens via K-means codebook (reduces train-test mismatch)
3. **Multi-scale Packing**: Sort by feature similarity for specialized tasks (world models, structured generation)

**Dual-Encoder Conditioning**:
- LLaMA-3 (32 layers, 4096-dim): Semantic understanding
- CLIP-L (768-dim): Visual-semantic alignment
- SiglipVision (1152-dim): Spatial feature extraction
- Combined: 4864-dim conditioning signal enables rich semantic control

### 3.3 Why FramePack Works for Photo Generation

FramePack's learned temporal coherence patterns translate to spatial coherence in single-frame generation:

- **Temporal → Spatial**: Video models learn that consecutive frames must be spatially consistent. This constraint becomes a structural prior for maintaining coherence in a single edited image.
- **Packed Window Design**: By conditioning on a single input image repeated across the latent window, we leverage FramePack's built-in coherence enforcement. The model naturally outputs edits that remain consistent with the input.
- **Bounded Context**: The fixed context length enables efficient processing without architectural changes — we simply reuse FramePack's pre-trained weights.
- **Top-Down Control**: Dual encoders provide semantic direction (what to change) while coherence mechanisms provide structural preservation (what to keep).
---

## 4. METHODOLOGY

### 4.1 Overall Pipeline

```
Input Image + Text Prompt
        ↓
┌───────┴──────────┬──────────────────┐
↓                  ↓                   ↓
[Input Image]  [Text Prompt]      [Control Signal]
                   ↓                   ↓
        ┌──────────┴─────────────┐    │
        ↓                        ↓    ↓
    [LLaMA-3]               [CLIP-L] [SiglipVision]
    (32 layers)           (768-dim)   (1152-dim)
    4096-dim               Pooled      Features
        ↓                  ↓           ↓
        └──────────┬──────┴──────────┘
                   ↓
        [LatentIndexManager]
        Pack multi-scale controls
        (1×, 2×, 4× resolution hierarchy)
                   ↓
    [DiT + MagCacheWrapper]
    24 attention heads
    UniPC sampler, 25 diffusion steps
                   ↓
            [VAE Decode]
            (with optional tiling)
                   ↓
            Output Image
```

### 4.2 Key Design Decisions

#### Decision 1: Multi-Scale Latent Packing
- Control indices sampled at **1×, 2×, and 4× resolutions** of the base spatial grid
- Enables progressive refinement across spatial scales during denoising
- Improves coherence without quadratic memory cost

#### Decision 2: Dual-Encoder Text Conditioning
- **LLaMA-3** handles complex semantic descriptions (compound phrases, negations, style descriptors)
- **CLIP-L** provides visual-semantic grounding, ensuring generated content aligns with natural image statistics
- Combined: 4864-dim total text conditioning signal vs. ~768-dim for CLIP-only models

#### Decision 3: Frame Index 9 Selection
- Positioned at **window boundary** to maximize input access
- Provides natural "next frame" interpretation for model familiar with sequential prediction
- Allows 8 prior context frames for denoising

---

### 4.3 Adaptation: Video Model → Photo Generation

**Core Principle**: Reposition the FramePack model to generate photorealistic outputs by:

1. **Conditioning Strategy**: Provide the input image and text prompt to the model's learned latent space
2. **Coherence Leverage**: The model's pre-trained temporal coherence (learned from video) constrains output to remain spatially consistent with the input
3. **Text Guidance**: Semantic understanding (via dual encoders) directs *what* to generate while coherence directs *how* (preserving spatial structure)

**Result**: The model simultaneously respects the input's spatial layout and the prompt's semantic intent — yielding coherent, semantically-accurate photo edits.

---

### 4.4 Core Conditioning Pipeline

**Two parallel conditioning streams:**

1. **Semantic Stream**: Text → [LLaMA-3 (32 layers, 4096-dim) + CLIP-L pooler (768-dim)]
   - LLaMA-3 extracts deep semantic understanding
   - CLIP-L ensures visual alignment
   - Combined: 4864-dim conditioning signal

2. **Visual Stream**: Image → [VAE encode] → [SiglipVision features (1152-dim, 577 tokens)]
   - Extracts rich spatial features at 384×384 resolution
   - Preserves input composition and structure

**Fusion**: Both streams feed into DiT's cross-attention layers, enabling:
- Text-guided editing (appearance, style, objects)
- Spatial coherence anchoring (via visual stream)
- Natural balancing (semantics vs. structure preservation)

---

### 4.5 Engineering Optimizations

To enable practical deployment, the pipeline incorporates:
- Memory efficiency (VAE tiling, mixed precision)
- Consistent output quality (aspect ratio handling, high-quality resizing)
- Reproducible inference (configuration management)

These support the core photorealistic generation capability.

---

## 5. ACKNOWLEDGMENTS

[Acknowledge collaborators, funding agencies, computational resources, etc.]

---

## Poster Layout (A1 / A0 Portrait)

```
┌─────────────────────────────────────────┐
│         HEADER + TEASER                 │
│   Title | Photo Examples | Author       │
│        (25-30% of poster height)        │
├─────────────────┬───────────────────────┤
│   INTRODUCTION  │   BACKGROUND          │
│   & CHALLENGE   │  (FramePack Base)     │
│    (15%)        │      (12%)            │
├─────────────────┴───────────────────────┤
│                                         │
│   METHODOLOGY: VIDEO→PHOTO (50%)        │
│  • Why Video Coherence Works for        │
│    Photo Generation                     │
│  • Adaptation Strategy                  │
│  • Dual-Encoder Conditioning            │
│  • Pipeline Architecture                │
│                                         │
├─────────────────────────────────────────┤
│   INSTITUTION | QR CODE | CONTACT      │
│              (8%)                       │
└─────────────────────────────────────────┘
```

---

**[QR Code to GitHub Repository]**

---

*Generated for: [Conference/Workshop Name], [Date]*
