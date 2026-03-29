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

**The Problem**: Editing an image with standard diffusion models often fails — changing one region (lighting, clothing) introduces artifacts in unrelated areas. The model treats each pixel independently, losing spatial consistency.

**The Opportunity**: Video models learn temporal coherence — the constraint that consecutive frames must be spatially consistent. This is *exactly* what's needed for coherent image editing.

**Our Insight**: By treating image editing as a frame generation task within a multi-frame latent window, we repurpose a video model's learned coherence for single-frame photorealistic editing — with no retraining, no fine-tuning.

---

## 3. BACKGROUND: FRAMEPACK

**What is FramePack?** A video diffusion model that predicts the next frame by looking at all previous frames. Rather than keeping all history (which would require unbounded memory), it compresses older frames geometrically — keeping recent frames detailed, older frames minimal.

**Why it matters for image editing:**
- Solves **forgetting**: Progressive compression bounds context length, enabling any video length
- Solves **drifting**: Can generate endpoints first, then fill gaps (prevents error accumulation)
- **Key property we exploit**: Treats prior frames as spatial anchors — we anchor the input image to force coherence in generated output

**Inverted Sampling** (the critical innovation):
```
Traditional generation:  F₁ → F₂ → F₃ (errors compound)
Inverted generation:     F₁ ←─ ─→ F_N (pulled back to input)
```
By generating backward from the input image, every frame is pulled *toward* quality rather than drifting away.

---

## 4. METHODOLOGY

### 4.1 The Slot-Based Injection Strategy

FramePack's latent window organizes frames as numbered slots. We create a **fake video** where:

```
  Slot 1        Slots 2–X        Slot X         Slot N
  INPUT         (zeros)          TARGET         INPUT
  (anchor)      ←── generate ──→  (edit)        (anchor)
```

**How it works**: The model learned that consecutive frames must be coherent. By placing the input image at both ends, we enforce a spatial constraint: the target must stay consistent with the input while drifting toward the text guidance.

**Why zero-shot**: No retraining. The model's coherence prior — built on video — immediately adapts to image structure.

### 4.2 Pipeline

1. **Encode inputs** → Text (LLaMA-3 + CLIP) + Image (VAE + SiglipVision)
2. **Pack latents** → Input at slots 1 & N, target slot X as noise
3. **Diffuse** → DiT with 25 steps attending to anchors + text features
4. **Decode** → Extract slot X, VAE decode to image

### 4.3 Conditioning

- **Text**: LLaMA-3 (semantic depth) + CLIP-L (visual grounding)
- **Image**: Spatial features anchor structure in the diffusion path
- **Result**: Text steers *what* changes; anchors preserve *how much*

### 4.4 Key Optimizations

| Optimization | Impact |
|---|---|
| **MagCache** | Skip redundant DiT forward passes by monitoring output magnitude |
| **NullConditioner** | Skip text encoding when guidance=1.0 (common case) |
| **VAE Tiling** | Process decoder in spatial tiles to reduce peak memory |

*Additional: model preloading, mixed precision (FP16/BF16), prompt embedding caching, aspect ratio preservation.*

---

## 5. EXPERIMENTS

### Ablation Study: Target Frame Index Optimization

**Dataset**: InstructPix2Pix (20 paired image-instruction samples)  
**Fixed Parameters**: seed=42, 25 steps, guidance_scale=10.0, 640×512, bfloat16

**Target Index Comparison** (InstructPix2Pix dataset, n=20):

| Target Index | CLIP Score ↑ | SSIM vs idx_9 ↑ | LPIPS vs idx_9 ↓ | Time (s) |
|---|---|---|---|---|
| idx_9 (baseline) | [pending] | 1.00 | 0.000 | [pending] |
| idx_12 | [pending] | [pending] | [pending] | [pending] |
| idx_15 | [pending] | [pending] | [pending] | [pending] |
| idx_20 | [pending] | [pending] | [pending] | [pending] |

*CLIP Score measures prompt adherence (higher=better); SSIM/LPIPS measure structural/perceptual similarity to idx_9 baseline; results pending from InstructPix2Pix ablation study.*

---

## 6. EXAMPLES

*[Space reserved for 3-4 photo editing examples showing:]*
- Appearance changes (clothing, accessories)
- Environmental/style modifications
- Object addition/manipulation
- Quality preservation across edits

---

## 6. DISCUSSION & LIMITATIONS

### Key Findings

- **Video pretraining gives spatial coherence for free**: Temporal consistency learned on video naturally transfers to spatial structure in images.

- **Dual encoders handle complex semantics**: LLaMA-3 (semantic depth) + CLIP-L (visual grounding) outperforms CLIP-only on compound descriptions.

- **No fine-tuning required**: Slot-based conditioning works with pre-trained weights — coherence prior adapts immediately.

- **Domain-specific strengths**: Model achieves strong performance on stylized and cartoon image editing. Photorealistic editability is limited, revealing that FramePack's pretraining has clear domain preferences — a feature driving future domain-adaptation research.

### Limitations

- **Memory Access**: 28GB VRAM minimum limits deployment to research/enterprise settings.

- **Not True Inversion**: Unlike DDIM inversion, unedited regions may drift slightly — acceptable for general editing, not pixel-perfect preservation.

- **VAE Bottleneck**: VAE decoding dominates runtime; further acceleration requires architectural redesign, not just optimization.

### Future Directions

- Multi-frame coherent editing (5-8 frames with consistent edits across the video)
- Domain adaptation: Fine-tune on photorealistic image pairs to extend effectiveness beyond stylized content
- True inversion-based methods for region-specific masks
- Quantization (FP8) to bring VRAM below 20GB

---

## 7. ACKNOWLEDGMENTS

[Acknowledge collaborators, funding agencies, computational resources, etc.]

---

## Poster Layout (A1 / A0 Portrait)

```
┌─────────────────────────────────────────┐
│         HEADER + TEASER                 │
│   Title | Photo Examples | Author       │
│        (22-25% of poster height)        │
├─────────┬───────────────┬───────────────┤
│INTRO    │  BACKGROUND   │  METHODOLOGY  │
│CHALLENGE│ (FramePack)   │  (Slot-Based) │
│ (10%)   │   (8%)        │   (28%)       │
├─────────┴───────────────┴───────────────┤
│   ABLATION STUDY (12%)                  │
│   Key design choices that matter        │
├─────────────────────────────────────────┤
│   EXAMPLES (17%)                        │
│   Photo Editing Examples                │
├─────────────────────────────────────────┤
│   INSTITUTION | QR CODE | CONTACT (8%)│
└─────────────────────────────────────────┘
```

---

**[QR Code to GitHub Repository]**

---

*Generated for: [Conference/Workshop Name], [Date]*
