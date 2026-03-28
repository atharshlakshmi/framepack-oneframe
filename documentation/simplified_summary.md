# FramePack One-Frame Inference: A Beginner's Guide

## What Problem Are We Solving?

You have a photo and want to edit it using words: "Give the cat a hat" or "Change background to a beach". Instead of using Photoshop, you use **AI by typing a text description**.

## The Core Idea: Using a Video Model for Image Editing

### Video vs Image Models

**Normal image model:** Takes your image, tries to edit it directly.

**Video model (FramePack):** Understands sequences of frames and how they relate smoothly. It knows:
- Objects stay in the same place
- Colors transition smoothly
- Lighting is consistent across frames

**The trick:** We repurpose this video knowledge for image editing!

```
Video Generation:
Frame 1 → Frame 2 → Frame 3 → ... → Frame 9

One-Image Editing (Our Approach):
Your Image → (9-frame window) → Edited Image
(Position 1)                   (Position 9)
```

We ask: *"If this is frame 1, and the text describes what should happen, what should frame 9 look like?"*

The model naturally generates consistent edits because video models are trained to maintain coherence!

## How FramePack Adapts for Image Editing

### Step 1: Understanding Your Text

Your text description is processed by **two text encoders working together**:

1. **LLaMA-3** (semantic understanding): Understands meaning and context
   - Output: 4,096-dimensional representation
   
2. **CLIP-L** (visual understanding): Connects words to visual concepts
   - Output: 768-dimensional representation

Together they create a deep understanding of what you want: "girl with red hat" = person + headwear + color.

### Step 2: Encoding Your Image

Your image gets compressed into a **latent representation**:

```
Original: 640 × 512 pixels, 3 colors
    ↓
Compress: Using a VAE (neural network compressor)
    ↓
Latent: 80 × 64 (1/8 size), 16 channels
(Essence of the image, stripped of unnecessary details)
```

This compressed form makes processing faster and uses less memory.

### Step 3: The 9-Frame Window Setup

FramePack works with **9-frame sequences**. We arrange it like this:

```
Frame Index:  1      2      3      4      5      6      7      8      9
             [REF] [REF] [?]    [?]    [?]    [?]    [?]    [?]   [TARGET]
              ↑                                                      ↑
         Your image                                          Where the edit happens
         (reference)
```

- **Positions 1 & 10**: Your original image (tells the model "keep this subject")
- **Position 9**: Where the edited image is generated (the target)
- **2× & 4× scales**: Multi-resolution guidance for consistency

This arrangement gives the model context: "I know the subject, now make it match the text at position 9."

### Step 4: The Diffusion Loop (25 Iterations)

This is where the edit actually happens using **diffusion**: starting from noise and gradually refining.

```
Start (Step 0):     [Complete random noise 🎲]
                         ↓
Mid (Step 12-13):   [Recognizable subject 🐱]
                         ↓
End (Step 25):      [Perfect edited result 🎩🐱]
```

At each step, the AI thinks:
- "I have the original image (position 1)"
- "I have the text description: make this edit"
- "I should generate position 9 to be similar to position 1 BUT matching the text"

The multi-scale controls (2× and 4×) guide the model at different detail levels:
- **Coarse (4×):** "Where should the main shapes go?"
- **Medium (2×):** "What are the secondary features?"
- **Fine (1×):** "What are the precise details?"

### Step 5: Extract and Decode

Once generation is done:

```
Generated latent [1, 16, 1, 80, 64]
        ↓
Decode (expand back to full image)
        ↓
Final Image [640, 512, 3 RGB channels]
```

## Visual Summary: The Complete Journey

```
┌─────────────────────────────────────────────────────────────┐
│                   YOUR IMAGE                                 │
│                   "cat sitting"                              │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: IMAGE TO COMPACT FORM                                │
│ Compress image → [1, 16, 1, 80, 64] latent                  │
│ (Remove unnecessary details, keep essence)                   │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: UNDERSTAND YOUR TEXT                                 │
│ "Create a hat on the cat"                                    │
│ ↓                                                             │
│ Encoder 1 (LLaMA): 4,096-dimensional meaning vector         │
│ Encoder 2 (CLIP):  768-dimensional visual vector            │
│ = Deep understanding of your request                        │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: PREPARE CONTEXT                                      │
│ Reference: Your original cat image                          │
│ Position 9: Where edited image will be generated            │
│ Multi-scale: Coarse, medium, fine detail levels             │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: GENERATE WITH DIFFUSION (25 steps)                  │
│                                                              │
│ Iteration 0:  [Random noise 🎲]                             │
│ Iteration 5:  [Vague cat shapes 👻]                         │
│ Iteration 10: [Recognizable cat 🐱]                         │
│ Iteration 15: [Cat with rough hat 🎩🐱]                     │
│ Iteration 20: [Clear cat with hat 🎩🐱✓]                    │
│ Iteration 25: [Perfect final result 🎩🐱💯]                 │
│                                                              │
│ Magic: At each step, AI asks:                               │
│ "Does this match the reference AND the text? Yes/No → Adjust"
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: EXPAND BACK TO FULL IMAGE                            │
│ Take compressed result and expand it back                    │
│ (Reverse of Step 1)                                          │
│ Result: Full-size edited image                               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│              YOUR EDITED IMAGE                                │
│           "cat with a red hat 🎩🐱"                          │
└──────────────────────────────────────────────────────────────┘
```

## Why This Approach Works So Well

### 1. Video Model Knowledge Translates Perfectly

FramePack was trained on videos where adjacent frames must be coherent. When we ask "generate position 9 similar to position 1, but edited," the model naturally generates smooth, consistent changes—exactly what we want for image editing.

### 2. Multi-Reference Guidance

Using reference frames at positions 1 and 10:
- **Keep the subject recognizable** (from position 1)
- **Match the text description** (from the text encoders)
- **Maintain spatial coherence** (across the window)

Result: Edits that preserve the original subject while transforming it.

### 3. Multi-Scale Generation

The 2× and 4× scales work together:
- **Coarse (4×):** Where should main shapes go?
- **Medium (2×):** What are secondary features?
- **Fine (1×):** What are precise details?

## Full Pipeline Summary

```
┌──────────────────────────────────────────────────┐
│         Your Image: "cat sitting"                │
└─────────────────┬────────────────────────────────┘
                  ↓
        ┌─────────────────────┐
        │ Compress Image      │
        │ → Latent [80,64,16] │
        └────────┬────────────┘
                 ↓
    ┌────────────────────────────────┐
    │ Encode Text: "add a hat"       │
    │ = [4096] + [768] dimensions    │
    └──────┬───────────────────────┘
           ↓
    ┌──────────────────────────────┐
    │ Setup 9-Frame Window:        │
    │ Ref: Your image (pos 1&10)   │
    │ Target: Generate (pos 9)     │
    │ Multi-scale guidance (2×,4×) │
    └──────┬───────────────────────┘
           ↓
    ┌────────────────────────────────┐
    │ Diffusion Loop (25 steps):     │
    │ Start: Random noise            │
    │ Mid: Recognizable subject      │
    │ End: Image matching text       │
    │ (Optimized with MagCache)      │
    └──────┬────────────────────────┘
           ↓
    ┌──────────────────┐
    │ Decode Latent    │
    │ → Full Image     │
    └────────┬─────────┘
             ↓
    ┌──────────────────────┐
    │ Final Result:        │
    │ "cat with a hat 🎩🐱" │
    └──────────────────────┘
```

## Key Optimizations

- **MagCache:** Skip redundant diffusion steps (20-30% faster)
- **Text Caching:** Reuse encodings for repeated prompts (40× faster)
- **Model Choreography:** Move encoders off GPU when not needed (save ~20GB)
- **VAE Tiling:** Process large images in chunks (prevent out-of-memory)

## The Bottom Line

FramePack One-Frame Inference takes a **video generation model** (trained to make smooth, coherent sequences) and cleverly adapts it for **image editing** by:

1. Putting your image at the start of a 9-frame window
2. Asking the model to generate position 9 based on your text
3. Leveraging the model's natural tendency to maintain coherence

Result: **High-quality image edits using a pre-trained video model**, without rebuilding anything from scratch!
