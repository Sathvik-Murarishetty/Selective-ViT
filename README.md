# SelectiveViT: Improving Vision Transformer Interpretability via Sparse Attention

> **"Bridging the gap between performance and transparency in ViTs."**

---

## Overview

**SelectiveViT** is a novel modification to Vision Transformers that enhances **interpretability** without compromising **performance**. By applying a **threshold-based sparsification mechanism** on self-attention maps, the model suppresses low-impact token interactions—resulting in **focused, transparent** attention visualizations.

---

## Abstract

While ViTs achieve high accuracy, their dense attention maps hinder interpretability. SelectiveViT addresses this by:

* Eliminating low-importance token pairs after softmax.
* Re-normalizing the attention weights.
* Retaining only the most relevant visual cues.

Evaluated on **CIFAR-10**, SelectiveViT:

* Outperforms ViT-Base by **\~3.6%**.
* Achieves **\~70% attention sparsity**.
* Produces interpretable visualizations using **Grad-CAM** and **Captum**.

---

## Architecture Highlights

* Base Model: `ViT-Base`
* Attention Modification:

```python
SparseAttention(Q, K, V) = ReLU(softmax(QKᵀ / √dk) - τ) / sum(ReLU(...)) * V
```

* Threshold `τ = 0.01` applied post-softmax to sparsify attention.
* No additional parameters added to the model.
* Training performed using `PyTorch + Timm`.

---

## Experimental Setup

* **Dataset**: CIFAR-10 (60K images, 10 classes)
* **Resolution**: 224x224 (resized)
* **Training Epochs**: 70
* **Optimizer**: AdamW
* **Batch Size**: 16
* **Learning Rate**: 3e-4 (cosine decay)
* **Loss**: CrossEntropyLoss
* **Pretrained on**: ViT-Base (Timm)

---

## Results

| Epoch | ViT-Base Accuracy | SelectiveViT Accuracy |
| ----- | ----------------- | --------------------- |
| 10    | 80.65%            | 81.77%                |
| 30    | 83.01%            | 86.02%                |
| 50    | 84.62%            | 87.89%                |
| 70    | 85.47%            | **89.11%**            |

---

## Interpretability Tools

* **Grad-CAM**: Heatmaps showing where the model focused.
* **Captum (LayerGradCam)**: Fine-grained visual attribution.
* **Token Sparsity Tracker**: Measures avg. tokens per head.

### Visual Impact:

* ViT-Base: Diffused, noisy attention
* SelectiveViT: Clean, object-focused maps

---

## Key Findings

* Selective attention improves **both accuracy and transparency**
* Ideal threshold: `τ = 0.01` (balances performance and sparsity)
* ViT-Base attends to background noise; SelectiveViT filters it out

---

## Trade-Offs & Limitations

* +70% sparsity, better interpretability
* ⚠Too high a threshold (τ ≥ 0.10) harms performance
* Uniform threshold might miss nuanced layer-wise importance

---

## Future Work

* Learnable attention threshold per head/layer
* Test on ImageNet and larger datasets
* Apply to other tasks like detection and segmentation
* Combine with attention rollout for deeper insights

---

## 🙌 Acknowledgments

* [Timm Library](https://github.com/huggingface/pytorch-image-models)
* [Grad-CAM](https://arxiv.org/abs/1610.02391)
* [Captum by PyTorch](https://captum.ai/)
