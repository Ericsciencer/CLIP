# Contrastive Language-Image Pre-training (CLIP)

### 选择语言 | Language

[中文简介](#简介) | [English](#introduction)

### 结果 | Result

训练结果：
<img width="1784" height="581" alt="image" src="https://github.com/user-attachments/assets/6fc5a4de-e6b1-4862-b543-e5184f1a0449" />

图文相似度矩阵可视化（测试集 10 类 × 10 类）：
<img width="1165" height="1031" alt="image" src="https://github.com/user-attachments/assets/d779adb3-6139-464a-83ab-695a1de17942" />


---

## 简介

Contrastive Language-Image Pre-training（简称 CLIP）由 **Alec Radford, Jong Wook Kim, Chris Hallacy 等人** 于 2021 年在 OpenAI 提出（ICML 2021），是**多模态视觉-语言表征学习领域的奠基性工作**。其核心贡献在于：**首次以大规模自然语言监督替代人工标注类别标签**，通过**双塔对比学习**将图像与文本映射到同一语义空间，使模型仅凭文本提示即可完成零样本图像分类，彻底改变了视觉模型的训练与使用范式，为后续的图文检索、文生图、视觉问答、多模态大模型（如 DALL·E、Stable Diffusion、LLaVA 等）奠定了核心架构与训练范式基础。

在 CLIP 出现之前，计算机视觉模型普遍依赖固定类别的有监督标注（如 ImageNet 的 1000 类），模型只能识别闭集标签，跨任务迁移需要重新标注与微调。尽管视觉-语言联合建模已有若干探索（如 VirTex、ICMLM、ConVIRT 等），但普遍依赖特定任务结构或小规模图文对。CLIP 首次系统性地验证了「**用 4 亿互联网图文对 + 对称对比学习目标**」可以直接学到强迁移性的视觉表征，并证明了零样本分类的可行性与竞争力：在 ImageNet 上，CLIP 零样本即可达到与 ResNet-50 有监督训练相当的水平，且具备显著的分布鲁棒性。

在 ImageNet-1K 零样本分类任务上，CLIP ViT-L/14 在 336 分辨率微调后达到 **76.6% Top-1 准确率**，无需任何 ImageNet 训练样本；在 27 个下游数据集上的平均零样本准确率全面超过同期有监督基线。本复现基于 PyTorch 从零实现完整的 CLIP 架构，严格遵循原论文的**双塔对称对比学习**、**可学习温度参数**、**L2 归一化余弦相似度**、**对称 InfoNCE 损失**等核心设计，采用 **CIFAR-10 彩色图像数据集**构造图文对进行验证，使用零样本分类准确率作为核心评估指标，完整包含数据预处理、图文对构造、模型训练、零样本评估、图文相似度可视化与训练曲线可视化全流程。

## 架构

本次复现严格遵循《Learning Transferable Visual Models From Natural Language Supervision》原文的标准双塔架构，整体分为「图像编码器」「文本编码器」「对比对齐头」三大模块，完整保留原论文的所有核心设计。

- **图像编码器（Image Encoder）**：采用 Vision Transformer 架构。输入图像被切分为固定大小的图像块（Patch），经线性投影得到 Patch 嵌入序列，拼接一个可学习的 CLS 令牌后叠加可学习位置编码，送入堆叠 Transformer 编码器；取 CLS 令牌的最终输出作为图像全局表征，再经线性投影映射到共享嵌入空间，并做 L2 归一化。原论文中图像塔可选 ResNet 或 ViT，本复现采用 ViT 以保持结构一致性。
- **文本编码器（Text Encoder）**：采用因果 Transformer 架构。输入文本经 Token Embedding 与可学习位置编码后，送入带因果掩码的 Transformer 编码器，取句末 EOT（End-of-Text）位置的特征作为文本全局表征，再经 `text_projection` 线性投影映射到共享嵌入空间，并做 L2 归一化。因果掩码保证每个位置只能看到其之前的 token，与原论文一致。
- **共享嵌入空间与 L2 归一化**：两个编码器的输出都被投影到 `embed_dim` 维空间并 L2 归一化，使图像与文本特征落在同一单位球面上，点积即为余弦相似度，从而保证跨模态相似度的尺度一致性与可比性。
- **可学习温度参数（logit_scale）**：原论文发现对比学习中相似度矩阵的尺度对训练稳定性至关重要，因此引入一个**可学习的对数尺度参数**（初值 `log(1/0.07) ≈ 2.659`，对应温度 τ≈0.07），在训练中自动调节相似度分布的锐化程度，无需人工调参。
- **对称 InfoNCE 损失**：对一个大小为 B 的 batch，计算 B×B 的图像-文本相似度矩阵，以对角线为正样本、其余为负样本，分别在「图像→文本」与「文本→图像」两个方向上计算交叉熵损失并取平均。该对称设计保证两种模态的特征都能被均匀优化，避免单向偏置。
- **零样本推理**：推理时无需分类头，将所有类别名套入提示模板（如 `"a photo of a {class}"`）编码为文本特征矩阵，与图像特征做点积后取 argmax 即为预测类别。

<img width="1238" height="738" alt="image" src="https://github.com/user-attachments/assets/79b60250-8e5a-47a6-8076-fcd3ff25c8d4" />



**Mini-CLIP 配置**：为降低训练开销、便于快速复现验证，本实现采用轻量化 Mini-CLIP 配置——图像塔为 6 层、256 维、8 头的 ViT（Patch 4×4，32×32 输入共 64 个 Patch）；文本塔为 6 层、128 维、4 头的 Transformer（上下文长度 77）；共享嵌入维度 128。模型结构、双塔对比学习目标、温度参数、零样本评估范式均严格对齐原论文标准。

**注意**：原论文使用 4 亿互联网图文对，本复现使用 CIFAR-10 训练集的 10 个类别名自动构造图文对（`"a photo of a {class}"`），属于**小规模可复现的教学级对齐验证**，因此零样本准确率不可与原论文直接对比。此外，本实现采用字符级哈希 Tokenizer 代替原论文的 BPE 分词器，词表利用率较低，也会影响最终效果。

## 数据集

本次复现使用 **CIFAR-10 彩色图像数据集** 构造图文对，是计算机视觉领域经典的小型基准数据集。

- **任务类型**：10 类别图文对比对齐任务（下游评估为零样本图像分类）。
- **数据规模**：训练集 50000 张图像，测试集 10000 张图像；所有图像均为 32×32 分辨率三通道彩色图片，涵盖飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车共 10 个日常物体类别。**为控制训练开销，本复现从每类中分层抽样，训练集默认取 300 张/类（共 3000 张），测试集默认取 100 张/类（共 1000 张）**，可通过 `train_per_class`、`test_per_class` 调节。
- **图文对构造**：每条样本的文本描述由类别标签套入固定提示模板得到，即 `text = "a photo of a {class_name}"`，与原论文零样本推理时使用的提示模板保持一致，保证训练与评估的文本分布对齐。
- **预处理方式**：训练集采用随机裁剪（带填充）、随机水平翻转的数据增强策略，缓解过拟合；测试集仅做归一化；使用 CIFAR-10 全局统计的均值与标准差进行像素值归一化。
- **评估指标**：采用多模态领域通用的 **Zero-Shot Top-1 准确率**，即在完全冻结模型、不训练任何分类头的条件下，用文本提示直接匹配图像，预测类别与真实标签一致的样本比例。同时监控**对称对比损失**与**可学习温度**，输出损失、准确率、温度三维度训练曲线。

| Zero-Shot Top-1 准确率 | 效果说明 |
| --- | --- |
| 10%（随机水平） | 图文特征未有效对齐，零样本分类等价于随机猜测 |
| 25 ~ 40% | 初步建立跨模态对齐，能区分部分语义差异明显的类别 |
| 40 ~ 55% | 图文对齐良好，常见类别识别稳定，模型具备基础零样本能力 |
| 55 ~ 70% | 对齐质量较高，接近轻量有监督分类模型水平，泛化性良好 |
| 70%+ | 优秀对齐效果，接近全量 CIFAR-10 训练下 Mini-CLIP 的性能上限 |

> 提示：小规模图文对（每类数百张）下，零样本准确率主要受限于**正样本多样性不足**——同一类别只有一种文本提示，对比学习容易退化为「按类别聚类」而非「按语义对齐」。若想提升准确率，可优先考虑：① 每类构造多种提示模板（prompt ensemble）；② 增大 `train_per_class`；③ 将 Tokenizer 替换为真正的子词分词器。

数据集官方地址：[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Introduction

Contrastive Language-Image Pre-training (CLIP for short), proposed by **Alec Radford, Jong Wook Kim, Chris Hallacy et al.** at OpenAI in 2021 (ICML 2021), is a **foundational work in multimodal vision-language representation learning**. Its core contribution is that it **replaces manually annotated class labels with large-scale natural language supervision for the first time**, and maps images and texts into a shared semantic space via **dual-tower contrastive learning**, enabling zero-shot image classification with only textual prompts. This fundamentally reshapes the training and usage paradigm of vision models, and lays the architectural and training foundation for subsequent image-text retrieval, text-to-image generation, visual question answering, and multimodal foundation models (e.g., DALL·E, Stable Diffusion, LLaVA).

Before CLIP, computer vision models generally relied on supervised annotation of a fixed set of classes (e.g., 1000 classes in ImageNet). Models could only recognize closed-set labels, and cross-task transfer required re-annotation and fine-tuning. Although joint vision-language modeling had been explored (e.g., VirTex, ICMLM, ConVIRT), these approaches typically relied on task-specific structures or small-scale image-text pairs. CLIP was the first to systematically validate that "**400M internet image-text pairs + a symmetric contrastive objective**" can directly learn highly transferable visual representations, and demonstrated the feasibility and competitiveness of zero-shot classification: on ImageNet, CLIP achieves zero-shot performance comparable to supervised ResNet-50, with notable distributional robustness.

On the ImageNet-1K zero-shot classification task, CLIP ViT-L/14 fine-tuned at 336 resolution reaches **76.6% Top-1 accuracy** without any ImageNet training samples; its average zero-shot accuracy across 27 downstream datasets comprehensively surpasses contemporaneous supervised baselines. This reproduction implements the complete CLIP architecture from scratch based on PyTorch, strictly following the original paper's core designs: **dual-tower symmetric contrastive learning**, **learnable temperature parameter**, **L2-normalized cosine similarity**, and **symmetric InfoNCE loss**. It is validated on the **CIFAR-10 color image dataset** with automatically constructed image-text pairs, uses zero-shot classification accuracy as the core evaluation metric, and covers the full pipeline of data preprocessing, image-text pair construction, model training, zero-shot evaluation, image-text similarity visualization, and training curve visualization.

## Architecture

This implementation faithfully follows the standard dual-tower architecture from *Learning Transferable Visual Models From Natural Language Supervision*, consisting of three main modules: image encoder, text encoder, and contrastive alignment head, with all core designs from the original paper preserved.

- **Image Encoder**: A Vision Transformer. The input image is split into fixed-size patches, linearly projected into a patch embedding sequence, prepended with a learnable CLS token, augmented with learnable positional encodings, and fed into a stacked Transformer encoder. The final output of the CLS token serves as the global image representation, which is then linearly projected into the shared embedding space and L2-normalized. The original paper allows ResNet or ViT for the image tower; this reproduction adopts ViT for structural consistency.
- **Text Encoder**: A causal Transformer. The input text is embedded via token embedding and learnable positional encoding, then passed through a Transformer encoder with a causal mask. The feature at the EOT (End-of-Text) position is taken as the global text representation, linearly projected by `text_projection` into the shared embedding space, and L2-normalized. The causal mask ensures each position can only attend to preceding tokens, consistent with the original paper.
- **Shared Embedding Space & L2 Normalization**: Outputs of both encoders are projected into an `embed_dim`-dimensional space and L2-normalized, placing image and text features on the same unit hypersphere. Their dot product thus equals cosine similarity, ensuring scale consistency and comparability of cross-modal similarity.
- **Learnable Temperature (logit_scale)**: The original paper found that the scale of the similarity matrix is critical to contrastive training stability. A **learnable log-scale parameter** is therefore introduced (initialized as `log(1/0.07) ≈ 2.659`, corresponding to τ≈0.07) to automatically adjust the sharpness of the similarity distribution during training, eliminating manual tuning.
- **Symmetric InfoNCE Loss**: For a batch of size B, a B×B image-text similarity matrix is computed with the diagonal as positives and the rest as negatives. Cross-entropy loss is computed in both the "image→text" and "text→image" directions and averaged. This symmetric design ensures both modalities are uniformly optimized, avoiding one-sided bias.
- **Zero-Shot Inference**: No classification head is needed at inference. All class names are wrapped in a prompt template (e.g., `"a photo of a {class}"`) and encoded into a text feature matrix; the predicted class is the argmax of the dot product with the image feature.

**Mini-CLIP Configuration**: To reduce training cost and facilitate quick reproduction, this implementation adopts a lightweight Mini-CLIP configuration — a 6-layer, 256-dim, 8-head ViT image tower (4×4 patches, 64 patches for a 32×32 input); a 6-layer, 128-dim, 4-head Transformer text tower (context length 77); and a shared embedding dimension of 128. Model structure, dual-tower contrastive objective, temperature parameter, and zero-shot evaluation paradigm all strictly align with the original paper.

**Note**: The original paper uses 400M internet image-text pairs; this reproduction automatically constructs image-text pairs from the 10 class names of the CIFAR-10 training set (`"a photo of a {class}"`), serving as a **small-scale, reproducible, pedagogical alignment validation**, so zero-shot accuracy is not directly comparable to the original paper. Additionally, a character-level hashing tokenizer is used instead of the original BPE tokenizer, resulting in low vocabulary utilization and affecting final performance.

## Dataset

This reproduction uses the **CIFAR-10 color image dataset** with constructed image-text pairs, a classic small benchmark dataset in computer vision.

- **Task**: 10-class image-text contrastive alignment (downstream evaluation: zero-shot image classification).
- **Data Scale**: 50,000 training images and 10,000 test images. All images are 32×32 resolution three-channel color pictures, covering 10 daily object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. **To control training cost, this reproduction performs stratified subsampling per class: 300 per class (3,000 total) for training and 100 per class (1,000 total) for testing by default**, adjustable via `train_per_class` and `test_per_class`.
- **Image-Text Pair Construction**: The text description of each sample is obtained by wrapping the class label in a fixed prompt template, i.e., `text = "a photo of a {class_name}"`, consistent with the prompt template used in zero-shot inference in the original paper, ensuring alignment between training and evaluation text distributions.
- **Preprocessing**: The training set adopts data augmentation strategies including random crop (with padding) and random horizontal flip to alleviate overfitting; the test set only performs normalization. Pixel value normalization uses CIFAR-10's global mean and standard deviation.
- **Evaluation Metric**: **Zero-Shot Top-1 Accuracy**, the universal metric for multimodal tasks — the proportion of samples whose predicted class matches the ground truth when the model is fully frozen, no classification head is trained, and classes are matched purely via textual prompts. Symmetric contrastive loss and the learnable temperature are also monitored, with a three-dimensional training curve (loss / accuracy / temperature) output.

Official dataset page: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 原文章 | Original article

Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision[C]//International Conference on Machine Learning. PMLR, 2021: 8748-8763.
