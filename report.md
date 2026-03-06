# Visual Intelligence Project — Binary Image Classification using CNN and ScatNet

**Course:** Visual Intelligence
**Task:** Binary Image Classification — Cats vs. Dogs
**Models:** Custom VGG-style CNN and Kymatio ScatNet

---

## 1. Introduction

Image classification is one of the foundational problems in computer vision, and deep learning methods have produced remarkable advances in this area over the past decade. This report presents a comparative study of two distinct approaches to binary image classification on the well-known Cats vs. Dogs dataset: a custom Convolutional Neural Network (CNN) designed in the VGG style, and a Scattering Network (ScatNet) built on the mathematical framework of wavelet scattering transforms provided by the Kymatio library. The two architectures represent different philosophies of feature extraction — the CNN learns hierarchical representations entirely from data through gradient-based optimisation, while ScatNet computes fixed, mathematically defined scattering coefficients and learns only a lightweight classification head on top of them.

Beyond classification performance, this work applies six Explainability AI (XAI) methods to both trained models in order to interpret and compare the spatial focus of their predictions. Understanding where a model "looks" when making a decision is increasingly important both for scientific validation and for identifying potential failure modes. The six methods employed — GradCAM, Integrated Gradients, Saliency Maps, DeepLIFT, Guided Backpropagation, and Occlusion — span gradient-based, attribution-based, and perturbation-based families of explanation, allowing a comprehensive view of model behaviour.

The remainder of this report is structured as follows. Section 2 describes the dataset and preprocessing pipeline. Sections 3 and 4 detail the CNN and ScatNet architectures together with their respective training configurations. Section 5 presents the cross-validation and test results. Section 6 discusses filter visualisations for both models. Sections 7 and 8 describe the XAI methods and analyse their outputs. Section 9 concludes with a comparative discussion and suggestions for future work.

---

## 2. Dataset

The dataset used in this project is a subset of the Kaggle Cats vs. Dogs benchmark, a standard binary classification benchmark derived from a Microsoft Research challenge. The full pool of images was partitioned into a training and validation set of 22,498 images and a held-out test set of 2,500 images. The two classes — cats and dogs — are balanced across all splits to ensure that accuracy is a meaningful evaluation metric without requiring class-weighted corrections.

All images were resized to 128×128 pixels during loading, a resolution chosen to balance spatial detail with computational tractability across five-fold cross-validation and repeated training runs. Pixel values were normalised using the ImageNet channel-wise mean and standard deviation (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), as this initialisation is known to accelerate convergence when using architectures and weight initialisations derived from ImageNet pretraining conventions.

A pre-flight cleaning step was applied before any training to remove images that could not be loaded by Pillow even with truncated-image tolerance enabled. Images that were not in RGB mode were converted before use. This ensured that no corrupt or grayscale-only images propagated into the data pipeline and contaminated gradient estimates during training.

Data augmentation was applied during training to improve generalisation. For the CNN, the augmentation pipeline comprised random horizontal and vertical flips, random rotation up to 25 degrees, colour jitter (brightness, contrast, saturation, and hue perturbations), random conversion to grayscale with a 5% probability, and random erasing of small rectangular patches. For ScatNet, a lighter augmentation scheme was used — random horizontal flip, random rotation up to 15 degrees, colour jitter, and random erasing — reflecting the fact that wavelet scattering transforms already encode a degree of translation and deformation invariance by construction, reducing the need for aggressive spatial augmentation.

---

## 3. CNN Architecture and Training

### 3.1 Architecture

The custom CNN follows the VGG design philosophy of stacking double convolutional blocks separated by spatial downsampling. The network accepts inputs of shape 3×128×128 and processes them through four sequential blocks, each consisting of two successive Conv2d–BatchNorm2d–ReLU units followed by a MaxPool2d layer with stride 2. The channel widths progress as 3→32, 32→64, 64→128, and 128→256, doubling at each stage to compensate for the spatial reduction. All convolutions use a 3×3 kernel with padding 1 to preserve spatial dimensions within each block. After the four blocks the spatial map is 256×8×8; an AdaptiveAvgPool2d layer collapses this to 256×1×1, which is then flattened to a 256-dimensional vector and passed to a shared ClassifierHead module.

The ClassifierHead consists of a fully connected layer mapping the input features to 256 units, followed by ReLU activation, a Dropout layer with probability 0.5, and a final linear projection to a single logit. This logit is interpreted as the log-odds of the "dog" class; predictions are obtained by applying a sigmoid and thresholding at 0.5. The loss function is Binary Cross-Entropy with Logits (BCEWithLogitsLoss), which combines the sigmoid and BCE in a numerically stable manner.

Weight initialisation followed the Kaiming normal scheme for convolutional layers (fan-out mode, ReLU nonlinearity), Xavier normal for linear layers, and unit weights with zero bias for BatchNorm layers. This combination is well-suited to the predominantly ReLU activations in the network and helps avoid vanishing or exploding gradients at initialisation.

### 3.2 Training Configuration

Training used the AdamW optimiser with a base learning rate of 3×10⁻⁴ and weight decay of 5×10⁻⁴. The learning rate schedule followed OneCycleLR with a maximum learning rate of 1.5×10⁻³ and a warm-up fraction (pct_start) of 0.1, meaning the rate increased linearly over the first 10% of steps before annealing. Gradient norms were clipped to a maximum of 1.0 to prevent exploding gradients in early epochs. Training ran for up to 20 epochs per fold with early stopping patience of 5 epochs, halting when validation accuracy did not improve. A mild label smoothing of 0.05 was applied during training to reduce overconfidence in the loss computation. The batch size was 64 throughout.

---

## 4. ScatNet Architecture and Training

### 4.1 Scattering Transform Background

The scattering transform, introduced by Mallat (2012), is a cascade of wavelet modulus operators that produces translation-invariant, deformation-stable representations of signals. Unlike learned convolutional filters, the scattering filters are fixed Morlet wavelets parameterised by scale J and orientation L. For an input image of spatial size H×W, the second-order scattering transform with parameters J and L produces a coefficient tensor of dimensionality determined by the number of first- and second-order paths through the wavelet tree. These coefficients are inherently multi-scale and encode texture and shape information in a form that is stable to small geometric perturbations.

### 4.2 Architecture

The ScatNet used in this project employs Kymatio's Scattering2D with J=3 octaves of scale and L=8 orientations per scale. For a 3-channel, 128×128 input, the scattering transform outputs a tensor of shape B×651×16×16, where 651 is the number of scattering coefficient channels across all orders and RGB channels combined, and 16×16 is the reduced spatial resolution (downsampled by 2^J = 8). These fixed, non-trainable features are passed to a small trainable convolutional head called conv_head, which consists of a Conv2d reducing 651 channels to 256, followed by BatchNorm2d and ReLU, then a second Conv2d from 256 to 128 channels, BatchNorm2d and ReLU, and finally an AdaptiveAvgPool2d(1). The resulting 128-dimensional vector is processed by the same ClassifierHead structure as the CNN (128→256→1), giving a single output logit. Only the conv_head and ClassifierHead parameters are updated during training; the scattering transform itself is deterministic and requires no gradient computation.

A thin ScatteringWrapper module was introduced to encapsulate the Kymatio Scattering2D object inside a standard nn.Module, enabling it to be moved to GPU via .to(device) and to participate in hook registration for gradient-based XAI methods. The wrapper's forward method calls the Kymatio transform and reshapes its 5-dimensional output to the 4-dimensional (B, C, H, W) format expected by the conv_head.

### 4.3 Training Configuration

ScatNet used the same AdamW optimiser, OneCycleLR scheduler, gradient clipping, BCEWithLogitsLoss, batch size, and early stopping settings as the CNN. The lighter augmentation policy described in Section 2 was applied. Because the scattering stage is frozen, the effective number of trainable parameters is substantially smaller than in the CNN, which means ScatNet converges in fewer effective gradient updates but also has less representational capacity for capturing class-discriminative patterns beyond those expressible as linear combinations of scattering coefficients.

---

## 5. K-Fold Cross-Validation Results

Both models were evaluated using five-fold stratified cross-validation on the combined training and validation pool of 22,498 images. StratifiedKFold was used to ensure that each fold maintained the same class balance as the full pool. After cross-validation, each model was retrained on the full training and validation data for final evaluation on the held-out test set.

The CNN achieved a mean cross-validation accuracy of 95.34% and a mean F1 score of 95.33% across the five folds, demonstrating consistent and high generalisation performance. On the held-out test set, the CNN reached 96.92% accuracy and 96.93% F1 score, indicating that the final model trained on the full pool was slightly stronger than the per-fold estimates, as expected. Learning curves for the CNN across all five folds are shown in Figure 1.

ScatNet achieved a mean cross-validation accuracy of 89.13% and a mean F1 score of 89.34%, reflecting the inherent trade-off between the model's built-in geometric invariance and its reduced capacity for purely data-driven feature learning. The test set accuracy of ScatNet was 93.40% with a test F1 score of 93.43%, showing that the final model trained on the full pool generalised well beyond the fold-average estimate. Learning curves for ScatNet are shown in Figure 2.

The gap of approximately six percentage points between the CNN and ScatNet in cross-validation narrows somewhat when comparing their final test results, suggesting that ScatNet benefits more from additional training data than the CNN does. This is consistent with the smaller effective parameter count of ScatNet: each additional image seen during full-pool training contributes proportionally more to the classifier head's decision boundary calibration.

---

## 6. Filter Visualisation and Comparison

### 6.1 CNN Learned Filters

The 32 filters in the first convolutional layer of the CNN were visualised after training to inspect the low-level features the network learned to detect (Figure 3). The filters exhibit a diverse range of orientations, spatial frequencies, and colour-opponent patterns, including edge detectors at multiple angles, colour channels selective for reddish or greenish tones, and centre-surround structures. This diversity is consistent with filters observed in well-trained convolutional networks and suggests that the first block successfully specialised its filters to the textural statistics of natural pet images. Filters were visualised by normalising each 3×3×3 weight tensor to the [0, 1] range independently.

### 6.2 ScatNet Morlet Wavelet Filters

Rather than learned filters, ScatNet employs fixed first-order Morlet wavelets ψ parameterised by scale and orientation (Figure 4). The J=3 first-order wavelets were visualised in the spatial domain by computing the inverse Fourier transform of each filter's frequency-domain representation and applying an fftshift to centre the wavelet. The resulting images show the characteristic oscillatory Gabor-like structure of Morlet wavelets: each filter is elongated along a preferred orientation and decays with a Gaussian envelope. The L=8 orientations span the full angular range at each scale, providing isotropic coverage of directional texture features. Unlike the CNN filters, these wavelets are determined analytically and do not change during training, making them fully interpretable but also less adaptive to the specific statistics of the Cats vs. Dogs domain.

---

## 7. XAI Methods

Six explainability methods were applied to both trained models on the same four test images (two cats and two dogs), providing a consistent basis for comparison. The methods span gradient-based, integrated attribution, and perturbation-based paradigms.

**GradCAM (Gradient-weighted Class Activation Mapping)** was implemented from scratch using PyTorch forward and backward hooks, following the original formulation of Selvaraju et al. (2017). During a forward pass, the activation tensor at the target convolutional layer is stored. A backward pass with respect to the predicted class score propagates gradients back to the same layer. The global-average-pooled gradients serve as channel importance weights, and a weighted sum of activation maps is computed and passed through ReLU to retain only positively contributing spatial regions. The resulting 2-D map is normalised to [0, 1] and upsampled to the input image resolution for overlay. For the CNN the target layer is the final convolutional block (model.features[3]), and for ScatNet it is the trainable convolutional head (model.conv_head), which is the last spatially resolved layer with meaningful gradient flow.

**Integrated Gradients**, introduced by Sundararajan et al. (2017), attributes a prediction to each input feature by integrating the model's gradients along a straight path from a baseline input (a black image of all zeros) to the actual input. With 50 integration steps, the method provides an attribution that satisfies the completeness axiom: the sum of attributions equals the difference in model output between the input and the baseline. In this work, the absolute values of the channel-wise attributions are averaged across the three colour channels to produce a single-channel saliency map.

**Saliency Maps** represent the simplest gradient-based attribution method: the absolute value of the gradient of the class score with respect to each input pixel. This gradient indicates the sensitivity of the prediction to infinitesimal perturbations at each spatial location and provides a first-order approximation of feature importance. Saliency maps are computed using Captum's Saliency module with absolute-value output, followed by channel aggregation.

**DeepLIFT** (Deep Learning Important FeaTures), proposed by Shrikumar et al. (2017), decomposes the difference in model output from a reference baseline into contributions from each input neuron using a backpropagation-based rule that compares activations to their reference values. DeepLIFT is well-suited to architectures with standard feedforward connections. However, because ScatNet's scattering transform is a fixed non-parametric operator without learnable weights, the DeepLIFT reference-propagation rule cannot be meaningfully applied to the scattering stage, and the method was found to raise a runtime exception for ScatNet. In those cases, the attribution is skipped gracefully and the corresponding panel in the output figure is marked as not available.

**Guided Backpropagation**, introduced by Springenberg et al. (2015), modifies the standard gradient backpropagation rule so that only positive gradients flowing through ReLU activations are propagated backward. This yields sharper, less noisy attribution maps than vanilla saliency by suppressing gradients that are negative at either the forward activation or the backward gradient. The method was applied using Captum's GuidedBackprop module.

**Occlusion** is a perturbation-based method that systematically replaces patches of the input image with a baseline value (zero) and measures the change in the model's output score. The magnitude of the output change when a given patch is occluded provides a direct measure of that region's contribution to the prediction. A sliding window of size 15×15 pixels with a stride of 8 pixels was used across all three channels simultaneously, with a zero (black) baseline. This produces a dense attribution map at roughly 15-pixel resolution, which is then bilinearly upsampled to the full image size.

---

## 8. XAI Results and Analysis

### 8.1 Per-Image Attribution Maps

For each of the four test images, two figures were generated: one showing all six attributions for the CNN (Figures 5–8) and one for ScatNet (Figures 9–12). Each figure contains seven panels: the original image followed by the six attribution overlays, each produced by blending the JET-colourmap heatmap with the original image at an opacity of 0.5.

Across both models, the gradient-based methods — GradCAM, Saliency, and Guided Backpropagation — tended to produce sharper, more localised heatmaps concentrated on the animal's head and face region. Integrated Gradients produced attribution maps of intermediate sharpness, with the highlighted region generally encompassing both the animal's face and portions of the body. Occlusion produced broader, lower-resolution maps due to its larger window size, but consistently identified the central foreground region containing the subject as most discriminative. DeepLIFT produced meaningful results for the CNN and showed strong concentration on the facial region, consistent with the other gradient-based methods.

For ScatNet, the attribution patterns were broadly similar in terms of localisation, with GradCAM and Guided Backpropagation highlighting the facial region of cats and dogs respectively. Integrated Gradients for ScatNet tended to produce more diffuse activations, possibly reflecting the fact that the conv_head operates on scattering coefficients that encode frequency-band information more globally than the pointwise feature maps of the CNN.

### 8.2 GradCAM Comparison: From-Scratch vs. Captum

Figure 13 presents a side-by-side comparison of the from-scratch GradCAM implementation against Captum's LayerGradCam, applied to both models across all four test images. The heatmaps produced by the two implementations are visually nearly identical for both the CNN and ScatNet in all cases, validating the correctness of the custom GradCAM implementation. The minor pixel-level differences between the two outputs are attributable to slight differences in how each implementation handles numerical precision during the gradient accumulation step, and do not affect the qualitative interpretation of the results.

### 8.3 Misclassification Analysis

A notable finding arose for the first dog image (dogs #0): both the CNN and ScatNet classified this image as a cat. Inspection of the XAI heatmaps reveals that both models concentrated attention on the background region of the image rather than on the animal itself. The GradCAM overlay in particular shows high activation in areas of the scene that contain texture patterns — such as fur-like fabric or carpet — that the models appear to have associated with the cat class. This behaviour is consistent with known dataset biases in which background context leaks into the classifier's decision boundary. The finding illustrates the practical value of XAI methods: without attribution analysis, the misclassification might appear as a random error, but the heatmaps reveal a systematic reliance on spurious background features that could be addressed through more aggressive masking or region-of-interest augmentation during training.

---

## 9. Discussion and Conclusions

### 9.1 Comparative Performance

The CNN outperformed ScatNet on both cross-validation and test set metrics. The CNN's test accuracy of 96.92% versus ScatNet's 93.40% represents a gap of approximately 3.5 percentage points. This difference reflects the fundamental trade-off between the two approaches: the CNN's fully learned feature hierarchy adapts entirely to the specific visual statistics of the training data, enabling it to discover discriminative patterns that may not align with the fixed orientation and scale structure of Morlet wavelets. ScatNet, by contrast, provides theoretical guarantees of stability to deformations and does not require as much data to reach reasonable performance, but its fixed feature extractor imposes a ceiling on the representational expressiveness that can be achieved without retraining the scattering filters.

It is also worth noting that ScatNet required significantly fewer trainable parameters — only the conv_head and ClassifierHead — compared to the CNN's full four-block architecture. This property makes ScatNet attractive in settings where labelled data is scarce or computational resources are limited, as the model can generalise reasonably well from fewer gradient updates.

### 9.2 Insights from XAI

The XAI analysis demonstrated that both models attend to semantically meaningful regions of the input in the majority of cases, lending confidence to the validity of their predictions. The high degree of agreement between the from-scratch GradCAM implementation and Captum's LayerGradCam serves both as a methodological validation and as a cross-verification tool that can be used in future work to confirm that custom attribution implementations are correctly coded.

The misclassification of dogs #0 by both models, combined with the observation that both models' heatmaps focus on background texture in that image, suggests that the Cats vs. Dogs dataset contains a subset of images in which the background context is more visually salient than the foreground animal. This finding motivates further investigation into dataset curation, including saliency-guided cropping or foreground-background segmentation as a preprocessing step to reduce spurious correlations.

DeepLIFT's incompatibility with ScatNet highlights an important practical consideration when applying attribution methods to non-standard architectures: methods that rely on reference-value propagation through every layer of the network require that all operations be decomposable into input–output difference terms, which fixed non-parametric operators do not satisfy. Future work integrating ScatNet-style fixed feature extractors with attribution methods should consider perturbation-based approaches such as Occlusion or LIME as primary alternatives, since these methods treat the model as a black box and do not require access to internal gradients.

### 9.3 Conclusions

This project successfully demonstrated the design, training, and analysis of two architecturally distinct image classifiers on a large-scale binary classification benchmark. The custom CNN achieved strong performance with 96.92% test accuracy, while the ScatNet achieved 93.40% test accuracy with a significantly smaller trainable parameter count. Both models were thoroughly analysed using six state-of-the-art XAI methods, revealing that gradient-based and attribution-based methods consistently identify the animal's face as the primary discriminative region, while also uncovering a specific failure mode caused by background texture bias. The comparative GradCAM analysis confirmed the correctness of the custom from-scratch implementation. Together, these results validate the experimental pipeline and provide a foundation for future work exploring scattering-enhanced hybrid architectures, improved data augmentation strategies targeting background suppression, and extended attribution analysis over the full test set.

---

## References

Mallat, S. (2012). Group Invariant Scattering. *Communications on Pure and Applied Mathematics*, 65(10), 1331–1398.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.

Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important Features Through Propagating Activation Differences. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 3145–3153.

Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv preprint arXiv:1409.1556*.

Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2015). Striving for Simplicity: The All Convolutional Net. *ICLR Workshop*.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 3319–3328.

Andreux, M., Angles, T., Exarchakis, G., Leonarduzzi, R., Rochette, G., Thiry, L., Zarka, J., Mallat, S., Andén, J., Belilovsky, E., Bruna, J., Lostanlen, V., Chaudhary, M., Hirn, M. J., Oyallon, E., Zhang, S., Cella, C., & Eickenberg, M. (2020). Kymatio: Scattering Transforms in Python. *Journal of Machine Learning Research*, 21(60), 1–6.

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., Melnikov, A., Kliushkina, N., Araya, C., Yan, S., & Reblitz-Richardson, O. (2020). Captum: A unified and generic model interpretability library for PyTorch. *arXiv preprint arXiv:2009.07896*.

---

*Figure references in the text correspond to the following saved outputs:*
*Figure 1 — outputs/figures/cnn_learning_curves.png*
*Figure 2 — outputs/figures/scatnet_learning_curves.png*
*Figure 3 — outputs/figures/cnn_filters.png*
*Figure 4 — outputs/figures/scatnet_filters.png*
*Figures 5–8 — outputs/figures/xai_cnn_{cats/dogs}_{0/1}.png*
*Figures 9–12 — outputs/figures/xai_scatnet_{cats/dogs}_{0/1}.png*
*Figure 13 — outputs/figures/xai_gradcam_comparison.png*
