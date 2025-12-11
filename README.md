# Monet-Style Image Generation with Generative Deep Learning

This repository contains a deep learning project for generating high-quality images in the artistic style of Claude Monet using generative models. The work was developed for the Kaggle GAN Getting Started competition and executed primarily in a Kaggle environment.

> **Important**
> The Jupyter notebook contains detailed justifications, analysis, and code. Please refer to the plot renderings in `results/plots/` or run them directly in Kaggle.
>
> https://www.kaggle.com/code/walmsleylab/monet-style-generation-with-gans

## Problem Overview

The task is to learn the underlying distribution of Monet paintings and generate visually plausible images that resemble impressionist artwork. This is an unpaired image-to-image translation and generative modeling problem that requires capturing characteristic stylistic features such as soft edges, saturated color palettes, textured brush-like patterns, and reduced fine-grain detail. Model performance is evaluated using a hidden perceptual metric, making visual realism and stylistic consistency central to success.

## Repository Structure

```
monet-style-generation/
├── notebook/                    # Kaggle-compatible experiment notebook
├── results/
│   ├── model_comparison.csv     # Aggregated evaluation results
│   ├── plots/                   # Training convergence and EDA figures
│   └── summary.md               # Short experiment summary
├── models/                      # Trained model checkpoints (.pt)
├── README.md
└── requirements.txt
```

## Dataset

The dataset consists of two unpaired image domains from the Kaggle GAN Getting Started competition:

| Domain | Images | Description |
|--------|--------|-------------|
| Monet Paintings | 300 | Impressionist artwork with soft edges, saturated colors, brush-like textures |
| Photographs | 7,038 | Real-world images with sharper edges, higher structural contrast |

All images are resized and normalized to $256 \times 256$ spatial resolution with pixel values scaled to the range $[-1, 1]$.

## Models Evaluated

Three generative approaches were implemented and compared:

| Model | Architecture | Training Objective | Key Characteristics |
|-------|--------------|-------------------|---------------------|
| **DCGAN** | Convolutional GAN | Adversarial | Direct noise-to-image mapping; fast but unstable |
| **CycleGAN** | Dual Generator/Discriminator | Adversarial + Cycle Consistency | Bidirectional translation; preserves structure |
| **Diffusion** | DDPM-lite UNet | Denoising (MSE) | Iterative refinement; stable but slow to converge |

Training stabilization techniques applied across models:
- Learning rate decay and scheduling
- Exponential moving average (EMA) tracking
- Early stopping based on proxy FID
- Gradient clipping (CycleGAN)

## Key Results

| Model | Best Proxy FID | Final Diversity | Best Epoch | Epochs Trained | Avg Time/Epoch (s) |
|-------|----------------|-----------------|------------|----------------|-------------------|
| **CycleGAN** | **72.84** | 0.265 | 15 | 20 | 15.17  |
| DCGAN        | 146.19    | 0.219 | 24 | 28 | 0.60   |
| Diffusion    | 0.72*     | 0.606 | 5  | 10 | 0.46.  |

*\*Diffusion model metrics are misleading due to insufficient training; outputs remained noise-like despite favorable statistics.*

### Visual Quality Assessment

- **CycleGAN** produced the most visually compelling outputs with coherent scenes, painterly texture, and color blending reminiscent of Monet's style
- **DCGAN** achieved reasonable metrics but exhibited tiling artifacts and repetitive textures
- **Diffusion** failed to produce structured imagery within the limited training budget

The **CycleGAN** model was selected for final submission based on superior perceptual quality.

## Training Convergence Diagnostics

Four complementary metrics are tracked across epochs:

1. **Adversarial Losses** — Generator/discriminator balance
2. **Proxy FID** — Feature-space distance to Monet distribution
3. **Output Diversity** — Feature variance to detect mode collapse
4. **Cycle Consistency Loss** — Structural preservation (CycleGAN only)

Proxy FID is computed using features extracted from a pretrained ResNet-18 backbone:

$$\text{Proxy FID} = \sum_{i} \left( \mu_{\text{gen}}^{(i)} - \mu_{\text{real}}^{(i)} \right)^2$$

where $\mu_{\text{gen}}$ and $\mu_{\text{real}}$ are the mean feature vectors of generated and real Monet images.

## Results and Visualizations

All plots and evaluation artifacts are saved under `results/plots/`:

- RGB and luminance histogram comparisons
- Edge density (Sobel) distributions
- HSV saturation analysis
- Training loss curves with EMA smoothing
- Proxy FID and diversity trajectories
- Sample generation grids per model

## How to Run

### Kaggle (Recommended)

1. Upload the repository notebook to Kaggle or open the existing project
2. Attach the **GAN Getting Started** dataset
3. Add the **ResNet-18 pretrained weights** as a dataset input
4. Run all cells sequentially
5. Download outputs from `/kaggle/working/`

### Local Execution

```bash
pip install -r requirements.txt
python train.py --model cyclegan --epochs 20
```

CUDA will help training times.

## Generating Submissions

The final submission pipeline translates 7,000 photographs to Monet style:

```python
# Load best CycleGAN checkpoint
G_PM.load_state_dict(torch.load("cyclegan.pt")["model_state_dict"])

# Generate and save images at 256×256 resolution
for photo in photo_loader:
    fake_monet = G_PM(photo)
    # Save to images.zip for submission
```

## Notes

- All evaluation metrics are computed on generated samples using a frozen ResNet-18 feature extractor
- Early stopping prevents overfitting and texture degradation
- CycleGAN uses scheduled cycle loss weight reduction ($\lambda = 10 \rightarrow 5$) after epoch 10
- Diffusion model requires substantially longer training (100+ epochs) for meaningful outputs
- Visual inspection remains essential—proxy metrics do not fully capture perceptual quality

## Future Work

- Extended diffusion training with proper noise scheduling
- Hybrid adversarial–diffusion approaches
- Explicit perceptual loss functions (LPIPS, style loss)
- Evaluation against Kaggle's hidden MiFID metric

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593) — Zhu et al., 2017
- [DCGAN Paper](https://arxiv.org/abs/1511.06434) — Radford et al., 2015
- [DDPM Paper](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
- [Kaggle GAN Getting Started Competition](https://www.kaggle.com/competitions/gan-getting-started)
