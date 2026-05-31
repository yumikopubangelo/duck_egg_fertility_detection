# U-Net Architecture Documentation

## Overview

U-Net is a convolutional neural network architecture designed for biomedical image segmentation. In this project, it's used to segment duck eggs in candling images, identifying the embryo region for fertility assessment.

## Architecture Design

### Network Structure

The U-Net follows an encoder-decoder architecture with skip connections:

```
Encoder (Contracting Path)    →    Bottleneck    →    Decoder (Expansive Path)
      ↓                               ↓                       ↓
Input → [Conv] → [Pool] → ... → [Bottleneck] → ... → [UpConv] → [Conv] → Output
      ↓                                                   ↑
      └───────────────── Skip Connections ───────────────┘
```

### Key Characteristics

1. **Symmetric Design**: Encoder and decoder have mirrored structure
2. **Skip Connections**: Preserve spatial information across resolution levels
3. **No Fully Connected Layers**: Fully convolutional architecture
4. **Deep Supervision**: Multiple levels of feature abstraction

## Layer Details

### 1. Input Layer

- **Dimensions**: (batch_size, n_channels, height, width)
- **Default**: (N, 3, 256, 256) for RGB images
- **Preprocessing**: Normalized to [0, 1] range

### 2. Encoder (Contracting Path)

#### DoubleConv Block
```
Conv2d(in_channels, out_channels, kernel=3, padding=1)
  → BatchNorm2d
  → ReLU
  → Conv2d(out_channels, out_channels, kernel=3, padding=1)
  → BatchNorm2d
  → ReLU
```

Each DoubleConv block maintains spatial dimensions while increasing feature depth.

#### Down Block
```
MaxPool2d(kernel=2, stride=2)  # Halves spatial dimensions
  → DoubleConv
```

Reduces resolution by 2× while doubling feature channels.

**Encoder Levels:**

| Level | Input Channels | Output Channels | Spatial Size |
|-------|---------------|-----------------|--------------|
| inc   | 3 | 64 | 256×256 |
| down1 | 64 | 128 | 128×128 |
| down2 | 128 | 256 | 64×64 |
| down3 | 256 | 512 | 32×32 |
| down4 | 512 | 512 | 16×16 |

### 3. Bottleneck

```
DoubleConv(512, 512)
  → Dropout (optional)
```

- Operates at lowest spatial resolution
- Highest feature depth (512 channels)
- Dropout for regularization (p=0.0-0.5)

### 4. Decoder (Expansive Path)

#### Up Block

Two variants:

**Bilinear Mode:**
```
Upsample(scale_factor=2, mode='bilinear')
  → Concatenate with skip connection
  → DoubleConv
```

**Transposed Conv Mode:**
```
ConvTranspose2d(in_channels, in_channels//2, kernel=2, stride=2)
  → Concatenate with skip connection
  → DoubleConv
```

**Decoder Levels:**

| Level | Input Channels | Output Channels | Spatial Size |
|-------|---------------|-----------------|--------------|
| up1   | 1024 | 256 | 32×32 |
| up2   | 512 | 128 | 64×64 |
| up3   | 256 | 64 | 128×128 |
| up4   | 128 | 64 | 256×256 |

### 5. Output Layer

```
Conv2d(64, n_classes, kernel=1)
```

- 1×1 convolution reduces channels to number of classes
- No activation (logits output)
- Sigmoid applied during inference for binary segmentation

## Skip Connections

### Purpose

1. **Preserve Spatial Details**: High-resolution features from encoder
2. **Mitigate Vanishing Gradients**: Direct gradient flow
3. **Enable Precise Localization**: Combines semantic and spatial information

### Implementation

```python
x = torch.cat([x2, x1], dim=1)  # Concatenate encoder and decoder features
```

Where:
- `x2`: Encoder features (higher resolution, fewer channels)
- `x1`: Decoder features (lower resolution, more channels)

## Mathematical Formulation

### Forward Pass

Let:
- X = input image
- f_i = encoder block i
- g_j = decoder block j
- s_k = skip connection k
- b = bottleneck

```
x1 = f1(X)
x2 = f2(x1)
x3 = f3(x2)
x4 = f4(x3)
x5 = f5(x4)          # Bottleneck

y1 = g1(x5, s4)      # s4 = x4
y2 = g2(y1, s3)      # s3 = x3
y3 = g3(y2, s2)      # s2 = x2
y4 = g4(y3, s1)      # s1 = x1

Output = Conv1x1(y4)
```

### Skip Connection Dimensions

For concatenation to work, encoder and decoder features must match spatially:

```
H_encoder = H_decoder
W_encoder = W_decoder
```

Padding is applied when dimensions differ:

```python
diffY = x2.size()[2] - x1.size()[2]
diffX = x2.size()[3] - x1.size()[3]
x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
```

## Loss Functions

### Binary Cross-Entropy (BCE)

```python
loss = -[y*log(p) + (1-y)*log(1-p)]
```

Where:
- y = ground truth (0 or 1)
- p = predicted probability (after sigmoid)

### Dice Loss

```python
Dice = 2*|X∩Y| / (|X| + |Y|)
loss = 1 - Dice
```

Better for imbalanced segmentation tasks.

### Combined Loss

```python
loss = α*BCE + β*Dice
```

Where α, β are weighting hyperparameters.

## Model Variants

### 1. Standard U-Net

- Full capacity model
- ~31M parameters
- Best accuracy
- Higher memory usage

### 2. Lightweight U-Net

- Reduced channels (32→16→32→64→64)
- ~7.8M parameters
- Faster inference
- Lower memory footprint

### 3. U-Net with Dropout

- Dropout2d(p=0.2-0.5) after bottleneck
- Reduces overfitting
- Better generalization

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 16 | Samples per gradient update |
| Learning Rate | 1e-3 | Adam optimizer |
| Epochs | 100 | Training iterations |
| Optimizer | Adam | β1=0.9, β2=0.999 |
| Loss | Dice + BCE | Combined loss |
| Augmentation | Yes | Rotation, flip, brightness |

### Data Augmentation

```python
transforms = [
    RandomRotation(15),
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.5),
    ColorJitter(brightness=0.2),
]
```

## Evaluation Metrics

### Segmentation Metrics

1. **IoU (Intersection over Union)**
   ```
   IoU = |Prediction ∩ GroundTruth| / |Prediction ∪ GroundTruth|
   ```
   Range: [0, 1], higher is better

2. **Dice Coefficient**
   ```
   Dice = 2*|Prediction ∩ GroundTruth| / (|Prediction| + |GroundTruth|)
   ```
   Range: [0, 1], higher is better

3. **Pixel Accuracy**
   ```
   Accuracy = TP + TN / (TP + TN + FP + FN)
   ```

4. **Precision & Recall**
   ```
   Precision = TP / (TP + FP)
   Recall = TP / (TP + FN)
   ```

## Implementation Details

### Class: UNet

**Constructor Parameters:**

- `n_channels`: Input channels (1=grayscale, 3=RGB)
- `n_classes`: Output classes (1=binary)
- `bilinear`: Use bilinear upsampling (True) or transposed conv (False)
- `dropout_rate`: Dropout probability (0.0-0.5)

**Key Methods:**

- `forward(x)`: Forward pass
- `count_parameters()`: Count trainable parameters
- `get_model_summary()`: Print architecture summary

### Utility Functions

- `create_unet()`: Factory function for model creation
- `create_unet_for_eggs()`: Pre-configured for egg segmentation
- `save_model()`: Save checkpoint
- `load_model()`: Load checkpoint
- `calculate_iou()`: Compute IoU metric
- `calculate_dice_coefficient()`: Compute Dice metric

## Inference Pipeline

```python
# 1. Load image
image = cv2.imread('egg.jpg')
image = cv2.resize(image, (256, 256))

# 2. Preprocess
image = image / 255.0  # Normalize
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

# 3. Load model
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load('unet_best.pth'))
model.eval()

# 4. Predict
with torch.no_grad():
    output = model(image)
    mask = torch.sigmoid(output) > 0.5

# 5. Post-process
mask = mask.squeeze().numpy().astype(np.uint8) * 255
```

## Performance Characteristics

### Computational Requirements

| Metric | Standard U-Net | Lightweight U-Net |
|--------|---------------|------------------|
| Parameters | ~31M | ~7.8M |
| FLOPs | ~30B | ~7.5B |
| Memory (inference) | ~2GB | ~500MB |
| Inference time (GPU) | ~50ms | ~20ms |
| Inference time (CPU) | ~500ms | ~200ms |

### Accuracy Benchmarks

| Dataset | IoU | Dice | Accuracy |
|---------|-----|------|----------|
| Duck Eggs (val) | 0.85 | 0.91 | 0.97 |
| Duck Eggs (test) | 0.82 | 0.89 | 0.96 |

## Advantages

1. **Precise Segmentation**: Skip connections enable pixel-level accuracy
2. **Data Efficient**: Works well with limited training data
3. **Flexible**: Adaptable to various input sizes and modalities
4. **Robust**: Handles variations in lighting, orientation
5. **Interpretable**: Feature maps show learned representations

## Limitations

1. **Memory Intensive**: Large models require significant GPU memory
2. **Training Time**: Deep architecture requires many epochs
3. **Boundary Artifacts**: May produce jagged segmentations
4. **Fixed Receptive Field**: Limited context at high resolutions

## Extensions & Improvements

### Potential Enhancements

1. **Attention U-Net**: Add attention gates to focus on relevant features
2. **Residual U-Net**: Incorporate residual connections
3. **Nested U-Net**: U-Net++ architecture for multi-scale features
4. **Transformer U-Net**: Replace convolutions with self-attention
5. **Semi-supervised**: Use unlabeled data with consistency training

### Post-processing

1. **Morphological Operations**: Clean up masks
   ```python
   kernel = np.ones((5,5), np.uint8)
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   ```

2. **Contour Analysis**: Filter by size, shape
3. **CRF**: Conditional Random Fields for boundary refinement

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." arXiv:1505.04597
2. Çiçek, Ö., et al. (2016). "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." MICCAI
3. Zhou, Z., et al. (2018). "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." arXiv:1807.10165

---

*Last updated: 2026-04-27*
