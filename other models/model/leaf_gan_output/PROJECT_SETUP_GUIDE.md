
# Lightweight Leaf Reconstruction GAN - Setup Guide

## Google Drive Project Structure

Create the following folder structure in your Google Drive:

```
/content/drive/MyDrive/mobile_gan/
├── data/
│   ├── occluded_images/          # Place your occluded leaf images here
│   │   ├── leaf_001_occluded.jpg
│   │   ├── leaf_002_occluded.jpg
│   │   └── ...
│   └── original_images/          # Place corresponding original images here
│       ├── leaf_001_original.jpg
│       ├── leaf_002_original.jpg
│       └── ...
└── outputs/                      # This will be created automatically
    └── leaf_gan_output/
        ├── models/               # Final trained models
        ├── weights/              # Training checkpoints
        ├── samples/              # Generated sample images
        ├── metrics/              # Training metrics and plots
        ├── checkpoints/          # Full training checkpoints
        └── logs/                 # Training logs

```

## Setup Instructions

1. **Mount Google Drive** (if using Google Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Prepare your dataset**:
   - Ensure occluded and original images have matching names
   - Supported formats: .jpg, .jpeg, .png
   - Images will be automatically resized to 256x256

3. **Update paths** (if needed):
   - Modify DRIVE_BASE_PATH in the code to match your Google Drive structure

4. **Run training**:
   ```python
   train_gan()
   ```

## Expected Outputs

After training, you'll find:

- **Models**: `generator_final.pth`, `discriminator_final.pth`
- **Checkpoints**: Regular training checkpoints for resuming
- **Sample Images**: Generated images throughout training
- **Metrics**: Training curves, evaluation results, model analysis
- **Logs**: Complete training logs with timestamps

## Performance Expectations

- **Training Time**: ~100 epochs, time depends on dataset size and hardware
- **Model Size**: ~3.3 MB total
- **Memory Usage**: Optimized for limited GPU memory
- **Quality Metrics**: PSNR and SSIM calculated automatically

## Troubleshooting

1. **Path Issues**: Ensure the folder structure matches exactly
2. **Memory Issues**: Reduce BATCH_SIZE if running out of memory
3. **Dataset Issues**: Check that image pairs are properly matched
4. **Slow Training**: Consider reducing IMG_SIZE or NUM_EPOCHS for testing

## Model Architecture

### Generator (TinyUNet)
- **Type**: Lightweight U-Net with depthwise separable convolutions
- **Parameters**: ~177,822
- **Features**: Skip connections, residual blocks, edge-optimized
- **Input/Output**: RGB images (256x256)

### Discriminator (PatchGAN)
- **Type**: 70x70 PatchGAN discriminator  
- **Parameters**: ~696,193
- **Features**: Patch-based discrimination for fine details

### Loss Functions
- **Adversarial Loss**: Binary cross-entropy for GAN training
- **L1 Loss**: Pixel-wise reconstruction (weight: 100)
- **Perceptual Loss**: Feature-based similarity (weight: 10)
- **Color Consistency Loss**: Maintains color fidelity

## Usage Examples

```python
# Basic training
train_gan()

# Print model information
print_model_summary()

# Create setup guide
create_project_structure_guide()
```

## Citation

If you use this code, please cite:
```
Lightweight Leaf Reconstruction GAN for Edge Devices
Optimized U-Net architecture with depthwise separable convolutions
```
