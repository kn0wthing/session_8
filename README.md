# CIFAR10 Custom Model Training

This repository contains a custom CNN model implementation for CIFAR10 dataset with specific architectural constraints including depthwise separable convolution, dilated convolution, and no max pooling.

## Features

- Custom CNN architecture with:
  - Depthwise Separable Convolution
  - Dilated Convolution
  - No MaxPooling (using strided convolutions)
  - Global Average Pooling
- Albumentations for data augmentation
- Less than 200k parameters
- Modular code structure

## Setup

1. Create and activate virtual environment:

```bash
python -m venv cifar10_env
source cifar10_env/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python train.py
```

The best model will be saved as 'best_model.pth'.

## Testing

To run tests:

```bash
PYTHONPATH=$PYTHONPATH:. pytest tests/
```

Tests include:
- Model parameter count verification (<200k)
- Model architecture validation
- Input/Output shape verification
- Data augmentation verification

## Model Architecture

Block 1:
- Regular Convolutions (3x3)
- Channel reduction with 1x1 convolution

Block 2:
- Regular Convolution (3x3)
- Depthwise Separable Convolution
- Channel reduction with 1x1 convolution

Block 3:
- Regular Convolutions (3x3)
- Dilated Convolution with stride=2

Block 4:
- Regular Convolutions (3x3)
- Global Average Pooling
- 1x1 Convolution for classification

## Requirements

See requirements.txt for detailed dependencies.
