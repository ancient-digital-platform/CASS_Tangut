# Tangut Character Recognition Models

This repository contains deep learning models for Tangut (Western Xia) character recognition.

## Dataset

The dataset used for training and testing these models is available at:
**DOI**: https://www.scidb.cn/anonymous/SVJaUnpl

### Dataset Description
- **Character Classes**: 6,212 Tangut characters
- **Format**: Enhanced image dataset for character recognition
- **Files Included**:
  - `generated_enhanced_6145.tar.gz` - Enhanced Tangut character image dataset
  - `character_statistics_frequency_sorted.csv` - Character frequency statistics
  - `class_map.json` - Character to class ID mapping (6,212 classes)

### Download and Extract Dataset
```bash
# Download the dataset from the DOI link above
# Extract the dataset
tar -xzf generated_enhanced_6145.tar.gz
```

## Available Models

This repository provides implementations of five state-of-the-art deep learning models for Tangut character recognition:

### 1. ResNet50
- **Directory**: `models/resnet50/`
- **Architecture**: Deep residual network with 50 layers
- **Files**:
  - `model.py` - Model architecture definition
  - `dataset.py` - Dataset loading and preprocessing
  - `train_resnet.py` - Training script
  - `test_resnet.py` - Testing/inference script

### 2. ResNet18
- **Directory**: `models/resnet18/`
- **Architecture**: Lighter residual network with 18 layers
- **Files**: Similar structure to ResNet50

### 3. DenseNet
- **Directory**: `models/densenet/`
- **Architecture**: Densely connected convolutional network
- **Files**:
  - `model.py` - DenseNet architecture
  - `dataset.py` - Dataset utilities
  - `train_densenet.py` - Training script
  - `test_densenet.py` - Testing script

### 4. EfficientNet
- **Directory**: `models/efficientnet/`
- **Architecture**: Efficient neural network with compound scaling
- **Files**:
  - `model.py` - EfficientNet implementation
  - `dataset.py` - Data loading
  - `train_efficientnet.py` - Training script
  - `test_efficientnet.py` - Testing script

### 5. ConvNeXt
- **Directory**: `models/convnext/`
- **Architecture**: Modern ConvNet architecture
- **Files**:
  - `model.py` - ConvNeXt model
  - `dataset.py` - Dataset handler
  - `train_convnext.py` - Training script
  - `test_convnext.py` - Testing script

## Quick Start

### Prerequisites
```bash
pip install torch torchvision numpy scikit-learn pillow
```

### Training a Model
Each model directory contains a training script. Example with ResNet50:

```bash
cd models/resnet50
python train_resnet.py
```

You may need to modify the data paths in the training script to point to your downloaded dataset location.

### Testing/Inference
After training, use the test script to evaluate the model:

```bash
cd models/resnet50
python test_resnet.py
```

### Model Structure
Each model implementation includes:
- **model.py**: Defines the neural network architecture
- **dataset.py**: Handles data loading, preprocessing, and augmentation
- **train_*.py**: Complete training pipeline with metrics (accuracy, precision, recall, F1-score, mAP, top-k accuracy)
- **test_*.py**: Testing and inference utilities

## Evaluation Metrics

The training scripts calculate the following metrics:
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-class and macro-averaged metrics
- **Top-k Accuracy**: Top-5 prediction accuracy


## Dataset Citation

If you use this dataset in your research, please cite:
```
Dataset DOI: https://www.scidb.cn/anonymous/SVJaUnpl
Dataset Name: Enhanced Tangut Character Recognition Dataset
Total Classes: 6,212 Tangut Characters
```

## License

Please refer to the dataset source for licensing information.

## Contact

For questions about the dataset or models, please refer to the dataset DOI page for contact information.
# CASS_Tangut
