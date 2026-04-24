# CNN-BiLSTM for Network Traffic Classification

This repository contains implementations of Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) models for classifying network traffic patterns. It includes scripts for training the model on generated structured data, creating random data, and a Flask-based GUI for model training with uploaded data.

## 🚀 Features

- **CNN-BiLSTM Hybrid Model**: Leverages the strengths of CNNs for feature extraction and BiLSTMs for sequence understanding.
- **Data Generation**: Includes utilities to generate synthetic structured network traffic-like data for training.
- **Optimized Training**: Features aggressive pooling, `torch.compile` (in GUI), and advanced optimizers for efficient training.
- **Detailed Metrics & Plotting**: Automatically saves training accuracy and loss plots.
- **GUI for Training**: A Flask web interface to upload dataset, trigger training, and monitor progress.
- **Focal Loss & Attention Layer**: Advanced techniques integrated into the GUI model for improved performance.
- **Robust Data Handling**: Includes data scaling, augmentation, and handling for skewed datasets in the GUI.

## 🏗️ Architecture

The core of this project revolves around a CNN-BiLSTM neural network. Data can be synthetically generated or provided via CSV, then fed into the model for training and classification.
```

## 💻 Tech Stack

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **Numpy**: Numerical computing
- **Pandas**: Data manipulation (for GUI)
- **Matplotlib**: Plotting and visualization
- **Flask**: Web framework (for GUI)
- **Scikit-learn**: Data utilities (for GUI)
- **Tqdm**: Progress bars

## 📁 Project Structure

```
.
├── GUI_training_model.py       # Flask-based GUI for model training and deployment
├── Random_Generated_Data.py    # Script for generating random data and training the CNN-BiLSTM
├── Structured_data_training.py # Script for generating structured data and training the CNN-BiLSTM
└── README.md                   # This README file
```

## ⚙️ Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

```bash
python --version
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JAYNILiiits/CNN-BiLSTM.git
   cd CNN-BiLSTM
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn flask tqdm
   ```
   If you have a CUDA-enabled GPU, install the appropriate PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally/). For example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Running the Scripts

#### 1. Training with Structured Generated Data

This script `Structured_data_training.py` generates synthetic time-series data with distinct patterns and trains a `CNNBiLSTM` model on it.

```bash
python Structured_data_training.py
```
Training plots and the best model checkpoint will be saved in a directory like `results_Adaptive/`.

#### 2. Training with Random Generated Data

The `Random_Generated_Data.py` script also generates data, but uses slightly different configurations and training loops compared to `Structured_data_training.py`.

```bash
python Random_Generated_Data.py
```
Training plots will be saved in a directory like `results_Random/`.

#### 3. Running the GUI-based Training

The `GUI_training_model.py` script provides a web interface to upload your own CSV data, configure parameters, and train a more complex `UltraOptimizedPacketClassifier` model.

```bash
python GUI_training_model.py
```
After running, open your web browser and navigate to `http://127.0.0.1:5000/`.

**GUI Usage:**
- **Upload CSV**: Click "Choose File" and select your dataset in CSV format.
- **Start Training**: Click the "Start Training" button. The GUI will show real-time progress.
- Results (metrics, plots, model checkpoint) will be saved in the `./training_results` directory.

## Configuration / Environment Variables

The scripts dynamically detect the available device (`cuda` or `cpu`). No explicit environment variables are typically needed.

For `GUI_training_model.py`, configuration is managed within the `CONFIG` dictionary:

```python
# Inside GUI_training_model.py
CONFIG = {
    'sequence_length': 10000,
    'num_epochs':500,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'gradient_accumulation_steps': 2,
    'dropout_rate': 0.35,
    'label_smoothing': 0.05,
    'mixed_precision': False,
    'validation_frequency': 1,
}
```
You can modify these values directly in the `GUI_training_model.py` file to adjust training parameters.

## 🤝 Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvement or bug fixes.
