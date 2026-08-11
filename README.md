# Evaluating XAI Faithfulness for GradCAM, Integrated Gradients, and LIME

This project evaluates explanation fidelity for GradCAM, Integrated Gradients, and LIME. A perturbation pipeline is used to perform feature deletion and feature insertion to compute total mean AUC scores. These scores are measures of overall explanation fidelity for each of the XAI methods. The explanations are collected for a ResNet-18 model where inputs are a subset of correctly predicted images from the CIFAR-10 test dataset. The project includes Python files for training the model on CIFAR-10, collecting explanations for each XAI method, and executing the perturbation pipeline to perform feature deletion/insertion. An additional Python file is provided to compute total mean AUC scores along with various other metrics.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Hardware Requirements](#hardware-requirements)
- [Training the ResNet-18 Model](#training-the-resnet-18-model)
- [Collecting the Explanations](#collecting-the-explanations)
- [Perturbation Pipeline](#perturbation-pipeline)
- [Metrics](#metrics)

---

## Environment Setup

1. **Python Version:**  
This project requires Python 3.14 or higher.

2. **Dependencies:**  
Either uv or pip can be used to install the required Python packages:

```bash
uv sync
```
Or

```bash
pip install -r requirements.txt
```

# Key Libraries

- **PyTorch**
- **grad-cam**
- **captum**
- **lime**
- **numpy**
- **pandas**
- **matplotlib**
- **tqdm**


## Hardware Requirements

A CUDA-enabled GPU with at least 8 GB VRAM is recommended for training. The code automatically detects GPU availability.

---

## Training the ResNet-18 Model

#### Step 0: Configure Training Parameters

The training configuration is defined at the top of the `model_training.py` file. In particular, the following parameters can be updated:

- batch size
- learning rate
- number of output classes
- number of epochs
- The path where the saved model weights are stored.

#### Step 1: Run the training

Execute the `model_training.py` file, which will train a ResNet-18 model, store the weights, and save the indices to correctly predicted CIFAR-10 test images.

```python
python model_training.py
```

---

## Collecting the Explanations

### Duration Considerations

For statistical analysis testing, a total of 5 explanation collection iterations are performed. This can be adjusted by modifying the **NUMBER_OF_ITERATIONS** variable at the top of `explanations.py`. Additionally, note that collecting LIME explanations can be slow since each explanation is computed sequentially. Thus, even with setting **NUMBER_OF_ITERATIONS** to 1, collecting all explanations can still take several hours.

### Step 2: Generate explanations and store feature importance maps

Execute the `explanations.py` file, which will collect explanations for each XAI method and store feature importance maps to a local disk drive.

```python
python explanations.py
```

---

## Perturbation Pipeline

Execute the `perturbation_pipeline.py` file to run the perturbation pipeline. This will perform feature deletion/insertion while saving AUC scores for each correctly predicted test image sample. Additionally, model confidences for each sample and each perturbation ratio are also saved.

```python
python perturbation_pipeline.py
```

---

## Metrics

Execute the `metrics.py` file to compute total mean AUC scores for feature deletion/insertion along with various other metrics for each XAI method like average explanation latency, explanation throughput, and peak VRAM usage. Additionally, the ANOVA and Tukey HSD tests are executed on the AUC scores to determine their statistical significance.

```python
python metrics.py
```

---

## Summary of Steps

- **Step 1: Run Training**  
  Execute `model_traning.py` to train a ResNet-18 model on CIFAR-10 from scratch.

- **Step 2: Collect XAI explanations**
  Execute `explanations.py` to generate explanations for each XAI method and save the feature importance maps to a disk drive.

- **Step 3: Run the Perturbation Pipeline**
  Execute `perturbation_pipeline.py` to perform feature deletion/insertion guided by the collected feature importance maps. Additionally, this will compute and save model confidence AUC scores.

- **Step 4: Metrics**
  Execute `metrics.py` to total mean AUC scores for each XAI method along with various other metrics.

Happy Explaining!

---

## Additional Python Files

* **Visualizations.py** - Implements various helper functions for displaying/saving qualitative examples used for the project report.
* **GradCamXai.py** - Generates explanations, visualizations, and feature importance mappings for GradCAM.
* **IntegratedGradientsXai.py** - Generates explanations, visualizations, and feature importance mappings for Integrated Gradients.
* **LimeXai.py** - Generates explanations, visualizations, and feature importance mappings for LIME.
* **XaiMethodBase.py** - Base class for a generic XAI method. I.e., the above three files inherit from this class.