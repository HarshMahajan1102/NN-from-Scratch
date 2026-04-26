# Neural Network from Scratch

## 📌 About this Project

This project was built while learning neural networks from scratch in December 2025.
Instead of only studying theory, each concept was implemented step-by-step using NumPy to gain a deeper understanding of how neural networks actually work internally.

The repository was initially created as a placeholder and has now been properly structured to reflect the complete learning and implementation journey.

---

## What This Project Covers

* Manual forward propagation
* Batch processing using NumPy
* Dense (Fully Connected) layer implementation
* Activation functions:

  * ReLU
  * Softmax
* Loss function:

  * Categorical Cross Entropy (CCE)
* Multi-layer neural network pipeline
* Comparison with TensorFlow implementation

---

## Project Structure

```
NN-from-Scratch/
│
├── src/
│   ├── layers/
│   │   └── dense.py
│   ├── activations/
│   │   ├── relu.py
│   │   └── softmax.py
│   ├── losses/
│   │   └── cce.py
│
├── examples/
│   ├── full_pipeline.py
│   └── tensorflow_compare.py
│
├── experiments/
│   ├── initial_nn.py
│   └── batching_raw.py
│
├── notebooks/
│   └── nn.ipynb
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

* **Python**
* **NumPy** → core computations and vectorization
* **TensorFlow / Keras** → used for comparison with real-world implementation

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run full pipeline

```
python examples/full_pipeline.py
```

---

## Learning Highlights

* Implemented neural network components without using ML frameworks
* Understood how matrix multiplication drives forward propagation
* Explored numerical stability in Softmax (max-shift trick)
* Built modular structure (layers, activations, loss)
* Compared custom implementation with TensorFlow model

---

## Upcoming Work

The following features will be implemented and added soon:

* Backpropagation
* Gradient descent optimization
* Training loop
* Model evaluation metrics
* Support for multiple hidden layers
