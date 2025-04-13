# Simple Neural Network from Scratch 🧠⚙️

A basic neural network built using pure NumPy to understand how feedforward, backpropagation, and gradient descent work under the hood. No libraries like TensorFlow or PyTorch — just math and code.

## 🧪 What's Implemented?

- A 2-layer neural network:
  - **2 input features**
  - **1 hidden layer** with 2 neurons
  - **1 output neuron** with sigmoid activation

- **Forward pass**
- **Manual backpropagation**
- **Stochastic Gradient Descent (SGD)**
- **Training loop over epochs**
- **MSE Loss function**

---

## 🧠 What I Learned

### 1. **Sigmoid Function & Derivative**

\[
\sigma(x) = \frac{1}{1 + e^{-x}} \\
\sigma'(x) = \sigma(x)(1 - \sigma(x))
\]

Used to add non-linearity and squash outputs between 0 and 1.

---

### 2. **Loss Function: Mean Squared Error (MSE)**

\[
L = \frac{1}{n} \sum_{i=1}^n (y_{\text{true}} - y_{\text{pred}})^2
\]

Shows how far off predictions are from actual values.

---

### 3. **Gradient Descent**

We update weights to reduce loss:

\[
w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}
\]

Where:
- \( w \) is a weight
- \( \eta \) is the learning rate
- \( \frac{\partial L}{\partial w} \) is the gradient

---

### 4. **Backpropagation Logic**

Using the chain rule, we compute how each weight contributes to the final loss:

\[
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y_{\text{pred}}} \cdot \frac{\partial y_{\text{pred}}}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}
\]

We repeat this for all weights and biases.

---

## 🧪 Dataset Used

```python
data = np.array([
  [-2, -1],  # Alice → 1
  [25, 6],   # Bob   → 0
  [17, 4],   # Charlie → 0
  [-15, -6], # Diana → 1
])
