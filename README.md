

## 📂 Project Structure

This repository contains two complementary modules that evaluate BQQ from different perspectives:

### **`matrix_compression/`**

Focuses on **generic matrix datasets**.
It studies the trade-off between:

* model (compressed) size, and
* reconstruction error

This module is useful for understanding BQQ as a **matrix compression method**, independent of downstream neural networks.

---

### **`nn_compression/`**

Focuses on **neural network quantization**.
It evaluates how BQQ affects:

* model size,
* task performance

This module connects BQQ to practical deep-learning use cases.

