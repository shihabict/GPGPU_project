# Optimizing Adaptive Thresholding Using CUDA for Real-Time Image Processing

## 📝 Project Overview

This project implements a GPU-accelerated **adaptive thresholding** technique for real-time image processing using CUDA. The goal is to optimize per-pixel binary thresholding using shared memory and parallelism on NVIDIA GPUs. This method is well-suited for applications in low-latency surveillance and vision systems.

---

## ⚙️ Dependencies

- CUDA Toolkit: **12.6**
- Python (for pre/post-processing scripts): **>=3.6**
  - `numpy`
  - `Pillow` (PIL)

Tested on ASA and RCHAU GPU servers.

---

## 📁 Repository Structure

GPGPU_Project/
├── src/
│   ├── checkpoint2/
│     ├── host_side_implementation.py
│     ├── main.cu
│     └── Makefile 
│   ├── checkpoint3/
│     ├── main.cu 
│     ├── Makefile
│     ├── run_me.sh
│     └── submission.pbs
│     ├── optimization/
│       ├── main.cu 
│       ├── Makefile
│       ├── run_me.sh
│       └── submission.pbs   
├── scripts/
│   ├── jpg_to_raw.py
│   ├── raw_to_jpg.py
├── inputs/
│   └── sample_image.jpg
├── research/
├── requirements.txt
├── README.md 
└── submission.pbs
