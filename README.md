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
```bash
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
│     ├── metrices/  
├── scripts/
│   ├── jpg_to_raw.py
│   ├── raw_to_jpg.py
├── inputs/
│   └── sample_image.jpg
├── research/
├── requirements.txt
├── README.md 
└── submission.pbs
```
## 🛠️ Build Instructions
Navigate to the final optimized checkpoint3:
### Baseline implementation
```bash 
cd src/checkpoint3/
```
To generate JPG to RAW (Optional, already saved into the  inputs directory)
```bash
python ../scripts/jpg_to_raw.py image_path ----output
```

Build and run the code using the provided shell file:
```bash
./run_me.sh img.raw img.raw.meta
```
Metrics will be saved in the metrics directory.

### Optimized Implementation (APOD)
Build and run the code using the provided shell file:
```bash
./run_me.sh img.raw img.raw.meta
```
Metrics will be saved in the metrics directory.

To generate RAW to JPG 
```bash
python ../scripts/raw_to_jpg.py raw_image_path meta_path
```

