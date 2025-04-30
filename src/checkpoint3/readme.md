## 🧠 Optional Code Submission

### 📁 Source File
- `main.cu` — contains the shared memory–optimized CUDA implementation of the adaptive thresholding kernel.

### 🧰 Build Instructions
To compile the optimized CUDA program, use the following command:
```bash
module load cuda/12.0
nvcc -O3 -o main main.cu
```

### Run Instructions (Direct)
```./main input_image.raw input_image.raw.meta```

or
<br>
```./run_me.sh input_image.raw input_image.raw.meta```

