.PHONY: clean

main: mean_adpt_cuda.cu
	nvcc -std=c++17 -I. -o main mean_adpt_cuda.cu
clean:
	rm -f main
