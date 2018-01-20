#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memcpy */
#include <math.h>
#include <stdint.h>

void *cuda_upload_var(void *host_var, int size)
{
	void *cuda_var;
	cudaMalloc(&cuda_var, 4);
	cudaMemcpy(cuda_var, host_var, size, cudaMemcpyHostToDevice);
	return cuda_var;
}
void cuda_download_var(void *cuda_var, void *host_var, int size)
{
	cudaMemcpy(host_var, cuda_var, size, cudaMemcpyDeviceToHost);
	cudaFree(cuda_var);
}

int printf(const char *fmt, ...);

int main(int argc, char **argv)
{
    printf("Hello World!\n");

    return 0;
}
