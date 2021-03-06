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

typedef struct intfield1
{
    int *m;
    int size[1];
    int is_device_field;
} intfield1;

void memcpy_field_intfield1(intfield1 dst, intfield1 src)
{
    if (dst.is_device_field == 0 && src.is_device_field == 0) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0], cudaMemcpyHostToHost);
    }
    if (dst.is_device_field == 1 && src.is_device_field == 0) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0], cudaMemcpyHostToDevice);
    }
    if (dst.is_device_field == 0 && src.is_device_field == 1) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0], cudaMemcpyDeviceToHost);
    }
    if (dst.is_device_field == 1 && src.is_device_field == 1) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0], cudaMemcpyDeviceToDevice);
    }
}

int size_intfield1(intfield1 field, int index)
{
    return field.size[index];
}

intfield1 alloc_host_field_intfield1(int size_0)
{
    intfield1 field;
    field.m = (int*)malloc((sizeof(*field.m))*size_0);
    field.size[0] = size_0;
    field.is_device_field = 0;
    return field;
}

void free_host_field_intfield1(intfield1 field)
{
    free(field.m);
}

intfield1 alloc_device_field_intfield1(int size_0)
{
    intfield1 field;
    cudaMalloc((void**)&field.m, (sizeof(*field.m))*size_0);
    field.size[0] = size_0;
    field.is_device_field = 1;
    return field;
}

void free_device_field_intfield1(intfield1 field)
{
    cudaFree(field.m);
}

typedef struct intmat1
{
    int m[1];
} intmat1;

int printf(const char *fmt, ...); /* TODO: Remove */

typedef intfield1 Field; /* One-dimensional integer field type */
__global__ void kernel_0(intfield1 a, intfield1 b)
{
    if (threadIdx.x + blockIdx.x*blockDim.x >= a.size[0]) {
        return;
    }
    intmat1 id;
    id.m[1*0] = (threadIdx.x + blockIdx.x*blockDim.x) % a.size[0]/1;
    a.m[1*id.m[1*0]] += b.m[1*id.m[1*0]];
}


int main()
{

    int N = 5;

    intfield1 a_data = alloc_host_field_intfield1(N);

    intfield1 b_data = alloc_host_field_intfield1(N);
    a_data.m[1*0] = 1;
    a_data.m[1*1] = 2;
    a_data.m[1*2] = 3;
    a_data.m[1*3] = 4;
    a_data.m[1*4] = 5;
    b_data.m[1*0] = 10;
    b_data.m[1*1] = 20;
    b_data.m[1*2] = 30;
    b_data.m[1*3] = 40;
    b_data.m[1*4] = 50;

    intfield1 a = alloc_device_field_intfield1(N);

    intfield1 b = alloc_device_field_intfield1(N);
    memcpy_field_intfield1(a, a_data);
    memcpy_field_intfield1(b, b_data);

    {
        dim3 dim_grid(a.size[0]/128 + 1, 1, 1);
        dim3 dim_block(128, 1, 1);
        kernel_0<<<dim_grid, dim_block>>>(a, b);
    }
    memcpy_field_intfield1(a_data, a);

    for (int i = 0; i < N; ++i) {
        printf("%i ", a_data.m[1*i]);
    }
    free_host_field_intfield1(a_data);
    free_host_field_intfield1(b_data);
    free_device_field_intfield1(a);
    free_device_field_intfield1(b);

    return 0;
}
