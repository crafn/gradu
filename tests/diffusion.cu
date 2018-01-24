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

typedef struct floatfield2
{
    float *m;
    int size[2];
    int is_device_field;
} floatfield2;

floatfield2 alloc_field_floatfield2(int size_0, int size_1)
{
    floatfield2 field;
    field.m = (float*)malloc((sizeof(*field.m))*size_0*size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
    field.is_device_field = 0;
    return field;
}

floatfield2 alloc_device_field_floatfield2(int size_0, int size_1)
{
    floatfield2 field;
    cudaMalloc((void**)&field.m, (sizeof(*field.m))*size_0*size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
    field.is_device_field = 1;
    return field;
}

void free_field_floatfield2(floatfield2 field)
{
    free(field.m);
}

void free_device_field_floatfield2(floatfield2 field)
{
    cudaFree(field.m);
}

void memcpy_field_floatfield2(floatfield2 dst, floatfield2 src)
{
    if (dst.is_device_field == 0 && src.is_device_field == 0) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1], cudaMemcpyHostToHost);
    }
    if (dst.is_device_field == 1 && src.is_device_field == 0) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1], cudaMemcpyHostToDevice);
    }
    if (dst.is_device_field == 0 && src.is_device_field == 1) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1], cudaMemcpyDeviceToHost);
    }
    if (dst.is_device_field == 1 && src.is_device_field == 1) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1], cudaMemcpyDeviceToDevice);
    }
}

int size_floatfield2(floatfield2 field, int index)
{
    return field.size[index];
}

typedef struct intmat2
{
    int m[2];
} intmat2;

int printf(const char *fmt, ...);

typedef floatfield2 Field;
__global__ void kernel_0(floatfield2 *cuda_output, floatfield2 input, int size_x, int size_y)
{
    intmat2 id;
    id.m[1*0] = (threadIdx.x + blockIdx.x*blockDim.x) % (*cuda_output).size[0]/1;
    id.m[1*1] = (threadIdx.x + blockIdx.x*blockDim.x) % ((*cuda_output).size[0]*(*cuda_output).size[1])/(*cuda_output).size[0];

    int x = id.m[1*0];

    int y = id.m[1*1];

    int nx = (x + 1) % size_x;

    int ny = (y + 1) % size_y;

    int px = (x - 1 + size_x) % size_x;

    int py = (y - 1 + size_y) % size_y;
    (*cuda_output).m[(*cuda_output).size[1]*x + 1*y] = input.m[input.size[1]*x + 1*y] + input.m[input.size[1]*nx + 1*y] + input.m[input.size[1]*px + 1*y] + input.m[input.size[1]*x + 1*ny] + input.m[input.size[1]*x + 1*py];
    (*cuda_output).m[(*cuda_output).size[1]*x + 1*y] /= 5.000000;
}


int main(int argc, char **argv)
{

    int size_x = 20;

    int size_y = 20;

    Field host_field = alloc_field_floatfield2(size_x, size_y);

    Field device_field_1 = alloc_device_field_floatfield2(size_x, size_y);

    Field device_field_2 = alloc_device_field_floatfield2(size_x, size_y);

    /* Init field */

    {

        for (int x = 0; x < size_x; ++x) {

            for (int y = 0; y < size_y; ++y) {
                host_field.m[host_field.size[1]*x + 1*y] = 0;
            }
        }
        host_field.m[host_field.size[1]*(size_x/2) + 1*(size_y/2)] = 1000;
    }
    memcpy_field_floatfield2(device_field_1, host_field);

    for (int i = 0; i < 5; ++i) {

        Field *input = &device_field_1;

        Field *output = &device_field_2;

        /* Swap */

        if (i % 2 == 1) {

            Field *tmp = output;
            output = input;
            input = tmp;
        }

        /* Diffusion! */

        {
            floatfield2 *cuda_output = (floatfield2*)cuda_upload_var(&output, sizeof(output));
            dim3 dim_grid(100, 1, 1);
            dim3 dim_block((*output).size[0]*(*output).size[1]/100, 1, 1);
            kernel_0<<<dim_grid, dim_block>>>(cuda_output, *input, size_x, size_y);
            cuda_download_var(cuda_output, &output, sizeof(output));
        }
        memcpy_field_floatfield2(host_field, *output);

        /* Print current state */

        for (int y = 0; y < size_y; ++y) {

            for (int x = 0; x < size_x; ++x) {

                const char *ch = " ";

                if (host_field.m[host_field.size[1]*x + 1*y] > 0.500000) {
                    ch = "#";
                } else if (host_field.m[host_field.size[1]*x + 1*y] > 0.100000) {
                    ch = ".";
                }
                printf("%s", ch);
            }
            printf("\n");
        }
        printf("\n");
    }
    free_field_floatfield2(host_field);
    free_device_field_floatfield2(device_field_1);
    free_device_field_floatfield2(device_field_2);

    return 0;
}
