#include <stdio.h>
#include <stdlib.h>

typedef struct floatfield2
{
    float *m;
    int size[2];
} floatfield2;

floatfield2 alloc_field_floatfield2(int size_0, int size_1)
{
    floatfield2 field;
    field.m = (float *)malloc((sizeof(*field.m)) * size_0 * size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
    return field;
}

floatfield2 alloc_device_field_floatfield2(int size_0, int size_1)
{
    floatfield2 field;
    cudaMalloc((void **)field.m, (sizeof(*field.m)) * size_0 * size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
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

void memcpy_field_floatfield2(floatfield2 dst, floatfield2 src);

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
__global__ void TODO_proper_kernel_name(floatfield2 output, floatfield2 input, int size_x, int size_y)
{
    intmat2 id;
    id.m[1 * 1] = threadIdx.y;
    id.m[1 * 0] = threadIdx.x;
    int x = id.m[1 * 0];
    int y = id.m[1 * 1];
    int nx = (x + 1) % size_x;
    int ny = (y + 1) % size_y;
    int px = (x - 1 + size_x) % size_x;
    int py = (y - 1 + size_y) % size_y;
    output.m[1 * x + output.size[0] * y] = input.m[1 * x + input.size[0] * y] + input.m[1 * nx + input.size[0] * y] + input.m[1 * px + input.size[0] * y] + input.m[1 * x + input.size[0] * ny] + input.m[1 * x + input.size[0] * py];
    output.m[1 * x + output.size[0] * y] /= 5.000000;
}

int main(int argc, char **argv)
{
    int size_x = 20;
    int size_y = 20;
    Field host_field = alloc_field_floatfield2(size_x, size_y);
    Field device_field_1 = alloc_device_field_floatfield2(size_x, size_y);
    Field device_field_2 = alloc_device_field_floatfield2(size_x, size_y);
    {
        for (int x = 0; x < size_x; ++x) {
            for (int y = 0; y < size_y; ++y) {
                host_field.m[1 * x + host_field.size[0] * y] = 0;
            }
        }
        host_field.m[1 * size_x / 2 + host_field.size[0] * size_y / 2] = 1000;
    }
    for (int i = 0; i < 20; ++i) {
        Field *input = &device_field_1;
        Field *output = &device_field_2;

        /* Swap */
        if (i % 2 == 1) {
            Field *tmp = output;
            output = input;
            input = tmp;
        }
        {
            dim3 dim_grid(1, 1, 1);
            dim3 dim_block(-1, -1, 1);
            TODO_proper_kernel_name<<<dim_grid, dim_block>>>(output, input, size_x, size_y);
        }
        memcpy_field_floatfield2(host_field, *output);
        for (int y = 0; y < size_y; ++y) {
            for (int x = 0; x < size_x; ++x) {
                char *ch = " ";
                if (host_field.m[1 * x + host_field.size[0] * y] > 5.000000) {
                    ch = "#";
                } else if (host_field.m[1 * x + host_field.size[0] * y] > 1.000000) {
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

