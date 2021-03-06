#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memcpy */
#include <math.h>
#include <stdint.h>

typedef struct floatfield2
{
    float *m;
    int size[2];
    int is_device_field;
} floatfield2;

void memcpy_field_floatfield2(floatfield2 dst, floatfield2 src)
{
    memcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1]);
}

int size_floatfield2(floatfield2 field, int index)
{
    return field.size[index];
}

floatfield2 alloc_host_field_floatfield2(int size_0, int size_1)
{
    floatfield2 field;
    field.m = (float*)malloc((sizeof(*field.m))*size_0*size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
    field.is_device_field = 0;
    return field;
}

void free_host_field_floatfield2(floatfield2 field)
{
    free(field.m);
}

floatfield2 alloc_device_field_floatfield2(int size_0, int size_1)
{
    floatfield2 field;
    field.m = (float*)malloc((sizeof(*field.m))*size_0*size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
    field.is_device_field = 0;
    return field;
}

void free_device_field_floatfield2(floatfield2 field)
{
    free(field.m);
}

typedef struct intmat2
{
    int m[2];
} intmat2;

int printf(const char *fmt, ...);

typedef floatfield2 Field;

int main(int argc, char **argv)
{
    int i;

    int size_x = 20;

    int size_y = 20;

    floatfield2 host_field = alloc_host_field_floatfield2(size_x, size_y);

    floatfield2 device_field_1 = alloc_device_field_floatfield2(size_x, size_y);

    floatfield2 device_field_2 = alloc_device_field_floatfield2(size_x, size_y);

    /* Init field */

    {
        int x;

        for (x = 0; x < size_x; ++x) {
            int y;

            for (y = 0; y < size_y; ++y) {
                host_field.m[host_field.size[1]*x + 1*y] = 0;
            }
        }
        host_field.m[host_field.size[1]*(size_x/2) + 1*(size_y/2)] = 1000;
    }
    memcpy_field_floatfield2(device_field_1, host_field);

    for (i = 0; i < 5; ++i) {
        int y;

        floatfield2 *input = &device_field_1;

        floatfield2 *output = &device_field_2;

        /* Swap */

        if (i % 2 == 1) {

            floatfield2 *tmp = output;
            output = input;
            input = tmp;
        }

        /* Diffusion! */

        {
            int id_0;
            for (id_0 = 0; id_0 < size_floatfield2(*output, 0); ++id_0) {
                int id_1;
                for (id_1 = 0; id_1 < size_floatfield2(*output, 1); ++id_1) {
                    intmat2 id;

                    int x;

                    int y;

                    int nx;

                    int ny;

                    int px;

                    int py;
                    id.m[1*1] = id_1;
                    id.m[1*0] = id_0;
                    x = id.m[1*0];
                    y = id.m[1*1];
                    nx = (x + 1) % size_x;
                    ny = (y + 1) % size_y;
                    px = (x - 1 + size_x) % size_x;
                    py = (y - 1 + size_y) % size_y;
                    output->m[output->size[1]*x + 1*y] = input->m[input->size[1]*x + 1*y] + input->m[input->size[1]*nx + 1*y] + input->m[input->size[1]*px + 1*y] + input->m[input->size[1]*x + 1*ny] + input->m[input->size[1]*x + 1*py];
                    output->m[output->size[1]*x + 1*y] /= 5.000000;
                }
            }
        }
        memcpy_field_floatfield2(host_field, *output);

        /* Print current state */

        for (y = 0; y < size_y; ++y) {
            int x;

            for (x = 0; x < size_x; ++x) {

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
    free_host_field_floatfield2(host_field);
    free_device_field_floatfield2(device_field_1);
    free_device_field_floatfield2(device_field_2);

    return 0;
}
