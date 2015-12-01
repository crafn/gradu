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

void free_field_floatfield2(floatfield2 field)
{
    free(field.m);
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
int main(int argc, char **argv)
{
    int size_x = 20;
    int size_y = 20;
    Field a = alloc_field_floatfield2(size_x, size_y);
    Field b = alloc_field_floatfield2(size_x, size_y);
    int i;
    {
        int x;
        int y;
        for (x = 0; x < size_x; ++x) {
            for (y = 0; y < size_y; ++y) {
                a.m[1 * x + a.size[0] * y] = 0;
            }
        }
        a.m[1 * size_x / 2 + a.size[0] * size_y / 2] = 1000;
    }
    for (i = 0; i < 20; ++i) {
        Field *input = &a;
        Field *output = &b;

        /* Swap */
        if (i % 2 == 1) {
            Field *tmp = output;
            output = input;
            input = tmp;
        }
        {
            for (int id_0 = 0; id_0 < size_floatfield2(*output, 0); ++id_0) {
                for (int id_1 = 0; id_1 < size_floatfield2(*output, 1); ++id_1) {
                    intmat2 id;
                    id.m[1 * 0] = id_0;
                    id.m[1 * 1] = id_1;
                    int x = id.m[1 * 0];
                    int y = id.m[1 * 1];
                    int nx = (x + 1) % size_x;
                    int ny = (y + 1) % size_y;
                    int px = (x - 1 + size_x) % size_x;
                    int py = (y - 1 + size_y) % size_y;
                    output->m[1 * x + output->size[0] * y] = input->m[1 * x + input->size[0] * y] + input->m[1 * nx + input->size[0] * y] + input->m[1 * px + input->size[0] * y] + input->m[1 * x + input->size[0] * ny] + input->m[1 * x + input->size[0] * py];
                    output->m[1 * x + output->size[0] * y] /= 5.000000;
                }
            }
        }

        /* Print current state */
        {
            int x;
            int y;
            for (y = 0; y < size_y; ++y) {
                for (x = 0; x < size_x; ++x) {
                    char *ch = " ";
                    if (output->m[1 * x + output->size[0] * y] > 5.000000) {
                        ch = "#";
                    } else if (output->m[1 * x + output->size[0] * y] > 1.000000) {
                        ch = ".";
                    }
                    printf("%s", ch);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    free_field_floatfield2(a);
    free_field_floatfield2(b);
    return 0;
}

