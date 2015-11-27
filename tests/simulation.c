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
    field.m = (float *)malloc(sizeof(*field.m) * size_0 * size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
    return field;
}

void free_field_floatfield2(floatfield2 field)
{
    free(field.m);
}

int printf(const char *fmt, ...);

typedef floatfield2 Field;

int main(int argc, char **argv)
{
    Field a;
    Field b;
    int i;
    a = alloc_field_floatfield2(20, 20);
    b = alloc_field_floatfield2(20, 20);

    for (i = 0; i < 10; ++i) {
        Field *input = &a;
        Field *output = &b;

        /* Swap */
        if (i % 2 == 1) {
            Field *tmp = output;
            output = input;
            input = output;
        }
    }

    free_field_floatfield2(a);
    free_field_floatfield2(b);
    return 0;
}
