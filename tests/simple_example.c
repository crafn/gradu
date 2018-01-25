#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memcpy */
#include <math.h>
#include <stdint.h>

typedef struct intfield1
{
    int *m;
    int size[1];
    int is_device_field;
} intfield1;

void memcpy_field_intfield1(intfield1 dst, intfield1 src)
{
    memcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]);
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
    field.m = (int*)malloc((sizeof(*field.m))*size_0);
    field.size[0] = size_0;
    field.is_device_field = 0;
    return field;
}

void free_device_field_intfield1(intfield1 field)
{
    free(field.m);
}

typedef struct intmat1
{
    int m[1];
} intmat1;

int printf(const char *fmt, ...); /* TODO: Remove */

typedef intfield1 Field; /* One-dimensional integer field type */

int main()
{
    int i;

    int N = 5;

    intfield1 a_data = alloc_host_field_intfield1(N);

    intfield1 b_data = alloc_host_field_intfield1(N);

    intfield1 a;

    intfield1 b;
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
    a = alloc_device_field_intfield1(N);
    b = alloc_device_field_intfield1(N);
    memcpy_field_intfield1(a, a_data);
    memcpy_field_intfield1(b, b_data);

    {
        int id_0;
        for (id_0 = 0; id_0 < size_intfield1(a, 0); ++id_0) {
            intmat1 id;
            id.m[1*0] = id_0;
            a.m[1*id.m[1*0]] += b.m[1*id.m[1*0]];
        }
    }
    memcpy_field_intfield1(a_data, a);

    for (i = 0; i < N; ++i) {
        printf("%i ", a_data.m[1*i]);
    }
    free_host_field_intfield1(a_data);
    free_host_field_intfield1(b_data);
    free_device_field_intfield1(a);
    free_device_field_intfield1(b);

    return 0;
}
