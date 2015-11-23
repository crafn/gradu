#include <stdio.h>
#include <stdlib.h>
typedef struct intfield3
{
    int *m;
    int size[3];
} intfield3;

intfield3 alloc_field_intfield3(int size_0, int size_1, int size_2)
{
    intfield3 field;
    field.m = malloc(sizeof(*field.m) * size_0 * size_1 * size_2);
    field.size[0] = size_0;
    field.size[1] = size_1;
    field.size[2] = size_2;
    return field;
}

void free_field_intfield3(intfield3 field)
{
    free(field.m);
}

typedef struct floatmat2x2
{
    float m[4];
} floatmat2x2;

floatmat2x2 floatmat2x2_mul(floatmat2x2 lhs, floatmat2x2 rhs)
{
    floatmat2x2 ret;
    ret.m[0] = lhs.m[0] * rhs.m[0] + lhs.m[1] * rhs.m[2];
    ret.m[2] = lhs.m[2] * rhs.m[0] + lhs.m[3] * rhs.m[2];
    ret.m[1] = lhs.m[0] * rhs.m[1] + lhs.m[1] * rhs.m[3];
    ret.m[3] = lhs.m[2] * rhs.m[1] + lhs.m[3] * rhs.m[3];
    return ret;
}

typedef struct floatmat2x2field2
{
    floatmat2x2 *m;
    int size[2];
} floatmat2x2field2;

floatmat2x2field2 alloc_field_floatmat2x2field2(int size_0, int size_1)
{
    floatmat2x2field2 field;
    field.m = malloc(sizeof(*field.m) * size_0 * size_1);
    field.size[0] = size_0;
    field.size[1] = size_1;
    return field;
}

void free_field_floatmat2x2field2(floatmat2x2field2 field)
{
    free(field.m);
}

typedef struct intmat3x3
{
    int m[9];
} intmat3x3;

intmat3x3 intmat3x3_mul(intmat3x3 lhs, intmat3x3 rhs)
{
    intmat3x3 ret;
    ret.m[0] = lhs.m[0] * rhs.m[0] + lhs.m[1] * rhs.m[3] + lhs.m[2] * rhs.m[6];
    ret.m[3] = lhs.m[3] * rhs.m[0] + lhs.m[4] * rhs.m[3] + lhs.m[5] * rhs.m[6];
    ret.m[6] = lhs.m[6] * rhs.m[0] + lhs.m[7] * rhs.m[3] + lhs.m[8] * rhs.m[6];
    ret.m[1] = lhs.m[0] * rhs.m[1] + lhs.m[1] * rhs.m[4] + lhs.m[2] * rhs.m[7];
    ret.m[4] = lhs.m[3] * rhs.m[1] + lhs.m[4] * rhs.m[4] + lhs.m[5] * rhs.m[7];
    ret.m[7] = lhs.m[6] * rhs.m[1] + lhs.m[7] * rhs.m[4] + lhs.m[8] * rhs.m[7];
    ret.m[2] = lhs.m[0] * rhs.m[2] + lhs.m[1] * rhs.m[5] + lhs.m[2] * rhs.m[8];
    ret.m[5] = lhs.m[3] * rhs.m[2] + lhs.m[4] * rhs.m[5] + lhs.m[5] * rhs.m[8];
    ret.m[8] = lhs.m[6] * rhs.m[2] + lhs.m[7] * rhs.m[5] + lhs.m[8] * rhs.m[8];
    return ret;
}

int printf(const char *fmt, ...);

int main(int argc, char **argv)
{
    /* Not yet a simulation! */
    intfield3 test_field;
    floatmat2x2field2 test_field2;

    intmat3x3 mat1;
    intmat3x3 mat2;
    intmat3x3 mat3;
    intmat3x3 result;
    floatmat2x2 test;

    mat1.m[1 * 0 + 3 * 0] = 1;
    mat1.m[1 * 1 + 3 * 0] = -1 + -2 * -3;
    mat1.m[1 * 0 + 3 * 1] = 3;
    mat1.m[1 * 1 + 3 * 1] = 4;

    mat2.m[1 * 0 + 3 * 0] = 5;
    mat2.m[1 * 1 + 3 * 0] = 5;
    mat2.m[1 * 0 + 3 * 1] = 5;
    mat2.m[1 * 1 + 3 * 1] = 6;

    mat3.m[1 * 0 + 3 * 0] = 1;
    mat3.m[1 * 1 + 3 * 0] = 1;
    mat3.m[1 * 0 + 3 * 1] = 2;
    mat3.m[1 * 1 + 3 * 1] = 3;

    test_field = alloc_field_intfield3(100, 100, 100);
    test_field2 = alloc_field_floatmat2x2field2(200, 200);

    test_field.m[test_field.size[0] * 1 + test_field.size[1] * 2 + test_field.size[2] * 3] = 4;
    test_field2.m[test_field2.size[0] * 1 + test_field2.size[1] * 2] = test;

    test_field2.m[test_field2.size[0] * 13 + test_field2.size[1] * 37].m[1 * 1 + 2 * 0] = 3;

    /* This comment should survive to the generated code */
    result = intmat3x3_mul(intmat3x3_mul(mat1, mat2), mat3);
    test = floatmat2x2_mul(test_field2.m[test_field2.size[0] * 0 + test_field2.size[1] * 0], test_field2.m[test_field2.size[0] * 1 + test_field2.size[1] * 1]);

    printf("(%i, %i)\n(%i, %i)\n", result.m[1 * 0 + 3 * 0], result.m[1 * 1 + 3 * 0], result.m[1 * 0 + 3 * 1], result.m[1 * 1 + 3 * 1]);

    free_field_intfield3(test_field);
    free_field_floatmat2x2field2(test_field2);

    return 0;
}
