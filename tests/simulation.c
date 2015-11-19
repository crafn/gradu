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

    intmat3x3 mat1;
    intmat3x3 mat2;
    intmat3x3 mat3;
    intmat3x3 result;

    mat1.m[0 + 0 * 3] = 1;
    mat1.m[1 + 0 * 3] = -1 + -2 * -3;
    mat1.m[0 + 1 * 3] = 3;
    mat1.m[1 + 1 * 3] = 4;

    mat2.m[0 + 0 * 3] = 5;
    mat2.m[1 + 0 * 3] = 5;
    mat2.m[0 + 1 * 3] = 5;
    mat2.m[1 + 1 * 3] = 6;

    mat3.m[0 + 0 * 3] = 1;
    mat3.m[1 + 0 * 3] = 1;
    mat3.m[0 + 1 * 3] = 2;
    mat3.m[1 + 1 * 3] = 3;

    /* This comment should survive to the generated code */
    result = intmat3x3_mul(intmat3x3_mul(mat1, mat2), mat3);

    printf("(%i, %i)\n(%i, %i)\n", result.m[0 + 0 * 3], result.m[1 + 0 * 3], result.m[0 + 1 * 3], result.m[1 + 1 * 3]);

    return 0;
}
