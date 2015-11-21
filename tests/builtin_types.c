typedef struct intmat2x2
{
    int m[4];
} intmat2x2;

intmat2x2 intmat2x2_mul(intmat2x2 lhs, intmat2x2 rhs)
{
    intmat2x2 ret;
    ret.m[0] = lhs.m[0] * rhs.m[0] + lhs.m[1] * rhs.m[2];
    ret.m[2] = lhs.m[2] * rhs.m[0] + lhs.m[3] * rhs.m[2];
    ret.m[1] = lhs.m[0] * rhs.m[1] + lhs.m[1] * rhs.m[3];
    ret.m[3] = lhs.m[2] * rhs.m[1] + lhs.m[3] * rhs.m[3];
    return ret;
}

typedef struct floatmat3x3
{
    float m[9];
} floatmat3x3;

floatmat3x3 floatmat3x3_mul(floatmat3x3 lhs, floatmat3x3 rhs)
{
    floatmat3x3 ret;
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


int main(int argc, char **argv)
{
    intmat2x2 mat1;
    intmat2x2 mat2;
    floatmat3x3 mat3;
    floatmat3x3 mat4;
    int i;
    int k;

    mat1.m[1 * 0 + 2 * 0] = 0;
    mat1.m[1 * 1 + 2 * 0] = 1;
    mat1.m[1 * 1 + 2 * 1] = 2;
    mat1.m[1 * 0 + 2 * 1] = 3;

    mat2.m[1 * 0 + 2 * 0] = 0;
    mat2.m[1 * 1 + 2 * 0] = 1;
    mat2.m[1 * 1 + 2 * 1] = 2;
    mat2.m[1 * 0 + 2 * 1] = 3;

    for (i = 0; i < 3; i = i + 1) {
        for (k = 0; k < 3; k = k + 1) {
            mat3.m[1 * i + 3 * k] = 1;
            mat4.m[1 * i + 3 * k] = 2;
        }
    }

    /* This comment should survive to the generated code */
    mat1 = intmat2x2_mul(mat1, mat2);
    mat3 = floatmat3x3_mul(floatmat3x3_mul(floatmat3x3_mul(mat4, mat3), mat3), mat3);

    return 0;
}
