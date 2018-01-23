#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memcpy */
#include <math.h>
#include <stdint.h>

typedef struct intfield5
{
    int *m;
    int size[5];
    int is_device_field;
} intfield5;


/* the lattice is of dimensions SIZE**4  */
const int SIZE = 10;

intfield5 link;

const int RAND_DATA_COUNT = 128;
/* Poor man's random generator */

const float rand_data[128] = {
    0.765778,
    0.380508,
    0.976123,
    0.047972,
    0.027949,
    0.493132,
    0.145068,
    0.937659,
    0.688443,
    0.317046,
    0.803646,
    0.917738,
    0.513913,
    0.363706,
    0.137274,
    0.666660,
    0.250019,
    0.622242,
    0.021247,
    0.406825,
    0.707708,
    0.856293,
    0.947693,
    0.207796,
    0.362935,
    0.902242,
    0.427960,
    0.704711,
    0.613763,
    0.660261,
    0.378255,
    0.654958,
    0.936904,
    0.683342,
    0.891384,
    0.299881,
    0.064560,
    0.300503,
    0.572774,
    0.132678,
    0.132292,
    0.438706,
    0.594546,
    0.837315,
    0.180435,
    0.215016,
    0.726831,
    0.767127,
    0.556461,
    0.860724,
    0.132273,
    0.288679,
    0.001132,
    0.946316,
    0.740891,
    0.502307,
    0.189147,
    0.609733,
    0.716687,
    0.098146,
    0.650990,
    0.476326,
    0.958396,
    0.458836,
    0.834419,
    0.876043,
    0.820873,
    0.433127,
    0.800544,
    0.939788,
    0.741833,
    0.905454,
    0.796914,
    0.567545,
    0.054171,
    0.333496,
    0.247967,
    0.880176,
    0.760589,
    0.769755,
    0.011049,
    0.361483,
    0.829162,
    0.228125,
    0.572835,
    0.854979,
    0.070170,
    0.759810,
    0.022272,
    0.477994,
    0.014528,
    0.991334,
    0.314297,
    0.940028,
    0.235618,
    0.840691,
    0.882266,
    0.840194,
    0.985364,
    0.713334,
    0.697650,
    0.090573,
    0.262273,
    0.534600,
    0.761973,
    0.146971,
    0.667842,
    0.069159,
    0.102225,
    0.982492,
    0.933260,
    0.441284,
    0.149844,
    0.039490,
    0.520590,
    0.071531,
    0.141776,
    0.701622,
    0.213773,
    0.717888,
    0.621524,
    0.285984,
    0.442431,
    0.471437,
    0.197912,
    0.314655,
    0.496274,
    0.896794
};
intfield5 alloc_field_intfield5(int size_0, int size_1, int size_2, int size_3, int size_4)
{
    intfield5 field;
    field.m = (int*)malloc((sizeof(*field.m))*size_0*size_1*size_2*size_3*size_4);
    field.size[0] = size_0;
    field.size[1] = size_1;
    field.size[2] = size_2;
    field.size[3] = size_3;
    field.size[4] = size_4;
    field.is_device_field = 0;
    return field;
}

void free_field_intfield5(intfield5 field)
{
    free(field.m);
}

void memcpy_field_intfield5(intfield5 dst, intfield5 src)
{
    memcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1]*dst.size[2]*dst.size[3]*dst.size[4]);
}

int size_intfield5(intfield5 field, int index)
{
    return field.size[index];
}

typedef struct intmat5
{
    int m[5];
} intmat5;

/* Adapted from: */
/* Z_2 lattice gauge simulation */
/* Michael Creutz <creutz@bnl.gov>     */
/* http://thy.phy.bnl.gov/~creutz/z2.c */

typedef intfield5 Links; /* Last index is link direction */

void moveup(intmat5 *x, int d)
{
    x->m[1*d] += 1;
    if (x->m[1*d] >= SIZE) {
        x->m[1*d] -= SIZE;
    }
}

void movedown(intmat5 *x, int d)
{
    x->m[1*d] -= 1;
    if (x->m[1*d] < 0) {
        x->m[1*d] += SIZE;
    }
}

void coldstart()
{
    {
        int id_0;
        for (id_0 = 0; id_0 < size_intfield5(link, 0); ++id_0) {
            int id_1;
            for (id_1 = 0; id_1 < size_intfield5(link, 1); ++id_1) {
                int id_2;
                for (id_2 = 0; id_2 < size_intfield5(link, 2); ++id_2) {
                    int id_3;
                    for (id_3 = 0; id_3 < size_intfield5(link, 3); ++id_3) {
                        int id_4;
                        for (id_4 = 0; id_4 < size_intfield5(link, 4); ++id_4) {
                            intmat5 id;
                            id.m[1*4] = id_4;
                            id.m[1*3] = id_3;
                            id.m[1*2] = id_2;
                            id.m[1*1] = id_1;
                            id.m[1*0] = id_0;
                            link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*id.m[1*4]] = 1;
                        }
                    }
                }
            }
        }
    }
}

double update(double beta, int iter)
{
    float action = 0.000000;

    {
        int id_0;
        for (id_0 = 0; id_0 < size_intfield5(link, 0); ++id_0) {
            int id_1;
            for (id_1 = 0; id_1 < size_intfield5(link, 1); ++id_1) {
                int id_2;
                for (id_2 = 0; id_2 < size_intfield5(link, 2); ++id_2) {
                    int id_3;
                    for (id_3 = 0; id_3 < size_intfield5(link, 3); ++id_3) {
                        int id_4;
                        for (id_4 = 0; id_4 < size_intfield5(link, 4); ++id_4) {
                            intmat5 id;
                            int dperp;
                            float staplesum;
                            int staple;
                            float bplus;
                            float bminus;

                            int d;

                            int rand_ix;
                            id.m[1*4] = id_4;
                            id.m[1*3] = id_3;
                            id.m[1*2] = id_2;
                            id.m[1*1] = id_1;
                            id.m[1*0] = id_0;
                            staplesum = 0;
                            d = id.m[1*4];
                            for (dperp = 0; dperp < 4; dperp += 1) {
                                if (dperp != d) {
                                    int v1;
                                    int v2;
                                    movedown(&id, dperp);
                                    v1 = link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*dperp];
                                    v2 = link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*d];
                                    staple = v1*v2;
                                    moveup(&id, d);
                                    staple *= link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*dperp];
                                    moveup(&id, dperp);
                                    staplesum += staple;
                                    staple = link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*dperp];
                                    moveup(&id, dperp);
                                    movedown(&id, d);
                                    staple *= link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*d];
                                    movedown(&id, dperp);
                                    staple *= link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*dperp];
                                    staplesum += staple;
                                }
                            }
                            bplus = exp(beta*staplesum);
                            bminus = 1/bplus;
                            bplus = bplus/(bplus + bminus);
                            rand_ix = id.m[1*0] + id.m[1*1]*SIZE + id.m[1*3]*SIZE*SIZE + id.m[1*4]*SIZE*SIZE*SIZE + iter*SIZE*SIZE*SIZE*SIZE;
                            if (rand_data[rand_ix % RAND_DATA_COUNT] < bplus) {
                                link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*d] = 1;
                                action += staplesum;
                            } else {
                                link.m[1*id.m[1*0] + link.size[0]*id.m[1*1] + link.size[0]*link.size[1]*id.m[1*2] + link.size[0]*link.size[1]*link.size[2]*id.m[1*3] + link.size[0]*link.size[1]*link.size[2]*link.size[3]*d] = -1;
                                action -= staplesum;
                            }
                        }
                    }
                }
            }
        }
    }
    action /= SIZE*SIZE*SIZE*SIZE*4*6;
    return 1.000000 - action;
}

int main()
{

    double beta;
    double action;
    double dbeta;

    int iter;
    link = alloc_field_intfield5(SIZE, SIZE, SIZE, SIZE, 4);
    dbeta = 0.010000;
    coldstart();
    iter = 0;
    for (beta = 1; beta > 0.000000; beta -= dbeta) {
        action = update(beta, iter);
        printf("%g\t%g\n", beta, action);
        ++iter;
    }
    printf("\n\n");
    for (beta = 0; beta < 1.000000; beta += dbeta) {
        action = update(beta, iter);
        printf("%g\t%g\n", beta, action);
        ++iter;
    }
    free_field_intfield5(link);

    return 0;
}
