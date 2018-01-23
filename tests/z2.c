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

double quasirand(uint64_t ix)
{
    return ix*2011 % 137/137.000000; /* @todo Better generator */
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

                            if (quasirand(id.m[1*0] + id.m[1*1]*SIZE + id.m[1*3]*SIZE*SIZE + id.m[1*4]*SIZE*SIZE*SIZE + iter*SIZE*SIZE*SIZE*SIZE) < bplus) {
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
