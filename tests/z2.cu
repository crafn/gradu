#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memcpy */
#include <math.h>
#include <stdint.h>

void *cuda_upload_var(void *host_var, int size)
{
	void *cuda_var;
	cudaMalloc(&cuda_var, 4);
	cudaMemcpy(cuda_var, host_var, size, cudaMemcpyHostToDevice);
	return cuda_var;
}
void cuda_download_var(void *cuda_var, void *host_var, int size)
{
	cudaMemcpy(host_var, cuda_var, size, cudaMemcpyDeviceToHost);
	cudaFree(cuda_var);
}

typedef struct intfield5
{
    int *m;
    int size[5];
    int is_device_field;
} intfield5;

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

intfield5 alloc_device_field_intfield5(int size_0, int size_1, int size_2, int size_3, int size_4)
{
    intfield5 field;
    cudaMalloc((void**)&field.m, (sizeof(*field.m))*size_0*size_1*size_2*size_3*size_4);
    field.size[0] = size_0;
    field.size[1] = size_1;
    field.size[2] = size_2;
    field.size[3] = size_3;
    field.size[4] = size_4;
    field.is_device_field = 1;
    return field;
}

void free_field_intfield5(intfield5 field)
{
    free(field.m);
}

void free_device_field_intfield5(intfield5 field)
{
    cudaFree(field.m);
}

void memcpy_field_intfield5(intfield5 dst, intfield5 src)
{
    if (dst.is_device_field == 0 && src.is_device_field == 0) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1]*dst.size[2]*dst.size[3]*dst.size[4], cudaMemcpyHostToHost);
    }
    if (dst.is_device_field == 1 && src.is_device_field == 0) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1]*dst.size[2]*dst.size[3]*dst.size[4], cudaMemcpyHostToDevice);
    }
    if (dst.is_device_field == 0 && src.is_device_field == 1) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1]*dst.size[2]*dst.size[3]*dst.size[4], cudaMemcpyDeviceToHost);
    }
    if (dst.is_device_field == 1 && src.is_device_field == 1) {
        cudaMemcpy(dst.m, src.m, (sizeof(*dst.m))*dst.size[0]*dst.size[1]*dst.size[2]*dst.size[3]*dst.size[4], cudaMemcpyDeviceToDevice);
    }
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





/* the lattice is of dimensions SIZE**4  */

__constant__ const int SIZE = 10;

typedef intfield5 Links; /* Last index is link direction */

intfield5 link;

__constant__ const int RAND_DATA_COUNT = 128;

/* Poor man's random generator */

__constant__ const float rand_data[128] = {
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

__host__ __device__ void moveup(intmat5 *x, int d)
{
    x->m[1*d] += 1;

    if (x->m[1*d] >= SIZE) {
        x->m[1*d] -= SIZE;
    }
}

__host__ __device__ void movedown(intmat5 *x, int d)
{
    x->m[1*d] -= 1;

    if (x->m[1*d] < 0) {
        x->m[1*d] += SIZE;
    }
}
__global__ void kernel_0(intfield5 link)
{
    if (threadIdx.x + blockIdx.x*blockDim.x >= link.size[0]*link.size[1]*link.size[2]*link.size[3]*link.size[4]) {
        return;
    }
    intmat5 id;
    id.m[1*0] = (threadIdx.x + blockIdx.x*blockDim.x) % link.size[0]/1;
    id.m[1*1] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1])/link.size[0];
    id.m[1*2] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1]*link.size[2])/(link.size[0]*link.size[1]);
    id.m[1*3] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1]*link.size[2]*link.size[3])/(link.size[0]*link.size[1]*link.size[2]);
    id.m[1*4] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1]*link.size[2]*link.size[3]*link.size[4])/(link.size[0]*link.size[1]*link.size[2]*link.size[3]);
    link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*id.m[1*4]] = 1;
}


void coldstart()
{

    {
        dim3 dim_grid(link.size[0]*link.size[1]*link.size[2]*link.size[3]*link.size[4]/128 + 1, 1, 1);
        dim3 dim_block(128, 1, 1);
        kernel_0<<<dim_grid, dim_block>>>(link);
    }
}
__global__ void kernel_1(intfield5 link, double beta, int iter, float *cuda_action, int is_odd)
{
    if (threadIdx.x + blockIdx.x*blockDim.x >= link.size[0]*link.size[1]*link.size[2]*link.size[3]*link.size[4]) {
        return;
    }
    intmat5 id;
    id.m[1*0] = (threadIdx.x + blockIdx.x*blockDim.x) % link.size[0]/1;
    id.m[1*1] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1])/link.size[0];
    id.m[1*2] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1]*link.size[2])/(link.size[0]*link.size[1]);
    id.m[1*3] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1]*link.size[2]*link.size[3])/(link.size[0]*link.size[1]*link.size[2]);
    id.m[1*4] = (threadIdx.x + blockIdx.x*blockDim.x) % (link.size[0]*link.size[1]*link.size[2]*link.size[3]*link.size[4])/(link.size[0]*link.size[1]*link.size[2]*link.size[3]);
    if ((id.m[1*0] + id.m[1*1] + id.m[1*2] + id.m[1*3] + id.m[1*4]) % 2 == is_odd) {
        return;
    }

    int dperp;

    float staplesum = 0;

    int staple;

    float bplus;

    float bminus;

    int d = id.m[1*4];

    for (dperp = 0; dperp < 4; dperp += 1) {

        if (dperp != d) {
            movedown(&id, dperp);

            int v1 = link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*dperp];

            int v2 = link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*d];
            staple = v1*v2;
            moveup(&id, d);
            staple *= link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*dperp];
            moveup(&id, dperp);
            staplesum += staple;
            staple = link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*dperp];
            moveup(&id, dperp);
            movedown(&id, d);
            staple *= link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*d];
            movedown(&id, dperp);
            staple *= link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*dperp];
            staplesum += staple;
        }
    }
    bplus = exp(beta*staplesum);
    bminus = 1/bplus;
    bplus = bplus/(bplus + bminus);

    int rand_ix = id.m[1*0] + id.m[1*1]*SIZE + id.m[1*3]*SIZE*SIZE + id.m[1*4]*SIZE*SIZE*SIZE + iter*SIZE*SIZE*SIZE*SIZE;

    if (rand_data[rand_ix % RAND_DATA_COUNT] < bplus) {
        link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*d] = 1;
        atomicAdd(cuda_action, staplesum);
    } else {
        link.m[link.size[1]*link.size[2]*link.size[3]*link.size[4]*id.m[1*0] + link.size[2]*link.size[3]*link.size[4]*id.m[1*1] + link.size[3]*link.size[4]*id.m[1*2] + link.size[4]*id.m[1*3] + 1*d] = -1;
        atomicAdd(cuda_action, -staplesum);
    }
}


double update(double beta, int iter)
{

    float action = 0.000000;

    {
        float *cuda_action = (float*)cuda_upload_var(&action, sizeof(action));
        dim3 dim_grid(link.size[0]*link.size[1]*link.size[2]*link.size[3]*link.size[4]/128 + 1, 1, 1);
        dim3 dim_block(128, 1, 1);
        kernel_1<<<dim_grid, dim_block>>>(link, beta, iter, cuda_action, 0);
        kernel_1<<<dim_grid, dim_block>>>(link, beta, iter, cuda_action, 1);
        cuda_download_var(cuda_action, &action, sizeof(action));
    }
    action /= SIZE*SIZE*SIZE*SIZE*4*6;

    return 1.000000 - action;
}

int main()
{
    link = alloc_device_field_intfield5(SIZE, SIZE, SIZE, SIZE, 4);

    double beta;

    double action;

    double dbeta = 0.010000;
    coldstart();

    int iter = 0;

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
    free_device_field_intfield5(link);

    return 0;
}
