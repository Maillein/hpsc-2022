#include <cstdio>
#include <utility>

#define nx  41
#define ny  41
#define nt  500
#define nit 50
#define dx  (2.0 / (nx - 1))
#define dy  (2.0 / (ny - 1))
#define dt  0.01
#define rho 1.0
#define nu  0.02

#define at(y, x) ((y) * nx + (x))
#define divup(a,b) (((a) + (b) - 1) / (b))

__global__ void initialize(double *u, double *v, double *p, double *b) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    u[at(j,i)] = 0.0;
    v[at(j,i)] = 0.0;
    p[at(j,i)] = 0.0;
    b[at(j,i)] = 0.0;
}

__global__ void compute_tmp_velocity(double *u, double *v, double *b) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (1 <= i && i < nx - 1 && 1 <= j && j < ny - 1) {
        b[at(j, i)] = rho * (1 / dt *
                 ((u[at(j, i+1)] - u[at(j, i-1)]) / (2 * dx) + (v[at(j+1, i)] - v[at(j-1, i)]) / (2 * dy)) -
                (((u[at(j, i+1)] - u[at(j, i-1)]) / (2 * dx)) * ((u[at(j, i+1)] - u[at(j, i-1)]) / (2 * dx))) - 2 * ((u[at(j+1, i)] - u[at(j-1, i)]) / (2 * dy) *
                  (v[at(j, i+1)] - v[at(j, i-1)]) / (2 * dx)) - (((v[at(j+1, i)] - v[at(j-1, i)]) / (2 * dy)) * ((v[at(j+1, i)] - v[at(j-1, i)]) / (2 * dy))));
    }
}

__global__ void solve_poisson_equation(double *p, double *pn, double *b) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    for (int _iter = 0; _iter < nit; _iter++) {
        pn[at(j,i)] = p[at(j,i)];
        __syncthreads();

        if (1 <= j && j < ny - 1 && 1 <= i && i < nx - 1) {
            p[at(j, i)] = (dy*dy * (pn[at(j, i+1)] + pn[at(j, i-1)]) +
                       dx*dx * (pn[at(j+1, i)] + pn[at(j-1, i)]) -
                       b[at(j, i)] * dx*dx * dy*dy)
                      / (2 * (dx*dx + dy*dy));
        }
        __syncthreads();

        // boundary condition
        if (i == nx - 1) {
            p[at(j, nx-1)] = p[at(j, nx-2)]; // right
        }
        if (i == 0) {
            p[at(j, 0)] = p[at(j, 1)];       // left
        }
        if (j == ny - 1) {
            p[at(ny-1, i)] = 0;              // top
        }
        if (j == 0) {
            p[at(0, i)] = p[at(1, i)];       // bottom
        }
        __syncthreads();
    }
}

__global__ void adjust_velocity(double *u, double *v, double *un, double *vn, double *p) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    un[at(j,i)] = u[at(j,i)];
    vn[at(j,i)] = v[at(j,i)];
    __syncthreads();

    if (1 <= j && j < ny - 1 && 1 <= i && i < nx - 1) {
        u[at(j, i)] = un[at(j, i)] - un[at(j, i)] * dt / dx * (un[at(j, i)] - un[at(j, i - 1)])
                           - un[at(j, i)] * dt / dy * (un[at(j, i)] - un[at(j - 1, i)])
                           - dt / (2 * rho * dx) * (p[at(j, i+1)] - p[at(j, i-1)])
                           + nu * dt / (dx*dx) * (un[at(j, i+1)] - 2 * un[at(j, i)] + un[at(j, i-1)])
                           + nu * dt / (dy*dy) * (un[at(j+1, i)] - 2 * un[at(j, i)] + un[at(j-1, i)]);
        v[at(j, i)] = vn[at(j, i)] - vn[at(j, i)] * dt / dx * (vn[at(j, i)] - vn[at(j, i - 1)])
                           - vn[at(j, i)] * dt / dy * (vn[at(j, i)] - vn[at(j - 1, i)])
                           - dt / (2 * rho * dx) * (p[at(j+1, i)] - p[at(j-1, i)])
                           + nu * dt / (dx*dx) * (vn[at(j, i+1)] - 2 * vn[at(j, i)] + vn[at(j, i-1)])
                           + nu * dt / (dy*dy) * (vn[at(j+1, i)] - 2 * vn[at(j, i)] + vn[at(j-1, i)]);
    }

    // boundary condition
    if (i == nx - 1) {
        u[at(j, nx-1)] = 0; // right
        v[at(j, nx-1)] = 0;
    }
    if (i == 0) {
        u[at(j, 0)] = 0;    // left
        v[at(j, 0)] = 0;
    }
    if (j == ny - 1) {
        u[at(ny-1, i)] = 1; // top
        v[at(ny-1, i)] = 0;
    }
    if (j == 0) {
        u[at(0, i)] = 0;    // bottom
        v[at(0, i)] = 0;
    }
    __syncthreads();
}

int main() {
    double *u, *v, *p, *b;
    double *un, *vn, *pn;
    
    cudaMallocManaged((void**)&u,  ny * nx * sizeof(double));
    cudaMallocManaged((void**)&v,  ny * nx * sizeof(double));
    cudaMallocManaged((void**)&p,  ny * nx * sizeof(double));
    cudaMallocManaged((void**)&b,  ny * nx * sizeof(double));
    cudaMallocManaged((void**)&un, ny * nx * sizeof(double));
    cudaMallocManaged((void**)&vn, ny * nx * sizeof(double));
    cudaMallocManaged((void**)&pn, ny * nx * sizeof(double));

    dim3 block(32, 32);
    dim3 grid(divup(nx,block.x), divup(ny,block.y));

    initialize<<< block, grid >>>(u, v, p, b);
    cudaDeviceSynchronize();

    for (int _i = 0; _i < nt; _i++) {
        compute_tmp_velocity<<< block, grid >>>(u, v, b);
        cudaDeviceSynchronize();

        solve_poisson_equation<<< block, grid >>>(p, pn, b);
        cudaDeviceSynchronize();
        
        adjust_velocity<<< block, grid >>>(u, v, un, vn, p);
        cudaDeviceSynchronize();
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
    return 0;
}
