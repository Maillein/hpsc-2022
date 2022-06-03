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

#define at(y, x) (y * nx + x)

void initialize(double *u, double *v, double *p, double *b) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            u[at(j, i)] = 0.0;
            v[at(j, i)] = 0.0;
            p[at(j, i)] = 0.0;
            b[at(j, i)] = 0.0;
        }
    }
}

void compute_tmp_velocity(double *u, double *v, double *b) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            b[at(j, i)] = rho * (1 / dt *
                     ((u[at(j, i+1)] - u[at(j, i-1)]) / (2 * dx) + (v[at(j+1, i)] - v[at(j-1, i)]) / (2 * dy)) -
                    (((u[at(j, i+1)] - u[at(j, i-1)]) / (2 * dx)) * ((u[at(j, i+1)] - u[at(j, i-1)]) / (2 * dx))) - 2 * ((u[at(j+1, i)] - u[at(j-1, i)]) / (2 * dy) *
                      (v[at(j, i+1)] - v[at(j, i-1)]) / (2 * dx)) - (((v[at(j+1, i)] - v[at(j-1, i)]) / (2 * dy)) * ((v[at(j+1, i)] - v[at(j-1, i)]) / (2 * dy))));
        }
    }
}

void solve_poisson_equation(double *u, double *v, double *p, double *pn, double *b) {
    for (int _iter = 0; _iter < nit; _iter++) {
        memcpy(pn, p, ny * nx * sizeof(double));
        for (int j = 1, j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                p[at(j, i)] = (dy*dy * (pn[at(j, i+1)] + pn[at(j, i-1)]) +
                           dx*dx * (pn[at(j+1, i)] + pn[at(j-1, i)]) -
                           b[at(j, i)] * dx*dx * dy*dy)
                          / (2 * (dx*dx + dy*dy));
            }
        }

        // boundary condition
        for (int j = 0; j < ny; j++) {
            p[at(j, nx-1)] = p[at(j, nx-2)]; // right
            p[at(j, 0)] = p[at(j, 1)];       // left
        }
        for (int i = 0; i < nx, i++) {
            p[at(ny-1, i)] = 0;              // top
            p[at(0, i)] = p[at(1, i)];       // bottom
        }
    }
}

void adjust_velocity(double *u, double *v, double *un, double * vn) {
    memcpy(un, u, ny * nx * sizeof(double));
    memcpy(vn, v, ny * nx * sizeof(double));
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
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
    }
    // boundary condition
    for (int j = 0; j < ny; j++) {
        u[at(j, nx-1)] = 0; // right
        v[at(j, nx-1)] = 0;
        u[at(j, 0)] = 0;    // left
        v[at(j, 0)] = 0;
    }
    for (int i = 0; i < nx, i++) {
        u[at(ny-1, i)] = 1; // top
        v[at(ny-1, i)] = 0;
        u[at(0, i)] = 0;    // bottom
        v[at(0, i)] = 0;
    }
}

int main() {
    double *u  = (*double)malloc(ny * nx * sizeof(double));
    double *un = (*double)malloc(ny * nx * sizeof(double));
    double *v  = (*double)malloc(ny * nx * sizeof(double));
    double *vn = (*double)malloc(ny * nx * sizeof(double));
    double *p  = (*double)malloc(ny * nx * sizeof(double));
    double *pn = (*double)malloc(ny * nx * sizeof(double));
    double *b  = (*double)malloc(ny * nx * sizeof(double));

    initialize(u, v, p, b);

    for (int _i = 0; _i < nt; _i++) {
        compute_tmp_velocity(u, v, b);
        solve_poisson_equation(u, v, p, pn, b);
        adjust_velocity(u, v, un, vn);
    }
    return 0;
}
