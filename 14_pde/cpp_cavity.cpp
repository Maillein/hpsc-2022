#include <cstdio>
#include <utility>
#include <vector>

const int nx  = 41;
const int ny  = 41;
const int nt  = 500;
const int nit = 50;
const double dx  = (2.0 / (nx - 1));
const double dy  = (2.0 / (ny - 1));
const double dx2 = dx * dx;
const double dy2 = dy * dy;
const double dt  = 0.01;
const double rho = 1.0;
const double nu  = 0.02;

// using matrix = std::vector<std::vector<double>>;
typedef std::vector< std::vector<double> > matrix;

void initialize(matrix &u, matrix &v, matrix &p, matrix &b) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            u[j][i] = 0.0;
            v[j][i] = 0.0;
            p[j][i] = 0.0;
            b[j][i] = 0.0;
        }
    }
}

void compute_tmp_velocity(matrix &u, matrix &v, matrix &b) {
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            b[j][i] = rho * (1 / dt *
                     ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    (((u[j][i+1] - u[j][i-1]) / (2 * dx)) * ((u[j][i+1] - u[j][i-1]) / (2 * dx))) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                      (v[j][i+1] - v[j][i-1]) / (2 * dx)) - (((v[j+1][i] - v[j-1][i]) / (2 * dy)) * ((v[j+1][i] - v[j-1][i]) / (2 * dy))));
        }
    }
}

void solve_poisson_equation(matrix &p, matrix &pn, matrix &b) {
    for (int _iter = 0; _iter < nit; _iter++) {
        for (int i = 0; i < ny; i++) {
            std::copy(p[i].begin(), p[i].end(), pn[i].begin());
        }
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                p[j][i] = (dy2 * (pn[j][i+1] + pn[j][i-1]) +
                           dx2 * (pn[j+1][i] + pn[j-1][i]) -
                           b[j][i] * dx2 * dy2)
                          / (2 * (dx2 + dy2));
            }
        }

        // boundary condition
        for (int j = 0; j < ny; j++) {
            p[j][nx-1] = p[j][nx-2]; // right
            p[j][0] = p[j][1];       // left
        }
        for (int i = 0; i < nx; i++) {
            p[ny-1][i] = 0;              // top
            p[0][i] = p[1][i];       // bottom
        }
    }
}

void adjust_velocity(matrix &u, matrix &v, matrix &un, matrix &vn, matrix &p) {
    for (int i = 0; i < ny; i++) {
        std::copy(u[i].begin(), u[i].end(), un[i].begin());
        std::copy(v[i].begin(), v[i].end(), vn[i].begin());
    }

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                               - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                               - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                               + nu * dt / (dx2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                               + nu * dt / (dy2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
            v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                               - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                               - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                               + nu * dt / (dx2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                               + nu * dt / (dy2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
        }
    }
    // boundary condition
    for (int j = 0; j < ny; j++) {
        u[j][nx-1] = 0; // right
        v[j][nx-1] = 0;
        u[j][0] = 0;    // left
        v[j][0] = 0;
    }
    for (int i = 0; i < nx; i++) {
        u[ny-1][i] = 1; // top
        v[ny-1][i] = 0;
        u[0][i] = 0;    // bottom
        v[0][i] = 0;
    }
}

int main() {
    matrix u(ny, std::vector<double>(nx));
    matrix v(ny, std::vector<double>(nx));
    matrix p(ny, std::vector<double>(nx));
    matrix b(ny, std::vector<double>(nx));
    matrix un(ny, std::vector<double>(nx));
    matrix vn(ny, std::vector<double>(nx));
    matrix pn(ny, std::vector<double>(nx));

    initialize(u, v, p, b);

    for (int _i = 0; _i < nt; _i++) {
        compute_tmp_velocity(u, v, b);
        solve_poisson_equation(p, pn, b);
        adjust_velocity(u, v, un, vn, p);
    }
    return 0;
}
