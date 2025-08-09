// Channel flow on a rectangular domain using a staggered MAC grid + projection (explicit FE)
// THIS CODE WORKS

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <sstream>  // Added missing header for std::ostringstream

namespace ChannelFlow {

// ANSI colors for pretty logs (safe to ignore if your terminal doesn't support them)
constexpr const char* RESET   = "\033[0m";   // Fixed escape sequences
constexpr const char* RED     = "\033[31m";
constexpr const char* GREEN   = "\033[32m";
constexpr const char* YELLOW  = "\033[33m";
constexpr const char* BLUE    = "\033[34m";
constexpr const char* CYAN    = "\033[36m";

using Field = std::vector<std::vector<double>>;
using SolverResult = std::pair<int, double>;
using VelocityFields = std::pair<Field, Field>;

// Allocate a 2D field (rows x cols) filled with initial_value
[[nodiscard]] Field createField(int rows, int cols, double initial_value = 0.0) {
    if (rows <= 0 || cols <= 0) throw std::invalid_argument("Field dims must be positive");
    Field f; f.reserve(rows);
    for (int r = 0; r < rows; ++r) f.emplace_back(cols, initial_value);
    return f;
}

// 2D SOR omega estimate from Jacobi spectral radius approximation
[[nodiscard]] constexpr double computeOptimalOmega2D(int nx, int ny) noexcept {
    constexpr double pi = 3.14159265358979323846;
    const double rho_j = 0.5 * (std::cos(pi / (nx + 1)) + std::cos(pi / (ny + 1)));
    const double denom = 1.0 + std::sqrt(std::max(1e-14, 1.0 - rho_j * rho_j));
    return 2.0 / denom; // typically ~1.7–1.95
}

// ---------------- VTK writer ----------------
class VTKWriter {
public:
    static void writeStructuredGrid(const std::string& filename,
                                    const Field& u_center,
                                    const Field& v_center,
                                    const Field& pressure,
                                    int nx, int ny,
                                    double dx, double dy,
                                    double time_value = 0.0) {
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot open VTK: " + filename);

        file << "# vtk DataFile Version 3.0\n";
        file << "Channel Flow Data - Time: " << std::fixed << std::setprecision(6) << time_value << "\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << nx << " " << ny << " 1\n";
        file << "ORIGIN " << dx * 0.5 << " " << dy * 0.5 << " 0.0\n";
        file << "SPACING " << dx << " " << dy << " 1.0\n";
        const int total_points = nx * ny;
        file << "POINT_DATA " << total_points << "\n";

        file << "SCALARS TimeValue double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                file << time_value << "\n";

        file << "VECTORS velocity double\n";
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                file << u_center[j][i] << " " << v_center[j][i] << " 0.0\n";

        file << "SCALARS u_velocity double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                file << u_center[j][i] << "\n";

        file << "SCALARS v_velocity double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                file << v_center[j][i] << "\n";

        file << "SCALARS velocity_magnitude double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                const double mag = std::sqrt(u_center[j][i] * u_center[j][i] + v_center[j][i] * v_center[j][i]);
                file << mag << "\n";
            }
        }

        file << "SCALARS pressure double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j)
            for (int i = 1; i <= nx; ++i)
                file << pressure[j][i] << "\n";

        // simple vorticity estimate with non-square spacing
        file << "SCALARS vorticity double 1\nLOOKUP_TABLE default\n";
        const double idx = 1.0 / dx, idy = 1.0 / dy;
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                double dvdx = 0.0, dudy = 0.0;
                if (i == 1)       dvdx = (v_center[j][i+1] - v_center[j][i]) * idx;
                else if (i == nx) dvdx = (v_center[j][i]   - v_center[j][i-1]) * idx;
                else              dvdx = 0.5 * (v_center[j][i+1] - v_center[j][i-1]) * idx;
                if (j == 1)       dudy = (u_center[j+1][i] - u_center[j][i]) * idy;
                else if (j == ny) dudy = (u_center[j][i]   - u_center[j-1][i]) * idy;
                else              dudy = 0.5 * (u_center[j+1][i] - u_center[j-1][i]) * idy;
                file << (dvdx - dudy) << "\n";
            }
        }
    }

    static std::string generate_filename(const std::string& base_name, int time_step) {
        std::ostringstream oss;
        oss << base_name << "_" << std::setfill('0') << std::setw(6) << time_step << ".vtk";
        return oss.str();
    }

    static void write_paraview_collection(const std::string& collection_filename,
                                          const std::vector<std::string>& vtk_filenames,
                                          const std::vector<double>& time_values) {
        if (vtk_filenames.size() != time_values.size())
            throw std::invalid_argument("VTK filenames and time values must have same size");
        std::ofstream file(collection_filename);
        if (!file) throw std::runtime_error("Cannot open PVD: " + collection_filename);
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <Collection>\n";
        for (size_t i = 0; i < vtk_filenames.size(); ++i) {
            file << "    <DataSet timestep=\"" << std::fixed << std::setprecision(6) << time_values[i]
                 << "\" group=\"\" part=\"0\" file=\"" << vtk_filenames[i] << "\"/>\n";
        }
        file << "  </Collection>\n";
        file << "</VTKFile>\n";
    }

    static void create_output_directory(const std::string& directory_path) {
        std::filesystem::create_directories(directory_path);
    }
};

// ---------------- Channel solver ----------------
class ChannelSolver {
private:
    static constexpr double LENGTH        = 3.0;   // domain length (x)
    static constexpr double HEIGHT        = 1.0;   // domain height (y)
    static constexpr int    NX_INT        = 93;    // interior cells in x
    static constexpr int    NY_INT        = 31;    // interior cells in y
    static constexpr double RE            = 100.0; // Reynolds number
    static constexpr double U_IN          = 1.0;   // inlet bulk velocity
    static constexpr double RHO           = 1.0;   // density
    static constexpr double CFL           = 0.25;   // Courant safety
    static constexpr double T_FINAL       = 10.0;  // final time
    static constexpr double TOL_FACTOR    = 1e-7;  // relative PPE tol
    static constexpr double ABS_TOL       = 1e-10; // absolute PPE tol
    static constexpr int    MAX_SOR_ITERS = 10000; // SOR cap
    static constexpr int    PRINT_INTERVAL= 100;   // log cadence
    static constexpr int    SAVE_INTERVAL = 100;   // VTK cadence

    // Derived
    const double nu;           // kinematic viscosity = U*H/Re
    const int    nx, ny;       // interior counts
    const double dx, dy;       // spacings
    const double omega;        // SOR relaxation
    const double dt;           // timestep
    const int    nsteps;       // total steps

    // Index extents for interior centers
    const int i_min = 1, j_min = 1;
    const int i_max, j_max;

    // Fields (staggered sizes)
    Field p, rhs, res;                 // (ny+2) x (nx+2)
    Field u_star, u, u_cc;             // u: (ny+2) x (nx+1)
    Field v_star, v, v_cc;             // v: (ny+1) x (nx+2)

    // Output bookkeeping
    const std::string out_dir = "vtk_output";
    std::vector<std::string> vtk_files;
    std::vector<double>      vtk_times;

public:
    ChannelSolver()
        : nu(U_IN * HEIGHT / RE)
        , nx(NX_INT), ny(NY_INT)
        , dx(LENGTH / NX_INT), dy(HEIGHT / NY_INT)
        , omega(computeOptimalOmega2D(NX_INT, NY_INT))
        , dt(CFL * std::min(0.25 * std::min(dx, dy) * std::min(dx, dy) / nu,
                            std::min(dx, dy) / std::max(1e-12, U_IN)))
        , nsteps(static_cast<int>(T_FINAL / dt))
        , i_max(nx), j_max(ny)
        , p(createField(j_max + 2, i_max + 2, 0.0))
        , rhs(createField(j_max + 2, i_max + 2, 0.0))
        , res(createField(j_max + 2, i_max + 2, 0.0))
        , u_star(createField(j_max + 2, i_max + 1, 0.0))
        , u(createField(j_max + 2, i_max + 1, 0.0))
        , u_cc(createField(j_max + 2, i_max + 2, 0.0))
        , v_star(createField(j_max + 1, i_max + 2, 0.0))
        , v(createField(j_max + 1, i_max + 2, 0.0))
        , v_cc(createField(j_max + 2, i_max + 2, 0.0)) {
        validate();
        VTKWriter::create_output_directory(out_dir);
        printHeader();
        applyVelocityBC(u, v);          // enforce ICs at boundaries
        interpolateToCellCenters();
        exportVTK(0, 0.0);
    }

    void run() {
        std::cout << GREEN << "Starting simulation..." << RESET << "\n";
        for (int n = 1; n <= nsteps; ++n) {
            const double t = n * dt;

            // Predictor (explicit conv-diff), then impose BCs on u*,v*
            computeTentativeVelocities();
            applyVelocityBC(u_star, v_star);

            // PPE build & solve
            buildRHS();
            auto [iters, final_res] = solvePPE_SOR();

            // Corrector, then re-impose velocity BCs (Dirichlet wins)
            pressureCorrect();
            applyVelocityBC(u, v);

            if (n % PRINT_INTERVAL == 0 || n == nsteps) logStats(n, t, iters, final_res);
            if (n % SAVE_INTERVAL  == 0 || n == nsteps) {
                interpolateToCellCenters();
                exportVTK(n, t);
            }
        }
        writePVD();
        std::cout << GREEN << "Done. Open '" << out_dir << "/channel_flow_animation.pvd' in ParaView." << RESET << "\n";
    }

private:
    void validate() const {
        if (NX_INT <= 0 || NY_INT <= 0 || RE <= 0 || CFL <= 0.0 || CFL >= 1.0 || T_FINAL <= 0.0 || dt <= 0.0)
            throw std::runtime_error("Invalid parameters");
    }

    void printHeader() const {
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6);

        std::cout << CYAN;
        std::cout << "=== Channel Flow Simulation ===\n";
        std::cout << "Domain: " << LENGTH << " x " << HEIGHT << "\n";
        std::cout << "Grid:   " << nx << " x " << ny << "  (dx=" << dx << ", dy=" << dy << ")\n";
        std::cout << "Time:   dt=" << dt << ", steps=" << nsteps << ", T_final=" << T_FINAL << "\n";
        std::cout << "Re= " << RE << ", nu=" << nu << ", CFL=" << CFL << "\n";
        std::cout << "SOR omega=" << omega << ", tol_factor=" << TOL_FACTOR << ", abs_tol=" << ABS_TOL << "\n";
        std::cout << "VTK every " << SAVE_INTERVAL << " steps\n";
        std::cout << "================================\n";
        std::cout << RESET << "\n";
    }

    // ========= Boundary Conditions =========
    void applyVelocityBC(Field& uF, Field& vF) const noexcept {
        // Inlet (x=0): u=U_IN (Dirichlet at u-face), v=0
        for (int j = 1; j <= j_max; ++j) uF[j][0] = U_IN;
        for (int j = 0; j <= j_max; ++j) vF[j][0] = 0.0;

        // Outlet (x=L): zero-gradient for u,v
        for (int j = 1; j <= j_max; ++j) uF[j][i_max]   = uF[j][i_max - 1];
        for (int j = 0; j <= j_max; ++j) vF[j][i_max+1] = vF[j][i_max];

        // Bottom wall (y=0): v=0 at face, u antisymmetric so wall value 0
        for (int i = 1; i <= i_max; ++i) vF[0][i] = 0.0;
        for (int i = 0; i <= i_max; ++i) uF[0][i] = -uF[1][i];

        // Top wall (y=H): v=0 at face, u antisymmetric so wall value 0
        for (int i = 1; i <= i_max; ++i) vF[j_max][i] = 0.0;
        for (int i = 0; i <= i_max; ++i) uF[j_max+1][i] = -uF[j_max][i];
    }

    void applyPressureGhosts(Field& pF) const noexcept {
        // Inlet Neumann: dp/dx = 0
        for (int j = 1; j <= j_max; ++j) pF[j][0] = pF[j][1];
        // Outlet Dirichlet at ghost: p = 0 (reference)
        for (int j = 1; j <= j_max; ++j) pF[j][i_max+1] = 0.0;
        // Walls Neumann: dp/dy = 0
        for (int i = 1; i <= i_max; ++i) {
            pF[0][i]       = pF[1][i];
            pF[j_max+1][i] = pF[j_max][i];
        }
    }

    // ========= Numerics =========
    void computeTentativeVelocities() noexcept {
        const double idx  = 1.0 / dx;
        const double idy  = 1.0 / dy;
        const double idx2 = 1.0 / (dx * dx);
        const double idy2 = 1.0 / (dy * dy);

        // u* on u-faces: j=1..j_max, i=1..i_max-1 (interior faces)
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                // diffusion (2D anisotropic)
                const double lap = (u[j][i+1] - 2.0*u[j][i] + u[j][i-1]) * idx2
                                  + (u[j+1][i] - 2.0*u[j][i] + u[j-1][i]) * idy2;
                // convective fluxes (central)
                const double uE = 0.5 * (u[j][i]   + u[j][i+1]);
                const double uW = 0.5 * (u[j][i-1] + u[j][i]);
                const double conv_x = (uE*uE - uW*uW) * idx;

                const double vN = 0.5 * (v[j][i]   + v[j][i+1]);
                const double vS = 0.5 * (v[j-1][i] + v[j-1][i+1]);
                const double uN = 0.5 * (u[j+1][i] + u[j][i]);
                const double uS = 0.5 * (u[j-1][i] + u[j][i]);
                const double conv_y = (vN*uN - vS*uS) * idy;

                u_star[j][i] = u[j][i] + dt * (nu * lap - conv_x - conv_y);
            }
        }

        // v* on v-faces: j=1..j_max-1, i=1..i_max (interior faces)
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max;   ++i) {
                const double lap = (v[j][i+1] - 2.0*v[j][i] + v[j][i-1]) * idx2
                                  + (v[j+1][i] - 2.0*v[j][i] + v[j-1][i]) * idy2;
                const double vN = 0.5 * (v[j][i]   + v[j+1][i]);
                const double vS = 0.5 * (v[j-1][i] + v[j][i]);
                const double conv_y = (vN*vN - vS*vS) * idy;

                const double uE = 0.5 * (u[j][i]   + u[j+1][i]);
                const double uW = 0.5 * (u[j][i-1] + u[j+1][i-1]);
                const double vE = 0.5 * (v[j][i]   + v[j][i+1]);
                const double vW = 0.5 * (v[j][i-1] + v[j][i]);
                const double conv_x = (uE*vE - uW*vW) * idx;

                v_star[j][i] = v[j][i] + dt * (nu * lap - conv_y - conv_x);
            }
        }
    }

    void buildRHS() noexcept {
        const double idx = 1.0 / dx, idy = 1.0 / dy;
        const double coeff = RHO / dt;
        double max_rhs = 0.0;

        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                rhs[j][i] = coeff * ((u_star[j][i] - u_star[j][i-1]) * idx
                                    + (v_star[j][i] - v_star[j-1][i]) * idy);
                max_rhs = std::max(max_rhs, std::abs(rhs[j][i]));
            }
        }
        // Optional: remove tiny mean to aid convergence
        if (max_rhs > 0) {
            double mean_rhs = 0.0; int cnt = 0;
            for (int j = j_min; j <= j_max; ++j)
                for (int i = i_min; i <= i_max; ++i) { mean_rhs += rhs[j][i]; ++cnt; }
            mean_rhs /= static_cast<double>(cnt);
            for (int j = j_min; j <= j_max; ++j)
                for (int i = i_min; i <= i_max; ++i) rhs[j][i] -= mean_rhs;
        }
    }

    SolverResult solvePPE_SOR() {
        Field p_new = p; // start from previous p
        Field p_prev = p_new; // for GS east/north lookups
        const double idx2 = 1.0 / (dx * dx);
        const double idy2 = 1.0 / (dy * dy);
        const double denom = 2.0 * (idx2 + idy2);

        // tolerance based on RHS magnitude
        double max_rhs = 0.0;
        for (int j = j_min; j <= j_max; ++j)
            for (int i = i_min; i <= i_max; ++i)
                max_rhs = std::max(max_rhs, std::abs(rhs[j][i]));
        const double tol = std::max(TOL_FACTOR * (max_rhs > 0 ? max_rhs : 1.0), ABS_TOL);

        double max_res = tol + 1.0;
        int it = 0;

        while (max_res > tol && it < MAX_SOR_ITERS) {
            ++it;
            p_prev = p_new; // capture state for GS east/north

            // Update interior with SOR (anisotropic 5-pt Laplacian, GS ordering)
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    const double pW = p_new[j][i-1];
                    const double pE = p_prev[j][i+1];
                    const double pS = p_new[j-1][i];
                    const double pN = p_prev[j+1][i];

                    const double sumNbrs = idx2 * (pE + pW) + idy2 * (pN + pS);
                    const double p_gs = (sumNbrs - rhs[j][i]) / denom;
                    p_new[j][i] = (1.0 - omega) * p_new[j][i] + omega * p_gs;
                }
            }
            // Refresh ghosts based on BCs
            applyPressureGhosts(p_new);

            // Residual (infty-norm of PPE)
            max_res = 0.0;
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    const double lap = (p_new[j][i+1] - 2.0*p_new[j][i] + p_new[j][i-1]) * idx2
                                      + (p_new[j+1][i] - 2.0*p_new[j][i] + p_new[j-1][i]) * idy2;
                    res[j][i] = lap - rhs[j][i];
                    max_res = std::max(max_res, std::abs(res[j][i]));
                }
            }
        }
        if (it >= MAX_SOR_ITERS) {
            std::cerr << YELLOW << "Warning: PPE SOR hit max iterations, max_res=" << max_res << RESET << "\n";
        }
        p.swap(p_new);
        return {it, max_res};
    }

    void pressureCorrect() noexcept {
        // u-correction on interior u-faces
        for (int j = j_min; j <= j_max; ++j)
            for (int i = i_min; i <= i_max - 1; ++i)
                u[j][i] = u_star[j][i] - (dt / (RHO * dx)) * (p[j][i+1] - p[j][i]);
        // v-correction on interior v-faces
        for (int j = j_min; j <= j_max - 1; ++j)
            for (int i = i_min; i <= i_max;   ++i)
                v[j][i] = v_star[j][i] - (dt / (RHO * dy)) * (p[j+1][i] - p[j][i]);
    }

    VelocityFields interpolateToCellCenters() noexcept {
        for (int j = j_min; j <= j_max; ++j)
            for (int i = i_min; i <= i_max; ++i)
                u_cc[j][i] = 0.5 * (u[j][i-1] + u[j][i]);
        for (int j = j_min; j <= j_max; ++j)
            for (int i = i_min; i <= i_max; ++i)
                v_cc[j][i] = 0.5 * (v[j-1][i] + v[j][i]);
        return {u_cc, v_cc};
    }

    void logStats(int step, double t, int iters, double max_res) {
        interpolateToCellCenters();
        const double idx = 1.0 / dx, idy = 1.0 / dy;
        double max_div = 0.0, KE = 0.0;
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                const double div = (u[j][i] - u[j][i-1]) * idx + (v[j][i] - v[j-1][i]) * idy;
                max_div = std::max(max_div, std::abs(div));
                KE += 0.5 * (u_cc[j][i]*u_cc[j][i] + v_cc[j][i]*v_cc[j][i]);
            }
        }
        KE /= (nx * ny);
        std::cout << "Step " << std::setw(6) << step << "/" << nsteps
                  << " | t=" << std::fixed << std::setprecision(3) << std::setw(8) << t
                  << " | max(div)=" << std::scientific << std::setprecision(2) << std::setw(10) << max_div
                  << " | avg_KE=" << std::fixed << std::setprecision(6) << std::setw(10) << KE
                  << " | PPE iters=" << std::setw(4) << iters
                  << " | res=" << std::scientific << std::setprecision(2) << std::setw(10) << max_res
                  << "\n";
    }

    void exportVTK(int step, double t) {
        const std::string fname = VTKWriter::generate_filename("channel_flow", step);
        const std::string path  = out_dir + "/" + fname;
        VTKWriter::writeStructuredGrid(path, u_cc, v_cc, p, nx, ny, dx, dy, t);
        if (step % PRINT_INTERVAL == 0 || step == 0) std::cout << BLUE << "Export: " << fname << RESET << "\n";
        vtk_files.push_back(fname);
        vtk_times.push_back(t);
    }

    void writePVD() const {
        VTKWriter::write_paraview_collection(out_dir + "/channel_flow_animation.pvd", vtk_files, vtk_times);
    }
};

} // namespace ChannelFlow

int main() {
    try {
        ChannelFlow::ChannelSolver solver;
        solver.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << ChannelFlow::RED << "Error: " << e.what() << ChannelFlow::RESET << "\n";
        return 1;
    } catch (...) {
        std::cerr << ChannelFlow::RED << "Unknown error\n" << ChannelFlow::RESET;
        return 1;
    }
}