/**
 * @file    main.cpp
 * @brief   Lid‑Driven Cavity – explicit staggered‑grid solver
 * @author  Tyler Jones
 * @date    2025‑07‑11
 * @version 2.5 - Fixed major bugs causing zero velocity field
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <tuple>

using Field = std::vector<std::vector<double>>;
static Field makeField(int rows, int cols, double val = 0.0) {
    return Field(rows, std::vector<double>(cols, val));
}

struct ResidualData {
    double time;
    int step;
    int ppe_iterations;
    double ppe_residual;
    double u_max;
    double v_max;
    double u_rms;
    double v_rms;
};

void createResultsDirectory() {
    try {
        if (!std::filesystem::exists("results")) {
            std::filesystem::create_directory("results");
            std::cout << "Created results directory" << std::endl;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating results directory: " << e.what() << std::endl;
        std::cerr << "Files will be saved in current directory instead." << std::endl;
    }
}

void writeVTK(const Field& p, const Field& u, const Field& v, 
              int imin, int imax, int jmin, int jmax, 
              double h, int step, double time) {
    
    std::ostringstream filename;
    filename << "results/cavity_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    
    std::ofstream file(filename.str());
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename.str() << " for writing!" << std::endl;

        std::ostringstream fallback_filename;
        fallback_filename << "cavity_" << std::setfill('0') << std::setw(6) << step << ".vtk";
        file.open(fallback_filename.str());
        if (!file.is_open()) {
            std::cerr << "Error: Could not open fallback file " << fallback_filename.str() << " either!" << std::endl;
            return;
        }
        std::cout << "    Wrote VTK file: " << fallback_filename.str() << " (fallback location)" << std::endl;
    } else {
        std::cout << "    Wrote VTK file: " << filename.str() << std::endl;
    }
    
    int i_start = 0;           
    int i_end = imax + 1;      
    int j_start = 0;           
    int j_end = jmax + 1;      
    
    int nx = i_end - i_start + 1;
    int ny = j_end - j_start + 1;
    int npoints = nx * ny;
    
    file << "# vtk DataFile Version 3.0\n";
    file << "Lid-driven cavity flow at t=" << std::fixed << std::setprecision(4) << time << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << nx << " " << ny << " 1\n";
    file << "POINTS " << npoints << " float\n";
    
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            double x = (i - 0.5) * h;  // Cell center x-coordinate
            double y = (j - 0.5) * h;  // Cell center y-coordinate
            file << std::fixed << std::setprecision(6) << x << " " << y << " 0.0\n";
        }
    }
    
    file << "POINT_DATA " << npoints << "\n";
    
    file << "SCALARS pressure float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            file << std::fixed << std::setprecision(6) << p[j][i] << "\n";
        }
    }
    
    file << "VECTORS velocity float\n";
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            double u_center, v_center;
            
            if (i == 0) {
                u_center = 0.0;  // No-slip wall
            } else if (i == imax + 1) {
                u_center = 0.0;  // No-slip wall
            } else {
                u_center = 0.5 * (u[j][i] + u[j][i-1]); // Interpolated
            }
            
            if (j == 0) {
                v_center = 0.0;  // No-slip wall
            } else if (j == jmax + 1) {
                v_center = 0.0;  // No vertical velocity at lid
            } else {
                v_center = 0.5 * (v[j][i] + v[j-1][i]); // Interpolated
            }
            
            file << std::fixed << std::setprecision(6) 
                 << u_center << " " << v_center << " 0.0\n";
        }
    }
    
    file << "SCALARS velocity_magnitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            double u_center, v_center;
            
            if (i == 0 || i == imax + 1) {
                u_center = 0.0;  // No-slip walls
            } else {
                u_center = 0.5 * (u[j][i] + u[j][i-1]);
            }
            if (j == jmax + 1 && i > 0 && i < imax + 1) {
                u_center = 1.0;  // Lid velocity (U_lid)
            }
            
            if (j == 0) {
                v_center = 0.0;  // South wall
            } else if (j == jmax + 1) {
                v_center = 0.0;  // North wall (lid)
            } else {
                v_center = 0.5 * (v[j][i] + v[j-1][i]);
            }
            
            double mag = std::sqrt(u_center*u_center + v_center*v_center);
            file << std::fixed << std::setprecision(6) << mag << "\n";
        }
    }
    
    file << "SCALARS vorticity float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            double vorticity = 0.0;
            
            if (i > 0 && i < imax + 1 && j > 0 && j < jmax + 1) {
                // Compute vorticity: ∂v/∂x - ∂u/∂y using central differences
                double dvdx = (0.5 * (v[j][i+1] + v[j-1][i+1]) - 0.5 * (v[j][i-1] + v[j-1][i-1])) / (2.0 * h);
                double dudy = (0.5 * (u[j+1][i] + u[j+1][i-1]) - 0.5 * (u[j-1][i] + u[j-1][i-1])) / (2.0 * h);
                vorticity = dvdx - dudy;
            }
            
            file << std::fixed << std::setprecision(6) << vorticity << "\n";
        }
    }
    
    file << "SCALARS boundary_id float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            double boundary_id = 0.0;  // Interior
            
            if (i == 0) boundary_id = 1.0;           // West wall
            else if (i == imax + 1) boundary_id = 2.0; // East wall
            else if (j == 0) boundary_id = 3.0;        // South wall
            else if (j == jmax + 1) boundary_id = 4.0; // North wall (lid)
            
            file << std::fixed << std::setprecision(6) << boundary_id << "\n";
        }
    }
    
    file.close();
}

void writeResidualData(const std::vector<ResidualData>& residuals, const std::string& filename) {
    std::string filepath = "results/" + filename;
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing!" << std::endl;

        file.open(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open fallback file " << filename << " either!" << std::endl;
            return;
        }
        std::cout << "Wrote residual data to: " << filename << " (fallback location)" << std::endl;
        filepath = filename;
    } else {
        std::cout << "Wrote residual data to: " << filepath << std::endl;
    }
    
    file << "# Residual history for lid-driven cavity simulation\n";
    file << "# Columns: Step Time PPE_Iterations PPE_Residual U_max V_max U_rms V_rms\n";
    file << std::fixed << std::setprecision(8);
    
    for (const auto& data : residuals) {
        file << data.step << " " 
             << data.time << " " 
             << data.ppe_iterations << " " 
             << data.ppe_residual << " " 
             << data.u_max << " " 
             << data.v_max << " " 
             << data.u_rms << " " 
             << data.v_rms << "\n";
    }
    
    file.close();
}

int main() {
    createResultsDirectory();
    
    // ---------------------------------
    // Physical and Numerical Parameters
    // ---------------------------------
    constexpr double L        = 1.0;    // cavity length (m)
    constexpr int    N_int    = 32;     // interior p‑nodes per side
    constexpr double Re       = 100.0;  // Reynolds number
    constexpr double U_lid    = 1.0;    // lid velocity (m/s)
    constexpr double CFL      = 0.5;    // CFL number
    constexpr double t_final  = 20.0;   // end time 
    constexpr double tol_P    = 1e-7;   // Poisson tolerance
    constexpr int    max_iter = 10000;  // Maximum SOR iterations
    
    // -------------------------
    // Output Control Parameters
    // -------------------------
    constexpr int vtk_interval      = 100;   // Write VTK every N steps
    constexpr int residual_interval = 100;   // Record residuals every N steps
    constexpr int print_interval    = 1000;   // Print progress every N steps

    // -----------------------------------
    // Derived grid metrics and parameters
    // -----------------------------------
    double nu     = U_lid * L / Re;   // Kinematic viscosity
    double h      = L / N_int;        // Spatial step length (dx and dy)

    // Optimal SOR overrelaxation parameter
    double rho_jacobi = 0.5 * (cos(3.1415 / (N_int + 1)) + cos(3.1415 / (N_int + 1)));
    double omega      = 2.0 / (1.0 + sqrt(1.0 - rho_jacobi * rho_jacobi)); 

    // Index helpers
    int imin = 1;          // first interior index (x)
    int imax = N_int;      // last interior index (x)
    int jmin = 1;          // first interior index (y)
    int jmax = N_int;      // last interior index (y)

    double dt  = CFL * std::min(0.25 * h * h / nu, h / U_lid);   // Time step
    int nSteps = static_cast<int>(t_final / dt);                 // Number of time steps
    
    // ---------------------------------
    // Data storage for history tracking
    // ---------------------------------
    std::vector<ResidualData> residual_history;
    residual_history.reserve(nSteps / residual_interval + 1);

    // ---------------
    // Allocate fields 
    // ---------------
    Field p      = makeField(jmax + 2, imax + 2);   // Pressure field
    Field u_star = makeField(jmax + 2, imax + 1);   // Tentative u-velocity field
    Field u      = makeField(jmax + 2, imax + 1);   // Corrected u-velocity field
    Field v_star = makeField(jmax + 1, imax + 2);   // Tentative v-velocity field
    Field v      = makeField(jmax + 1, imax + 2);   // Corrected v-velocity field
    Field rhs    = makeField(jmax + 2, imax + 2);   // Right-hand side of PPE (source term)
    
    std::cout 
        << "Grid: " << N_int << "×" << N_int << " (h=" << h << ")\n"
        << "Time: dt=" << dt << ", nSteps=" << nSteps << ", t_final=" << t_final << "\n"
        << "Re=" << Re << ", nu=" << nu << ", CFL=" << CFL << "\n"
        << "Output: VTK every " << vtk_interval << " steps, residuals every " 
        << residual_interval << " steps\n";

    // ---------------------------------------------
    // Subroutine: Apply Neumann Boundary Conditions
    // ---------------------------------------------
    auto applyBC = [&]() {
        // North lid (moving wall)
        for (int i = 0; i <= imax; ++i) {
            u[jmax+1][i] = 2.0 * U_lid - u[jmax][i]; 
        }

        // South wall (no-slip)
        for (int i = 0; i <= imax; ++i) {
            u[0][i] = -u[1][i];  
        }

        // East wall (no-slip)
        for (int j = 0; j <= jmax; ++j) {
            v[j][imax+1] = -v[j][imax];  
        }

        // West wall (no-slip)
        for (int j = 0; j <= jmax; ++j) {
            v[j][0] = -v[j][1];  
        }
    };

    // -----------------------------------------
    // Subroutine: Predictor step compute u*, v*
    // -----------------------------------------
    auto predictor = [&]() {
        double h2_inv = 1.0 / (h * h);
        double h_inv = 1.0 / h;
        
        // Compute u* from momentum equation (neglect pressure gradient)
        for (int j = jmin; j <= jmax; ++j) {  
            for (int i = imin; i <= imax-1; ++i) {
                // Viscous term: ν∇²u
                double Term1_U = nu * ((u[j][i+1] - 2.0*u[j][i] + u[j][i-1]) * h2_inv +
                                       (u[j+1][i] - 2.0*u[j][i] + u[j-1][i]) * h2_inv);
                
                // Convective term: ∂(u²)/∂x
                double fe      = 0.5 * (u[j][i] + u[j][i+1]);  
                double fw      = 0.5 * (u[j][i-1] + u[j][i]);  
                double Term2_U = (fe*fe - fw*fw) * h_inv;
                
                // Cross‐stream convective flux ∂(vu)/∂y
                double v_n     = 0.5*(v[j][i] + v[j][i+1]);    
                double v_s     = 0.5*(v[j-1][i] + v[j-1][i+1]); 
                double u_n     = 0.5*(u[j+1][i] + u[j][i]);    
                double u_s     = 0.5*(u[j-1][i] + u[j][i]);    
                double Term3_U = (v_n*u_n - v_s*u_s) * h_inv;
                
                // Explicit Euler update
                u_star[j][i] = u[j][i] + dt * (Term1_U - Term2_U - Term3_U);
            }
        }

        // Compute v* from momentum equation (neglect pressure gradient)
        for (int j = jmin; j <= jmax-1; ++j) {
            for (int i = imin; i <= imax; ++i) {
                // Viscous term: ν ∇²v
                double Term1_V = nu * ((v[j][i+1] - 2.0*v[j][i] + v[j][i-1]) * h2_inv +
                                       (v[j+1][i] - 2.0*v[j][i] + v[j-1][i]) * h2_inv);

                // Convective term: ∂(v²)/∂y
                double fn      = 0.5*(v[j][i] + v[j+1][i]);
                double fs      = 0.5*(v[j-1][i] + v[j][i]);
                double Term2_V = (fn*fn - fs*fs) * h_inv;

                // Cross‐stream convective flux ∂(uv)/∂x
                double u_e     = 0.5*(u[j][i] + u[j+1][i]);     
                double u_w     = 0.5*(u[j][i-1] + u[j+1][i-1]); 
                double v_e     = 0.5*(v[j][i] + v[j][i+1]);     
                double v_w     = 0.5*(v[j][i-1] + v[j][i]);     
                double Term3_V = (u_e*v_e - u_w*v_w) * h_inv;

                // Explicit Euler update
                v_star[j][i] = v[j][i] + dt * (Term1_V - Term2_V - Term3_V);
            }
        }
    };   
      
    // -------------------------------------------------------------
    // Subroutine: Solve pressure-Poisson equation via SOR algorithm
    // -------------------------------------------------------------
    auto solvePressure = [&]() -> std::pair<int,double> {
        Field p_old   = makeField(jmax + 2, imax + 2);   // Last pressure field
        Field p_new   = makeField(jmax + 2, imax + 2);   // Next pressure field
        double tol    = tol_P;                           // Residual tolerance  
        double maxRes = 1.0;                             // prevent short circuit
        int iter      = 0;                               // Initialize PPE iterations
        double h1_inv = 1.0 / h;
        double h2_inv = 1.0 / (h * h);

        // Precompute source term: RHS = g = [div(u*) + div(v*)]/dt
        for(int j=jmin; j<=jmax; ++j)
            for(int i=imin; i<=imax; ++i)
                rhs[j][i] = (1.0/dt) * (
                            (u_star[j][i] - u_star[j][i-1]) * h1_inv +
                            (v_star[j][i] - v_star[j-1][i]) * h1_inv );

        // SOR loop
        while (maxRes > tol && iter < max_iter) {
            ++iter;
            p_old.swap(p_new);

            // SOR update into p_new
            for (int j = jmin; j <= jmax; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    int count  = 0;
                    double sum = 0.0;

                    // West neighbor
                    if (i > imin) { sum += p_new[j][i-1];  ++count; }
                    // East neighbor
                    if (i < imax) { sum += p_old[j][i+1];  ++count; }
                    // South neighbor
                    if (j > jmin) { sum += p_new[j-1][i];  ++count; }
                    // North neighbor
                    if (j < jmax) { sum += p_old[j+1][i];  ++count; }

                    double p_guess = (sum - rhs[j][i] / h2_inv) / count;         // Gauss–Seidel/Jacobi estimate
                    p_new[j][i]    = (1.0 - omega)*p_old[j][i] + omega*p_guess;  // SOR upadte
                }
            }

            // Compute residual norm
            maxRes = 0.0;
            for (int j = jmin; j <= jmax; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    double lap = 0.0;
                    int count  = 0;

                    // Compute discrete laplacian
                    if (i > imin) { lap += (p_new[j][i-1] - p_new[j][i]) * h2_inv; ++count; }
                    if (i < imax) { lap += (p_new[j][i+1] - p_new[j][i]) * h2_inv; ++count; }
                    if (j > jmin) { lap += (p_new[j-1][i] - p_new[j][i]) * h2_inv; ++count; }
                    if (j < jmax) { lap += (p_new[j+1][i] - p_new[j][i]) * h2_inv; ++count; }

                    double r = lap - rhs[j][i];                 // Local defect
                    maxRes   = std::max(maxRes, std::abs(r));   // Inf norm
                }
            }

            // Convergence check
            if (maxRes < tol) break;
        }

        // Update pressure
        p = p_new;

        return {iter, maxRes};
    };

    // --------------------------------------
    // Subroutine: Corrector step compute u,v
    // --------------------------------------
    auto corrector = [&]() {
        double dt_h_inv = dt / h;
        
        // Correct u-velocities
        for (int j = jmin; j <= jmax; ++j) { 
            for (int i = imin; i <= imax-1; ++i) {
                u[j][i] = u_star[j][i] - dt_h_inv * (p[j][i+1] - p[j][i]);
            }
        }
        
        // Correct v-velocities
        for (int j = jmin; j <= jmax-1; ++j) {
            for (int i = imin; i <= imax; ++i) {
                v[j][i] = v_star[j][i] - dt_h_inv * (p[j+1][i] - p[j][i]);
            }
        }
    };

    //── Function to compute velocity statistics ───────────────────────────
    auto computeVelocityStats = [&]() -> std::tuple<double, double, double, double> {
        double u_max    = 0.0;
        double v_max    = 0.0;
        double u_sum_sq = 0.0;
        double v_sum_sq = 0.0;
        int u_count     = 0;
        int v_count     = 0;
        
        // U-velocity statistics
        for (int j = jmin; j <= jmax; ++j) {
            for (int i = imin; i <= imax-1; ++i) {
                u_max = std::max(u_max, std::abs(u[j][i]));
                u_sum_sq += u[j][i] * u[j][i];
                u_count++;
            }
        }
        
        // V-velocity statistics
        for (int j = jmin; j <= jmax-1; ++j) {
            for (int i = imin; i <= imax; ++i) {
                v_max = std::max(v_max, std::abs(v[j][i]));
                v_sum_sq += v[j][i] * v[j][i];
                v_count++;
            }
        }
        
        double u_rms = std::sqrt(u_sum_sq / u_count);
        double v_rms = std::sqrt(v_sum_sq / v_count);
        
        return {u_max, v_max, u_rms, v_rms};
    };

    // ------------------
    // Time marching loop
    // ------------------
    std::cout << ">> Entering time loop\n";
    writeVTK(p, u, v, imin, imax, jmin, jmax, h, 0, 0.0);
    
    for (int n = 0; n < nSteps; ++n) {
        double current_time = n * dt;
        applyBC();
        predictor();
        auto [ppe_iter, ppe_res] = solvePressure();
        corrector();

        // Record residual data
        if (n % residual_interval == 0) {
            auto [u_max, v_max, u_rms, v_rms] = computeVelocityStats();
            residual_history.push_back({current_time, n, ppe_iter, ppe_res, u_max, v_max, u_rms, v_rms});
        }

        // Write VTK output
        if (n % vtk_interval == 0 && n > 0) {
            writeVTK(p, u, v, imin, imax, jmin, jmax, h, n, current_time);
        }

        // Print progress
        if (n % print_interval == 0) {
            auto [u_max, v_max, u_rms, v_rms] = computeVelocityStats();
            std::cout << "step " << n << "/" << nSteps
                      << "  t=" << std::fixed << std::setprecision(4) << current_time 
                      << "  |u|_max=" << std::setprecision(6) << u_max
                      << "  |v|_max=" << v_max 
                      << "  PPE_iter=" << ppe_iter << '\n';
        }
    }

    // ------------------------------------
    // Post-Processing and Final Statistics
    // ------------------------------------
    writeVTK(p, u, v, imin, imax, jmin, jmax, h, nSteps, t_final); // Write final VTK file
    writeResidualData(residual_history, "residuals.dat"); // Export residual history
    std::cout << "Done. Steps: " << nSteps << '\n';
    
    // Final statistics and centerline comparison
    auto [u_max_final, v_max_final, u_rms_final, v_rms_final] = computeVelocityStats();
    int ic = imax/2;
    
    std::cout << "\nFinal statistics:\n";
    std::cout << "  Max velocities: |u|_max=" << u_max_final << ", |v|_max=" << v_max_final << std::endl;
    std::cout << "  RMS velocities: u_rms=" << u_rms_final << ", v_rms=" << v_rms_final << std::endl;
    
    std::cout << "\nCenterline velocities for comparison with Ghia et al.:\n";
    std::cout << "Vertical centerline (u-velocity at x=0.5):\n";
    for (int j = jmin; j <= jmax; j += N_int/8) {
        double y = (j - 0.5) * h;
        double u_center = 0.5 * (u[j][ic] + u[j][ic-1]);
        std::cout << "  y=" << std::fixed << std::setprecision(4) << y 
                  << ", u=" << std::setprecision(6) << u_center << std::endl;
    }
    
    std::cout << "\nHorizontal centerline (v-velocity at y=0.5):\n";
    int jc = jmax/2;
    for (int i = imin; i <= imax; i += N_int/8) {
        double x = (i - 0.5) * h;
        double v_center = 0.5 * (v[jc][i] + v[jc-1][i]);
        std::cout << "  x=" << std::fixed << std::setprecision(4) << x 
                  << ", v=" << std::setprecision(6) << v_center << std::endl;
    }
    
    return 0;
}