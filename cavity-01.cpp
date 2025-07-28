/**
 * @file    cavity-01.cpp
 * @brief   Fully-explicit, staggered-grid lid-driven cavity solver.
 *
 * @details
 *  • Time scheme:    Forward Euler  
 *  • Diffusion:      2nd-order central  
 *  • Convection:     1st-order central  
 *  • Pressure solver: SOR  
 *
 * @author Tyler Jones
 * @date   2025-07-11
 * @version 3.0
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

int main() {
    
    // ---------------------------------
    // Physical and Numerical Parameters
    // ---------------------------------
    constexpr double L        = 1.0;    // cavity length
    constexpr double H        = 1.0;    // cavity height
    constexpr int    N_int    = 33;     // interior p‑nodes per side
    constexpr double Re       = 100.0;  // Reynolds number
    constexpr double U_lid    = 1.0;    // lid velocity
    constexpr double CFL      = 0.5;    // CFL number
    constexpr double t_final  = 20.0;   // Final time of simulation
    constexpr double tolfac   = 1e-7;   // Poisson tolerance
    constexpr int    max_iter = 10000;  // Maximum SOR iterations

    constexpr int    print_interval = 100;

    // -----------------------------------
    // Derived grid metrics and parameters
    // -----------------------------------
    double nu = U_lid * L / Re;   // Kinematic viscosity
    double h  = L / N_int;        // Spatial step length (dx and dy)

    // Optimal SOR overrelaxation parameter
    constexpr double PI = 3.14159265358979323846;
    double rho_jacobi   = cos(PI / (N_int + 1));
    double omega        = 2.0 / (1.0 + sqrt(1.0 - rho_jacobi * rho_jacobi));

    // Indexing helpers
    int imin = 1;          // first interior index (x)
    int imax = L * N_int;      // last interior index (x)
    int jmin = 1;          // first interior index (y)
    int jmax = H * N_int;      // last interior index (y)

    double dt  = CFL * std::min(0.25 * h * h / nu, h / U_lid);   // Time step
    int nSteps = static_cast<int>(t_final / dt);                 // Number of time steps

    // ---------------
    // Allocate fields 
    // ---------------
    Field p        = makeField(jmax + 2, imax + 2);   // Pressure field
    Field rhs      = makeField(jmax + 2, imax + 2);   // Right-hand side of PPE (source term)
    Field ppe_res  = makeField(jmax + 2, imax + 2);   // local PPE residual
    Field u_star   = makeField(jmax + 2, imax + 1);   // Tentative u-velocity field
    Field u        = makeField(jmax + 2, imax + 1);   // Corrected u-velocity field
    Field u_center = makeField(jmax + 2, imax + 2);   // Interpolated u-velocity field onto cell centers
    Field v_star   = makeField(jmax + 1, imax + 2);   // Tentative v-velocity field
    Field v        = makeField(jmax + 1, imax + 2);   // Corrected v-velocity field
    Field v_center = makeField(jmax + 2, imax + 2);   // Interpolated v-velocity field onto cell centers
    
    std::cout 
        << "Grid: " << N_int << "×" << N_int << " (h=" << h << ")\n"
        << "Time: dt=" << dt << ", nSteps=" << nSteps << ", t_final=" << t_final << "\n"
        << "Re=" << Re << ", nu=" << nu << ", CFL=" << CFL << "\n";

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
        double h_inv  = 1.0 / h;
        
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
        double max_rhs = 0.0;
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

        // Compute tolerance
        for (int j = jmin; j <= jmax; ++j) {
            for (int i = imin; i <= imax; ++i) {
                max_rhs = std::max(max_rhs, std::abs(rhs[j][i]));
            }
        }
        double tol = tolfac * max_rhs;

        // SOR loop
        while (maxRes > tol && iter < max_iter) {
            ++iter;
            p_old.swap(p_new);

            // SOR update into p_new
            for (int j = jmin; j <= jmax; ++j) {
                for (int i = imin; i <= imax; ++i) {
                    int count  = 0;
                    double sum = 0.0;

                    if (i > imin) { sum += p_new[j][i-1];  ++count; } // West neighbor
                    if (i < imax) { sum += p_old[j][i+1];  ++count; } // East neighbor
                    if (j > jmin) { sum += p_new[j-1][i];  ++count; } // South neighbor
                    if (j < jmax) { sum += p_old[j+1][i];  ++count; } // North neighbor

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
                    ppe_res[j][i] = r;
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

    // --------------------------------------------------
    // Subroutine: Interpolate u,v onto the pressure node
    // --------------------------------------------------
    auto interpolateVelocitiesToCellCenters = [&]() -> std::pair<Field, Field> {
        
        // Interpolate u-velocity to cell centers
        for (int j = jmin; j <= jmax; ++j) {
            for (int i = imin; i <= imax; ++i) {
                u_center[j][i] = 0.5 * (u[j][i-1] + u[j][i]);
            }
        }
        
        // Interpolate v-velocity to cell centers  
        for (int j = jmin; j <= jmax; ++j) {
            for (int i = imin; i <= imax; ++i) {
                v_center[j][i] = 0.5 * (v[j-1][i] + v[j][i]);
            }
        }
        
        return {u_center, v_center};
    };

    // ---------------------------------------
    // Subroutine: Compute Statistics to Print
    // ---------------------------------------
    auto computeStats = [&]() -> std::tuple<double, double, double, double, double> {
        double u_max      = 0.0;
        double v_max      = 0.0;
        double div_max    = 0.0;
        double u_sum_sq   = 0.0;
        double v_sum_sq   = 0.0;
        double div_sum_sq = 0.0;
        int u_count       = 0;
        int v_count       = 0;
        int div_count     = 0;
        double h_inv      = 1 / h;
        
        // U-velocity statistics
        for (int j = jmin; j <= jmax; ++j) {
            for (int i = imin; i <= imax-1; ++i) {
                u_max = std::max(u_max, std::abs(u[j][i]));
                u_sum_sq += u[j][i] * u[j][i];
                u_count++;
            }
        }
        double u_rms = std::sqrt(u_sum_sq / u_count);
        
        // V-velocity statistics
        for (int j = jmin; j <= jmax-1; ++j) {
            for (int i = imin; i <= imax; ++i) {
                v_max = std::max(v_max, std::abs(v[j][i]));
                v_sum_sq += v[j][i] * v[j][i];
                v_count++;
            }
        }
        double v_rms = std::sqrt(v_sum_sq / v_count);
        
        // Divergence check
        for (int j = jmin; j <= jmax; ++j) {
            for (int i = imin; i <= imax; ++i) {
                double div = (u[j][i] - u[j][i-1] + v[j][i] - v[j-1][i]) * h_inv;
                div_max = std::max(div_max, std::abs(div));
                div_sum_sq += div * div;
                div_count++;
            }
        }
        // double rms_div = std::sqrt(div_sum_sq / div_count);
        
        return {u_max, v_max, u_rms, v_rms, div_max};
    };

    // ------------------
    // Time marching loop
    // ------------------
    for (int n = 0; n < nSteps; ++n) {
        double current_time = n * dt;
        applyBC();
        predictor();
        auto [ppe_iter, ppe_res] = solvePressure();
        corrector();

        // Print progress
        if (n % print_interval == 0) {
            auto [u_max, v_max, u_rms, v_rms, div_max] = computeStats();
            std::cout << "step " << n << "/" << nSteps
                      << " t:" << std::fixed << std::setprecision(2) << current_time 
                      << " max(|u|):" << std::setprecision(4) << u_max
                      << " max(|v|):" << std::setprecision(4) << v_max 
                      << " max(div): " << std::setprecision(9) << div_max
                      << " PPE_iter=" << ppe_iter << '\n';
        }
    }

    std::cout << "Simulation Complete! \n";
    
    return 0;
}