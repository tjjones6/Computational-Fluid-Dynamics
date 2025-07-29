/**
 * @file    cavity-03.cpp
 * @brief   Fully-explicit, staggered-grid lid-driven cavity solver with modern C++ practices.
 *
 * @details
 *  • Time scheme:    Forward Euler  
 *  • Diffusion:      2nd-order central  
 *  • Convection:     1st-order central  
 *  • Pressure solver: SOR  
 *
 * @author Tyler Jones
 * @date   2025-07-28
 * @version 4.0
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
#include <cassert>
#include <stdexcept>

constexpr char RESET[]   = "\033[0m";
constexpr char RED[]     = "\033[31m"; 
constexpr char GREEN[]   = "\033[32m";  
constexpr char YELLOW[]  = "\033[33m";  
constexpr char BLUE[]    = "\033[34m"; 
constexpr char MAGENTA[] = "\033[35m"; 
constexpr char CYAN[]    = "\033[36m";

namespace CavityFlow {

using Field = std::vector<std::vector<double>>;
using SolverResult = std::pair<int, double>;
using VelocityFields = std::pair<Field, Field>;

/**
 * @brief Create a 2D field initialized with a given value
 * @param rows Number of rows
 * @param cols Number of columns  
 * @param initial_value Initial value for all elements
 * @return Initialized 2D field
 */
[[nodiscard]] Field create_field(int rows, int cols, double initial_value = 0.0) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Field dimensions must be positive");
    }
    
    Field field;
    field.reserve(rows);
    for (int i = 0; i < rows; ++i) {
        field.emplace_back(cols, initial_value);
    }
    return field;
}

/**
 * @brief Compute optimal SOR relaxation parameter
 * @param n_interior Number of interior grid points per dimension
 * @return Optimal omega value
 */
[[nodiscard]] constexpr double compute_optimal_omega(int n_interior) noexcept {
    constexpr double pi = 3.14159265358979323846;
    const auto rho_jacobi = std::cos(pi / (n_interior + 1));
    return 2.0 / (1.0 + std::sqrt(1.0 - rho_jacobi * rho_jacobi));
}

/**
 * @brief Main solver class for lid-driven cavity flow
 */
class CavitySolver {
private:
    // Physical and numerical parameters
    static constexpr double cavity_length = 1.0;
    static constexpr double cavity_height = 1.0;
    static constexpr int n_interior = 32;
    static constexpr double reynolds_number = 100.0;
    static constexpr double lid_velocity = 1.0;
    static constexpr double cfl_number = 0.5;
    static constexpr double final_time = 20.0;
    static constexpr double tolerance_factor = 1e-7;
    static constexpr int max_sor_iterations = 10000;
    static constexpr int print_interval = 100;

    // Derived parameters
    const double kinematic_viscosity;
    const double grid_spacing;
    const double optimal_omega;
    const double time_step;
    const int total_time_steps;
    
    // Grid indexing
    const int i_min = 1;
    const int i_max;
    const int j_min = 1; 
    const int j_max;

    // Flow fields
    Field pressure;
    Field pressure_rhs;
    Field pressure_residual;
    Field u_tentative;
    Field u_velocity;
    Field u_cell_center;
    Field v_tentative;
    Field v_velocity;
    Field v_cell_center;

public:
    /**
     * @brief Constructor - initializes solver parameters and allocates memory
     */
    CavitySolver() 
        : kinematic_viscosity(lid_velocity * cavity_length / reynolds_number)
        , grid_spacing(cavity_length / n_interior)
        , optimal_omega(compute_optimal_omega(n_interior))
        , time_step(cfl_number * std::min(0.25 * grid_spacing * grid_spacing / kinematic_viscosity, 
                                         grid_spacing / lid_velocity))
        , total_time_steps(static_cast<int>(final_time / time_step))
        , i_max(static_cast<int>(cavity_length * n_interior))
        , j_max(static_cast<int>(cavity_height * n_interior))
    {
        validate_parameters();
        allocate_fields();
        print_simulation_info();
    }

    /**
     * @brief Run the complete simulation
     */
    void run() {
        std::cout << GREEN
                  <<"Starting simulation...\n"
                  << RESET;
        
        for (int time_step_idx = 0; time_step_idx < total_time_steps; ++time_step_idx) {
            const auto current_time = time_step_idx * time_step;
            
            apply_boundary_conditions();
            predictor_step();
            const auto [sor_iterations, residual] = solve_pressure_poisson();
            corrector_step();

            if (time_step_idx % print_interval == 0) {
                print_statistics(time_step_idx, current_time, sor_iterations);
            }
        }
        
        std::cout << GREEN
                  <<"Simulation completed successfully!\n"
                  << RESET;
    }

private:
    /**
     * @brief Validate input parameters for physical consistency
     */
    void validate_parameters() const {
        static_assert(n_interior > 0, "Grid size must be positive");
        static_assert(reynolds_number > 0, "Reynolds number must be positive");
        static_assert(cfl_number > 0 && cfl_number < 1, "CFL number must be between 0 and 1");
        static_assert(final_time > 0, "Simulation time must be positive");
        
        if (time_step <= 0) {
            throw std::runtime_error("Computed time step is non-positive");
        }
    }

    /**
     * @brief Allocate memory for all flow fields
     */
    void allocate_fields() {
        try {
            pressure = create_field(j_max + 2, i_max + 2);
            pressure_rhs = create_field(j_max + 2, i_max + 2);
            pressure_residual = create_field(j_max + 2, i_max + 2);
            u_tentative = create_field(j_max + 2, i_max + 1);
            u_velocity = create_field(j_max + 2, i_max + 1);
            u_cell_center = create_field(j_max + 2, i_max + 2);
            v_tentative = create_field(j_max + 1, i_max + 2);
            v_velocity = create_field(j_max + 1, i_max + 2);
            v_cell_center = create_field(j_max + 2, i_max + 2);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate memory for flow fields: " + std::string(e.what()));
        }
    }

    /**
     * @brief Print simulation parameters and setup information
     */
    void print_simulation_info() const {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << BLUE
                  << "=== Lid-Driven Cavity Flow Simulation ===\n"
                  << "Grid: " << n_interior << "×" << n_interior 
                  << " (spacing=" << grid_spacing << ")\n"
                  << "Time: dt=" << time_step 
                  << ", steps=" << total_time_steps 
                  << ", final_time=" << final_time << "\n"
                  << "Reynolds=" << reynolds_number 
                  << ", viscosity=" << kinematic_viscosity 
                  << ", CFL=" << cfl_number << "\n"
                  << "SOR omega=" << optimal_omega << "\n"
                  << "==========================================\n"
                  << RESET << "\n";
    }

    /**
     * @brief Apply boundary conditions using ghost cell approach
     */
    void apply_boundary_conditions() noexcept {
        // North lid (moving wall) - u velocity
        for (int i = 0; i <= i_max; ++i) {
            u_velocity[j_max + 1][i] = 2.0 * lid_velocity - u_velocity[j_max][i];
        }

        // South wall (no-slip) - u velocity  
        for (int i = 0; i <= i_max; ++i) {
            u_velocity[0][i] = -u_velocity[1][i];
        }

        // East wall (no-slip) - v velocity
        for (int j = 0; j <= j_max; ++j) {
            v_velocity[j][i_max + 1] = -v_velocity[j][i_max];
        }

        // West wall (no-slip) - v velocity
        for (int j = 0; j <= j_max; ++j) {
            v_velocity[j][0] = -v_velocity[j][1];
        }
    }

    /**
     * @brief Predictor step: compute tentative velocities without pressure gradient
     */
    void predictor_step() noexcept {
        const auto grid_spacing_inv = 1.0 / grid_spacing;
        const auto grid_spacing_sq_inv = 1.0 / (grid_spacing * grid_spacing);
        
        // Compute u* (tentative u-velocity)
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                // Viscous diffusion: ν∇²u
                const auto viscous_term = kinematic_viscosity * (
                    (u_velocity[j][i+1] - 2.0*u_velocity[j][i] + u_velocity[j][i-1]) * grid_spacing_sq_inv +
                    (u_velocity[j+1][i] - 2.0*u_velocity[j][i] + u_velocity[j-1][i]) * grid_spacing_sq_inv
                );
                
                // Convective flux: ∂(u²)/∂x
                const auto u_east = 0.5 * (u_velocity[j][i] + u_velocity[j][i+1]);
                const auto u_west = 0.5 * (u_velocity[j][i-1] + u_velocity[j][i]);
                const auto convection_x = (u_east*u_east - u_west*u_west) * grid_spacing_inv;
                
                // Cross-stream convective flux: ∂(vu)/∂y
                const auto v_north = 0.5 * (v_velocity[j][i] + v_velocity[j][i+1]);
                const auto v_south = 0.5 * (v_velocity[j-1][i] + v_velocity[j-1][i+1]);
                const auto u_north = 0.5 * (u_velocity[j+1][i] + u_velocity[j][i]);
                const auto u_south = 0.5 * (u_velocity[j-1][i] + u_velocity[j][i]);
                const auto convection_y = (v_north*u_north - v_south*u_south) * grid_spacing_inv;
                
                // Forward Euler update
                u_tentative[j][i] = u_velocity[j][i] + time_step * (viscous_term - convection_x - convection_y);
            }
        }

        // Compute v* (tentative v-velocity)
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                // Viscous diffusion: ν∇²v
                const auto viscous_term = kinematic_viscosity * (
                    (v_velocity[j][i+1] - 2.0*v_velocity[j][i] + v_velocity[j][i-1]) * grid_spacing_sq_inv +
                    (v_velocity[j+1][i] - 2.0*v_velocity[j][i] + v_velocity[j-1][i]) * grid_spacing_sq_inv
                );

                // Convective flux: ∂(v²)/∂y
                const auto v_north = 0.5 * (v_velocity[j][i] + v_velocity[j+1][i]);
                const auto v_south = 0.5 * (v_velocity[j-1][i] + v_velocity[j][i]);
                const auto convection_y = (v_north*v_north - v_south*v_south) * grid_spacing_inv;

                // Cross-stream convective flux: ∂(uv)/∂x
                const auto u_east = 0.5 * (u_velocity[j][i] + u_velocity[j+1][i]);
                const auto u_west = 0.5 * (u_velocity[j][i-1] + u_velocity[j+1][i-1]);
                const auto v_east = 0.5 * (v_velocity[j][i] + v_velocity[j][i+1]);
                const auto v_west = 0.5 * (v_velocity[j][i-1] + v_velocity[j][i]);
                const auto convection_x = (u_east*v_east - u_west*v_west) * grid_spacing_inv;

                // Forward Euler update
                v_tentative[j][i] = v_velocity[j][i] + time_step * (viscous_term - convection_y - convection_x);
            }
        }
    }

    /**
     * @brief Solve pressure Poisson equation using SOR iteration
     * @return Pair of (iterations, final_residual)
     */
    [[nodiscard]] SolverResult solve_pressure_poisson() {
        auto pressure_old = create_field(j_max + 2, i_max + 2);
        auto pressure_new = create_field(j_max + 2, i_max + 2);
        
        const auto grid_spacing_inv = 1.0 / grid_spacing;
        const auto grid_spacing_sq_inv = 1.0 / (grid_spacing * grid_spacing);
        const auto time_step_inv = 1.0 / time_step;
        
        auto max_rhs = 0.0;
        auto max_residual = 1.0;  // Initialize > 0 to enter loop
        auto iteration_count = 0;

        // Compute source term: RHS = ∇·u*/dt
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                pressure_rhs[j][i] = time_step_inv * (
                    (u_tentative[j][i] - u_tentative[j][i-1]) * grid_spacing_inv +
                    (v_tentative[j][i] - v_tentative[j-1][i]) * grid_spacing_inv
                );
                max_rhs = std::max(max_rhs, std::abs(pressure_rhs[j][i]));
            }
        }

        const auto tolerance = tolerance_factor * max_rhs;

        // SOR iteration loop
        while (max_residual > tolerance && iteration_count < max_sor_iterations) {
            ++iteration_count;
            pressure_old.swap(pressure_new);

            // SOR update sweep
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    auto neighbor_count = 0;
                    auto neighbor_sum = 0.0;

                    // Collect neighboring pressure values
                    if (i > i_min) { neighbor_sum += pressure_new[j][i-1]; ++neighbor_count; }
                    if (i < i_max) { neighbor_sum += pressure_old[j][i+1]; ++neighbor_count; }
                    if (j > j_min) { neighbor_sum += pressure_new[j-1][i]; ++neighbor_count; }
                    if (j < j_max) { neighbor_sum += pressure_old[j+1][i]; ++neighbor_count; }

                    // Gauss-Seidel estimate
                    const auto pressure_estimate = (neighbor_sum - pressure_rhs[j][i] / grid_spacing_sq_inv) / neighbor_count;
                    
                    // SOR update
                    pressure_new[j][i] = (1.0 - optimal_omega) * pressure_old[j][i] + optimal_omega * pressure_estimate;
                }
            }

            // Compute residual norm every iteration
            max_residual = 0.0;
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    auto laplacian = 0.0;
                    auto neighbor_count = 0;

                    // Compute discrete Laplacian
                    if (i > i_min) { laplacian += (pressure_new[j][i-1] - pressure_new[j][i]) * grid_spacing_sq_inv; ++neighbor_count; }
                    if (i < i_max) { laplacian += (pressure_new[j][i+1] - pressure_new[j][i]) * grid_spacing_sq_inv; ++neighbor_count; }
                    if (j > j_min) { laplacian += (pressure_new[j-1][i] - pressure_new[j][i]) * grid_spacing_sq_inv; ++neighbor_count; }
                    if (j < j_max) { laplacian += (pressure_new[j+1][i] - pressure_new[j][i]) * grid_spacing_sq_inv; ++neighbor_count; }

                    const auto local_residual = laplacian - pressure_rhs[j][i];
                    pressure_residual[j][i] = local_residual;
                    max_residual = std::max(max_residual, std::abs(local_residual));
                }
            }
        }

        // Check for convergence issues
        if (iteration_count >= max_sor_iterations) {
            std::cerr << "Warning: SOR solver did not converge in " << max_sor_iterations 
                      << " iterations. Final residual: " << max_residual << "\n";
        }

        // Update global pressure field
        pressure = std::move(pressure_new);

        return {iteration_count, max_residual};
    }

    /**
     * @brief Corrector step: apply pressure gradient to get final velocities
     */
    void corrector_step() noexcept {
        const auto dt_over_h = time_step / grid_spacing;
        
        // Correct u-velocities
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                u_velocity[j][i] = u_tentative[j][i] - dt_over_h * (pressure[j][i+1] - pressure[j][i]);
            }
        }
        
        // Correct v-velocities  
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                v_velocity[j][i] = v_tentative[j][i] - dt_over_h * (pressure[j+1][i] - pressure[j][i]);
            }
        }
    }

    /**
     * @brief Interpolate staggered velocities to cell centers
     * @return Pair of (u_center, v_center) fields
     */
    [[nodiscard]] VelocityFields interpolate_to_cell_centers() noexcept {
        // Interpolate u-velocity to cell centers
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                u_cell_center[j][i] = 0.5 * (u_velocity[j][i-1] + u_velocity[j][i]);
            }
        }
        
        // Interpolate v-velocity to cell centers
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                v_cell_center[j][i] = 0.5 * (v_velocity[j-1][i] + v_velocity[j][i]);
            }
        }
        
        return {u_cell_center, v_cell_center};
    }

    /**
     * @brief Compute and print flow statistics
     * @param step_number Current time step
     * @param current_time Current simulation time
     * @param sor_iterations Number of SOR iterations used
     */
    void print_statistics(int step_number, double current_time, int sor_iterations) {
        const auto [u_center, v_center] = interpolate_to_cell_centers();
        
        auto max_u_velocity = 0.0;
        auto max_v_velocity = 0.0;
        auto max_divergence = 0.0;
        auto total_kinetic_energy = 0.0;
        
        const auto grid_spacing_inv = 1.0 / grid_spacing;

        // Compute velocity statistics and kinetic energy
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                // Velocity magnitudes
                max_u_velocity = std::max(max_u_velocity, std::abs(u_center[j][i]));
                max_v_velocity = std::max(max_v_velocity, std::abs(v_center[j][i]));

                // Kinetic energy per cell: KE = 0.5 * (u² + v²)
                total_kinetic_energy += 0.5 * (u_center[j][i] * u_center[j][i] + 
                                              v_center[j][i] * v_center[j][i]);
            }
        }
        
        // Divergence check using staggered grid velocities
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                const auto divergence = (u_velocity[j][i] - u_velocity[j][i-1] + 
                                       v_velocity[j][i] - v_velocity[j-1][i]) * grid_spacing_inv;
                max_divergence = std::max(max_divergence, std::abs(divergence));
            }
        }
        
        const auto total_cells = (j_max - j_min + 1) * (i_max - i_min + 1);
        const auto average_kinetic_energy = total_kinetic_energy / total_cells;
        
        // Print formatted statistics
        std::cout << "Step " << std::setw(6) << step_number << "/" << total_time_steps
                  << " | t=" << std::fixed << std::setprecision(2) << std::setw(6) << current_time 
                  << " | max|u|=" << std::setprecision(4) << std::setw(8) << max_u_velocity
                  << " | max|v|=" << std::setprecision(4) << std::setw(8) << max_v_velocity
                  << " | max(div)=" << std::setprecision(2) << std::scientific << std::setw(10) << max_divergence
                  << " | avg_KE=" << std::fixed << std::setprecision(6) << std::setw(10) << average_kinetic_energy
                  << " | SOR_iter=" << std::setw(4) << sor_iterations << "\n";
    }
};

} //namespace CavityFlow

/**
 * @brief Main function - entry point for the simulation
 */
int main() {
    try {
        CavityFlow::CavitySolver solver;
        solver.run();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred\n";
        return 1;
    }
}