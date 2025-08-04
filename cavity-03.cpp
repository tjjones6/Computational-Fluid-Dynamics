/**
 * @file    cavity-03.cpp
 * @brief   Fully-explicit, staggered-grid lid-driven cavity solver with modern C++ practices and VTK export.
 *
 * @details
 *  • Time scheme:    Forward Euler  
 *  • Diffusion:      2nd-order central  
 *  • Convection:     1st-order central  
 *  • Pressure solver: SOR  
 *  • Output:         VTK files for ParaView with animation support
 *
 * @author Tyler Jones
 * @date   2025-07-28
 * @version 3.2
 * 
 * @todo Fix matrix sizes of u_corrected and v_corrected and BCs
 * 
 *  g++ -std=c++17 -O2 -Wall cavity-03.cpp -o cavity
 *  ./cavity
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
#include <sys/stat.h>

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
 * @brief VTK file writer for structured grid data with animation support
 */
class VTKWriter {
public:
    /**
     * @brief Write flow field data to VTK file with time information
     * @param filename Output filename
     * @param u_center U-velocity at cell centers
     * @param v_center V-velocity at cell centers
     * @param pressure Pressure field
     * @param grid_spacing Grid spacing
     * @param n_interior Number of interior grid points
     * @param time_value Current simulation time for temporal data
     */
    static void write_structured_grid(const std::string& filename,
                                    const Field& u_center,
                                    const Field& v_center, 
                                    const Field& pressure,
                                    double grid_spacing,
                                    int n_interior,
                                    double time_value = 0.0) {
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // VTK header with time information
        file << "# vtk DataFile Version 3.0\n";
        file << "Lid-Driven Cavity Flow Data - Time: " << std::fixed << std::setprecision(6) << time_value << "\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        
        // Grid dimensions (cell centers)
        file << "DIMENSIONS " << n_interior << " " << n_interior << " 1\n";
        
        // Origin (center of first cell)
        const double origin_x = grid_spacing * 0.5;
        const double origin_y = grid_spacing * 0.5;
        file << "ORIGIN " << origin_x << " " << origin_y << " 0.0\n";
        
        // Grid spacing
        file << "SPACING " << grid_spacing << " " << grid_spacing << " 1.0\n";
        
        // Point data with time value
        const int total_points = n_interior * n_interior;
        file << "POINT_DATA " << total_points << "\n";
        
        // Add time as a scalar field for ParaView temporal support
        file << "SCALARS TimeValue double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= n_interior; ++j) {
            for (int i = 1; i <= n_interior; ++i) {
                file << time_value << "\n";
            }
        }
        
        // Velocity vector field
        file << "VECTORS velocity double\n";
        for (int j = 1; j <= n_interior; ++j) {
            for (int i = 1; i <= n_interior; ++i) {
                file << u_center[j][i] << " " << v_center[j][i] << " 0.0\n";
            }
        }
        
        // U-velocity scalar field
        file << "SCALARS u_velocity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= n_interior; ++j) {
            for (int i = 1; i <= n_interior; ++i) {
                file << u_center[j][i] << "\n";
            }
        }
        
        // V-velocity scalar field
        file << "SCALARS v_velocity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= n_interior; ++j) {
            for (int i = 1; i <= n_interior; ++i) {
                file << v_center[j][i] << "\n";
            }
        }
        
        // Velocity magnitude
        file << "SCALARS velocity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= n_interior; ++j) {
            for (int i = 1; i <= n_interior; ++i) {
                const double mag = std::sqrt(u_center[j][i] * u_center[j][i] + 
                                           v_center[j][i] * v_center[j][i]);
                file << mag << "\n";
            }
        }
        
        // Pressure field
        file << "SCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= n_interior; ++j) {
            for (int i = 1; i <= n_interior; ++i) {
                file << pressure[j][i] << "\n";
            }
        }
        
        // Vorticity calculation (useful for cavity flow visualization)
        file << "SCALARS vorticity double 1\n";
        file << "LOOKUP_TABLE default\n";
        const double dx_inv = 1.0 / grid_spacing;
        for (int j = 1; j <= n_interior; ++j) {
            for (int i = 1; i <= n_interior; ++i) {
                double vorticity = 0.0;
                
                // Central differences for interior points
                if (i > 1 && i < n_interior && j > 1 && j < n_interior) {
                    const double dvdx = (v_center[j][i+1] - v_center[j][i-1]) * dx_inv * 0.5;
                    const double dudy = (u_center[j+1][i] - u_center[j-1][i]) * dx_inv * 0.5;
                    vorticity = dvdx - dudy;
                }
                // One-sided differences for boundary points
                else {
                    // Use available neighbors for boundary points
                    double dvdx = 0.0, dudy = 0.0;
                    
                    if (i == 1) {
                        dvdx = (v_center[j][i+1] - v_center[j][i]) * dx_inv;
                    } else if (i == n_interior) {
                        dvdx = (v_center[j][i] - v_center[j][i-1]) * dx_inv;
                    } else {
                        dvdx = (v_center[j][i+1] - v_center[j][i-1]) * dx_inv * 0.5;
                    }
                    
                    if (j == 1) {
                        dudy = (u_center[j+1][i] - u_center[j][i]) * dx_inv;
                    } else if (j == n_interior) {
                        dudy = (u_center[j][i] - u_center[j-1][i]) * dx_inv;
                    } else {
                        dudy = (u_center[j+1][i] - u_center[j-1][i]) * dx_inv * 0.5;
                    }
                    
                    vorticity = dvdx - dudy;
                }
                
                file << vorticity << "\n";
            }
        }
        
        file.close();
        
        if (file.fail()) {
            throw std::runtime_error("Error writing to file: " + filename);
        }
    }
    
    /**
     * @brief Generate filename with proper ParaView time series convention
     * @param base_name Base filename
     * @param time_step Current time step
     * @param time Current simulation time
     * @return Formatted filename for ParaView time series
     */
    static std::string generate_filename(const std::string& base_name, 
                                       int time_step, 
                                       double time) {
        std::ostringstream oss;
        // Use zero-padded format that ParaView recognizes for time series
        oss << base_name << "_" << std::setfill('0') << std::setw(6) << time_step << ".vtk";
        return oss.str();
    }
    
    /**
     * @brief Write a ParaView collection file (.pvd) for time series animation
     * @param collection_filename Name of the collection file
     * @param vtk_filenames Vector of VTK filenames
     * @param time_values Vector of corresponding time values
     */
    static void write_paraview_collection(const std::string& collection_filename,
                                        const std::vector<std::string>& vtk_filenames,
                                        const std::vector<double>& time_values) {
        if (vtk_filenames.size() != time_values.size()) {
            throw std::invalid_argument("VTK filenames and time values must have the same size");
        }
        
        std::ofstream file(collection_filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open collection file: " + collection_filename);
        }
        
        // Write PVD header
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <Collection>\n";
        
        // Write dataset entries
        for (size_t i = 0; i < vtk_filenames.size(); ++i) {
            file << "    <DataSet timestep=\"" << std::fixed << std::setprecision(6) 
                 << time_values[i] << "\" group=\"\" part=\"0\" file=\"" 
                 << vtk_filenames[i] << "\"/>\n";
        }
        
        file << "  </Collection>\n";
        file << "</VTKFile>\n";
        
        file.close();
        
        if (file.fail()) {
            throw std::runtime_error("Error writing collection file: " + collection_filename);
        }
    }
    
    /**
     * @brief Create output directory if it doesn't exist
     * @param directory_path Path to create
     */
    static void create_output_directory(const std::string& directory_path) {
        try {
            std::filesystem::create_directories(directory_path);
        } catch (const std::filesystem::filesystem_error& e) {
            throw std::runtime_error("Failed to create directory: " + directory_path + 
                                   " Error: " + e.what());
        }
    }
};

/**
 * @brief Main solver class for lid-driven cavity flow
 */
class CavitySolver {
private:
    // Physical and numerical parameters
    static constexpr double cavity_length    = 1.0;
    static constexpr double cavity_height    = 1.0;
    static constexpr int n_interior          = 63;
    static constexpr double reynolds_number  = 1000.0;
    static constexpr double lid_velocity     = 1.0;
    static constexpr double density          = 1.0;
    static constexpr double cfl_number       = 0.5;
    static constexpr double final_time       = 20.0;
    static constexpr double tolerance_factor = 1e-9;
    static constexpr int max_sor_iterations  = 10000;
    static constexpr int print_interval      = 100;
    static constexpr int save_data_interval  = 100;

    // Derived parameters
    const double kinematic_viscosity;
    const double grid_spacing;
    const double optimal_omega;
    const double time_step;
    const int total_time_steps;
    
    // Grid indexing helpers
    const int i_min = 1;   // First interior i index for cell center
    const int i_max;       // Last interior i index for cell center
    const int j_min = 1;   // First interior j index for cell center
    const int j_max;       // Last interior j index for cell center

    // Flow fields
    Field pressure;
    Field source_term;
    Field poisson_residual;
    Field u_tentative;
    Field u_corrected;
    Field u_center;
    Field v_tentative;
    Field v_corrected;
    Field v_center;
    
    // Output directory and tracking
    const std::string output_directory = "vtk_output";
    std::vector<std::string> exported_vtk_files;
    std::vector<double> exported_time_values;

public:
    /**
     * @brief Constructor - initializes solver parameters and allocates memory
     */
    CavitySolver() 
        : kinematic_viscosity(density * lid_velocity * cavity_length / reynolds_number)
        , grid_spacing(cavity_length / n_interior)
        , optimal_omega(compute_optimal_omega(n_interior))
        , time_step(cfl_number * std::min(0.25 * grid_spacing * grid_spacing / kinematic_viscosity, 
                                         grid_spacing / lid_velocity))
        , total_time_steps(static_cast<int>(final_time / time_step))
        , i_max(static_cast<int>(cavity_length * n_interior))
        , j_max(static_cast<int>(cavity_height * n_interior))
    {
        validateParameters();
        allocateFields();
        setupOutputDirectory();
        printSimulationInfo();
    }

    /**
     * @brief Run the complete simulation
     */
    void run() {
        std::cout << GREEN
                  <<"Starting simulation...\n"
                  << RESET;
        
        // Export initial conditions
        applyBoundaryConditions();
        interpolateToCellCenters();
        exportData(0, 0.0);
        
        for (int time_step_idx = 1; time_step_idx <= total_time_steps; ++time_step_idx) {
            const auto current_time = time_step_idx * time_step;
            
            applyBoundaryConditions();
            computeTentativeVelocities();
            const auto [sor_iterations, residual] = solverPressurePoisson();
            applyPressureCorrection();

            if (time_step_idx % print_interval == 0 || time_step_idx == total_time_steps) {
                logStatistics(time_step_idx, current_time, sor_iterations);
            }
            
            // Export data at specified intervals
            if (time_step_idx % save_data_interval == 0 || time_step_idx == total_time_steps) {
                interpolateToCellCenters();
                exportData(time_step_idx, current_time);
            }
        }
        
        // Create ParaView collection file for animation
        createParaviewCollection();
        
        std::cout << GREEN
                  <<"Simulation completed successfully!\n"
                  <<"VTK files saved in directory: " << output_directory << "\n"
                  <<"Open '" << output_directory << "/cavity_flow_animation.pvd' in ParaView for animation\n"
                  << RESET;
    }

private:
    /**
     * @brief Validate input parameters for physical consistency
     */
    void validateParameters() const {
        static_assert(n_interior > 0, "Grid size must be positive!");
        static_assert(reynolds_number > 0, "Reynolds number must be positive!");
        static_assert(cfl_number > 0 && cfl_number < 1, "CFL number must be between 0 and 1!");
        static_assert(final_time > 0, "Simulation time must be positive!");
        
        if (time_step <= 0) {
            throw std::runtime_error("Computed time step is non-positive. Check physical parameters!");
        }
    }

    /**
     * @brief Allocate memory for all flow fields
     */
    void allocateFields() {
        try {
            pressure          = create_field(j_max + 2, i_max + 2);
            source_term       = create_field(j_max + 2, i_max + 2);
            poisson_residual  = create_field(j_max + 2, i_max + 2);
            u_tentative       = create_field(j_max + 2, i_max + 1);
            u_corrected       = create_field(j_max + 2, i_max + 1);
            u_center          = create_field(j_max + 2, i_max + 2);
            v_tentative       = create_field(j_max + 1, i_max + 2);
            v_corrected       = create_field(j_max + 1, i_max + 2);
            v_center          = create_field(j_max + 2, i_max + 2);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate memory for flow fields: " + std::string(e.what()));
        }
    }

    /**
     * @brief Setup output directory for VTK files
     */
    void setupOutputDirectory() {
        try {
            VTKWriter::create_output_directory(output_directory);
            std::cout << BLUE << "Created output directory: " << output_directory << RESET << "\n";
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to setup output directory: " + std::string(e.what()));
        }
    }

    /**
     * @brief Export flow field data to VTK file
     * @param time_step_idx Current time step index
     * @param current_time Current simulation time
     */
    void exportData(int time_step_idx, double current_time) {
        try {
            const auto filename = VTKWriter::generate_filename("cavity_flow", time_step_idx, current_time);
            const auto filepath = output_directory + "/" + filename;
            
            VTKWriter::write_structured_grid(filepath, u_center, v_center, pressure, 
                                            grid_spacing, n_interior, current_time);
            
            // Track exported files for collection
            exported_vtk_files.push_back(filename);
            exported_time_values.push_back(current_time);
            
            if (time_step_idx % print_interval == 0 || time_step_idx == 0) {
                std::cout << BLUE << "Exported VTK file: " << filename << RESET << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << RED << "Error exporting VTK data: " << e.what() << RESET << "\n";
        }
    }
    
    /**
     * @brief Create ParaView collection file for time series animation
     */
    void createParaviewCollection() {
        try {
            const auto collection_filepath = output_directory + "/cavity_flow_animation.pvd";
            VTKWriter::write_paraview_collection(collection_filepath, exported_vtk_files, exported_time_values);
            
            std::cout << CYAN << "Created ParaView collection file: cavity_flow_animation.pvd" << RESET << "\n";
        } catch (const std::exception& e) {
            std::cerr << RED << "Error creating ParaView collection: " << e.what() << RESET << "\n";
        }
    }

    /**
     * @brief Print simulation parameters and setup information
     */
    void printSimulationInfo() const {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << CYAN
                  << "=== Lid-Driven Cavity Flow Simulation ===\n"
                  << "Domain: " << cavity_length << "x" << cavity_height << "\n"
                  << "Grid: " << n_interior << "x" << n_interior 
                  << " (spacing=" << grid_spacing << ")\n"
                  << "Time: dt=" << time_step 
                  << ", steps=" << total_time_steps 
                  << ", final_time=" << final_time << "\n"
                  << "Reynolds=" << reynolds_number 
                  << ", kinematic viscosity=" << kinematic_viscosity 
                  << ", CFL=" << cfl_number << "\n"
                  << "Relaxation factor=" << optimal_omega << "\n"
                  << "VTK export interval=" << save_data_interval << " steps\n"
                  << "==========================================\n"
                  << RESET << "\n";
    }

    /**
     * @brief Apply Nuemann boundary conditions using ghost cell method
     */
    void applyBoundaryConditions() noexcept {
        // North lid (moving wall) - u velocity
        for (int i = 0; i <= i_max; ++i) {
            u_corrected[j_max + 1][i] = 2.0 * lid_velocity - u_corrected[j_max][i];
        }

        // South wall (no-slip) - u velocity  
        for (int i = 0; i <= i_max; ++i) {
            u_corrected[0][i] = -u_corrected[1][i];
        }

        // East wall (no-slip) - v velocity
        for (int j = 0; j <= j_max; ++j) {
            v_corrected[j][i_max + 1] = -v_corrected[j][i_max];
        }

        // West wall (no-slip) - v velocity
        for (int j = 0; j <= j_max; ++j) {
            v_corrected[j][0] = -v_corrected[j][1];
        }
    }

    /**
     * @brief Predictor step: compute tentative velocities without pressure gradient
     */
    void computeTentativeVelocities() noexcept {
        const auto grid_spacing_inv = 1.0 / grid_spacing;
        const auto grid_spacing_sq_inv = 1.0 / (grid_spacing * grid_spacing);
        
        // Compute u* (tentative u-velocity)
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                // Viscous diffusion: ν∇²u
                const auto diffusion_term = kinematic_viscosity * (
                    (u_corrected[j][i+1] - 2.0*u_corrected[j][i] + u_corrected[j][i-1]) * grid_spacing_sq_inv +
                    (u_corrected[j+1][i] - 2.0*u_corrected[j][i] + u_corrected[j-1][i]) * grid_spacing_sq_inv
                );
                
                // Convective flux: ∂(u²)/∂x
                const auto u_east       = 0.5 * (u_corrected[j][i]   + u_corrected[j][i+1]);
                const auto u_west       = 0.5 * (u_corrected[j][i-1] + u_corrected[j][i]);
                const auto convection_x = (u_east*u_east - u_west*u_west) * grid_spacing_inv;
                
                // Cross-stream convective flux: ∂(vu)/∂y
                const auto v_north      = 0.5 * (v_corrected[j][i]   + v_corrected[j][i+1]);
                const auto v_south      = 0.5 * (v_corrected[j-1][i] + v_corrected[j-1][i+1]);
                const auto u_north      = 0.5 * (u_corrected[j+1][i] + u_corrected[j][i]);
                const auto u_south      = 0.5 * (u_corrected[j-1][i] + u_corrected[j][i]);
                const auto convection_y = (v_north*u_north - v_south*u_south) * grid_spacing_inv;
                
                // Forward Euler update
                u_tentative[j][i] = u_corrected[j][i] + time_step * (diffusion_term - convection_x - convection_y);
            }
        }

        // Compute v* (tentative v-velocity)
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                // Viscous diffusion: ν∇²v
                const auto diffusion_term = kinematic_viscosity * (
                    (v_corrected[j][i+1] - 2.0*v_corrected[j][i] + v_corrected[j][i-1]) * grid_spacing_sq_inv +
                    (v_corrected[j+1][i] - 2.0*v_corrected[j][i] + v_corrected[j-1][i]) * grid_spacing_sq_inv
                );

                // Convective flux: ∂(v²)/∂y
                const auto v_north      = 0.5 * (v_corrected[j][i]   + v_corrected[j+1][i]);
                const auto v_south      = 0.5 * (v_corrected[j-1][i] + v_corrected[j][i]);
                const auto convection_y = (v_north*v_north - v_south*v_south) * grid_spacing_inv;

                // Cross-stream convective flux: ∂(uv)/∂x
                const auto u_east       = 0.5 * (u_corrected[j][i]   + u_corrected[j+1][i]);
                const auto u_west       = 0.5 * (u_corrected[j][i-1] + u_corrected[j+1][i-1]);
                const auto v_east       = 0.5 * (v_corrected[j][i]   + v_corrected[j][i+1]);
                const auto v_west       = 0.5 * (v_corrected[j][i-1] + v_corrected[j][i]);
                const auto convection_x = (u_east*v_east - u_west*v_west) * grid_spacing_inv;

                // Forward Euler update
                v_tentative[j][i] = v_corrected[j][i] + time_step * (diffusion_term - convection_y - convection_x);
            }
        }
    }

    /**
     * @brief Solve pressure Poisson equation using SOR iteration
     * @return Pair of (iterations, final_residual)
     */
    [[nodiscard]] SolverResult solverPressurePoisson() {
        auto pressure_old = create_field(j_max + 2, i_max + 2);
        auto pressure_new = create_field(j_max + 2, i_max + 2);
        
        const auto grid_spacing_inv = 1.0 / grid_spacing;
        const auto grid_spacing_sq_inv = 1.0 / (grid_spacing * grid_spacing);
        const auto time_step_inv = 1.0 / time_step;
        
        auto max_source_term = 0.0;
        auto max_poisson_residual = 1.0;  // Initialize > 0 to enter loop
        auto iteration_count = 0;

        // Compute source term
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                source_term[j][i] = time_step_inv * density * (
                    (u_tentative[j][i] - u_tentative[j][i-1]) * grid_spacing_inv +
                    (v_tentative[j][i] - v_tentative[j-1][i]) * grid_spacing_inv
                );
                max_source_term = std::max(max_source_term, std::abs(source_term[j][i]));
            }
        }

        const auto tolerance = tolerance_factor * max_source_term;

        // SOR iteration loop
        while ((max_poisson_residual > tolerance) && (iteration_count < max_sor_iterations)) {
            ++iteration_count;
            pressure_old.swap(pressure_new);

            // SOR update sweep
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {

                    // Indicator functions
                    int eps_w = (i > i_min) ? 1 : 0;   // West neighbor
                    int eps_e = (i < i_max) ? 1 : 0;   // East neighbor  
                    int eps_n = (j < j_max) ? 1 : 0;   // North neighbor
                    int eps_s = j_min;                 // South neighbor (always true)
                    int neighbor_count = eps_w + eps_e + eps_n + eps_s;
                    
                    // SOR update
                    pressure_new[j][i] = pressure_old[j][i] * (1.0 - optimal_omega) + (optimal_omega / neighbor_count) * (
                        (eps_e*pressure_old[j][i+1] + eps_w*pressure_new[j][i-1]) + 
                        (eps_n*pressure_old[j+1][i] + eps_s*pressure_new[j-1][i]) - source_term[j][i] * (grid_spacing * grid_spacing)
                    );
                }
            }

            // Compute residual norm every iteration
            max_poisson_residual = 0.0;
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {

                    // Indicator functions
                    int eps_w = (i > i_min) ? 1 : 0;   // West neighbor
                    int eps_e = (i < i_max) ? 1 : 0;   // East neighbor  
                    int eps_n = (j < j_max) ? 1 : 0;   // North neighbor
                    int eps_s = j_min;                 // South neighbor (always true)

                    // Compute residual
                    poisson_residual[j][i] = grid_spacing_sq_inv * (
                        eps_e*(pressure_new[j][i+1] - pressure_new[j][i]) + eps_w*(pressure_new[j][i-1] - pressure_new[j][i]) + 
                        eps_n*(pressure_new[j+1][i] - pressure_new[j][i]) + eps_s*(pressure_new[j-1][i] - pressure_new[j][i])
                    ) - source_term[j][i];

                    max_poisson_residual = std::max(max_poisson_residual, std::abs(poisson_residual[j][i]));
                }
            }
        }

        // Check for convergence issues
        if (iteration_count >= max_sor_iterations) {
            std::cerr << "Warning: SOR solver did not converge in " << max_sor_iterations 
                      << " iterations. Final residual: " << max_poisson_residual << "\n";
        }

        // Update global pressure field
        pressure = std::move(pressure_new);

        return {iteration_count, max_poisson_residual};
    }

    /**
     * @brief Corrector step: apply pressure gradient to get final velocities
     */
    void applyPressureCorrection() noexcept {
        const auto dt_over_h = time_step / grid_spacing;
        
        // Correct u-velocities
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                u_corrected[j][i] = u_tentative[j][i] - dt_over_h * density * (pressure[j][i+1] - pressure[j][i]);
            }
        }
        
        // Correct v-velocities  
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                v_corrected[j][i] = v_tentative[j][i] - dt_over_h * density * (pressure[j+1][i] - pressure[j][i]);
            }
        }
    }

    /**
     * @brief Interpolate staggered velocities to cell centers
     * @return Pair of (u_center, v_center) fields
     */
    [[nodiscard]] VelocityFields interpolateToCellCenters() noexcept {
        // Interpolate u-velocity to cell centers
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                u_center[j][i] = 0.5 * (u_corrected[j][i-1] + u_corrected[j][i]);
            }
        }
        
        // Interpolate v-velocity to cell centers
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                v_center[j][i] = 0.5 * (v_corrected[j-1][i] + v_corrected[j][i]);
            }
        }
        
        return {u_center, v_center};
    }

    /**
     * @brief Compute and print flow statistics
     * @param step_number Current time step
     * @param current_time Current simulation time
     * @param sor_iterations Number of SOR iterations used
     */
    void logStatistics(int step_number, double current_time, int sor_iterations) {
        const auto [u_center, v_center] = interpolateToCellCenters();
        
        auto max_divergence = 0.0;
        auto total_kinetic_energy = 0.0;
        
        const auto grid_spacing_inv = 1.0 / grid_spacing;

        // Compute velocity statistics and kinetic energy
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                total_kinetic_energy += 0.5 * (u_center[j][i] * u_center[j][i] + 
                                              v_center[j][i] * v_center[j][i]);
            }
        }
        
        // Divergence check using staggered grid velocities
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                const auto divergence = (u_corrected[j][i] - u_corrected[j][i-1] + 
                                       v_corrected[j][i] - v_corrected[j-1][i]) * grid_spacing_inv;
                max_divergence = std::max(max_divergence, std::abs(divergence));
            }
        }
        
        const auto average_kinetic_energy = total_kinetic_energy / (n_interior * n_interior);
        
        // Print formatted statistics
        std::cout << "Step " << std::setw(6) << step_number << "/" << total_time_steps
                  << " | t=" << std::fixed << std::setprecision(2) << std::setw(6) << current_time 
                  << " | max(div)=" << std::setprecision(2) << std::scientific << std::setw(10) << max_divergence
                  << " | avg_KE=" << std::fixed << std::setprecision(6) << std::setw(10) << average_kinetic_energy
                  << " | SOR_iters=" << std::setw(4) << sor_iterations << "\n";
    }
};
}

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