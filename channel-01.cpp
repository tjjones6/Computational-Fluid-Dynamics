/**
 * @file    channel-01.cpp
 * @brief   Fully-explicit, staggered-grid, projection method channel flow solver.
 *          We are using a velocity inlet and pressure outlet
 *
 * @details
 *  * Time scheme:     Forward Euler  
 *  * Diffusion:       2nd-order central  
 *  * Convection:      1st-order central  
 *  * Pressure solver: SOR  
 *  * Output:          VTK files for ParaView with animation support
 *
 * @author  Tyler Jones
 * @date    2025-08-04
 * @version 2.0
 * 
 * @todo Compute vorticity in the main solver
 * 
 *  g++ -std=c++17 -O2 -Wall channel-01.cpp -o channel
 *  ./channel
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

namespace ChannelFlow {

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
[[nodiscard]] Field createField(int rows, int cols, double initial_value = 0.0) {
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
 * @brief Compute optimal SOR relaxation parameter for 2D anisotropic grid
 * @param nx Number of interior grid points in x-direction
 * @param ny Number of interior grid points in y-direction
 * @return Optimal omega value
 */
[[nodiscard]] constexpr double computeOptimalOmega2D(int nx, int ny) noexcept {
    constexpr double pi = 3.14159265358979323846;
    const double rho_j = 0.5 * (std::cos(pi / (nx + 1)) + std::cos(pi / (ny + 1)));
    const double denom = 1.0 + std::sqrt(std::max(1e-14, 1.0 - rho_j * rho_j));
    return 2.0 / denom; // typically ~1.7–1.95
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
     * @param nx Number of interior grid points in x
     * @param ny Number of interior grid points in y
     * @param dx Grid spacing in x
     * @param dy Grid spacing in y
     * @param time_value Current simulation time for temporal data
     */
    static void writeStructuredGrid(const std::string& filename,
                                    const Field& u_center,
                                    const Field& v_center, 
                                    const Field& pressure,
                                    int nx, int ny,
                                    double dx, double dy,
                                    double time_value = 0.0) {
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // VTK header with time information
        file << "# vtk DataFile Version 3.0\n";
        file << "Channel Flow Data - Time: " << std::fixed << std::setprecision(6) << time_value << "\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        
        // Grid dimensions (cell centers)
        file << "DIMENSIONS " << nx << " " << ny << " 1\n";
        
        // Origin (center of first cell)
        const double origin_x = dx * 0.5;
        const double origin_y = dy * 0.5;
        file << "ORIGIN " << origin_x << " " << origin_y << " 0.0\n";
        
        // Grid spacing
        file << "SPACING " << dx << " " << dy << " 1.0\n";
        
        // Point data with time value
        const int total_points = nx * ny;
        file << "POINT_DATA " << total_points << "\n";
        
        // Add time as a scalar field for ParaView temporal support
        file << "SCALARS TimeValue double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << time_value << "\n";
            }
        }
        
        // Velocity vector field
        file << "VECTORS velocity double\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << u_center[j][i] << " " << v_center[j][i] << " 0.0\n";
            }
        }
        
        // U-velocity scalar field
        file << "SCALARS u_velocity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << u_center[j][i] << "\n";
            }
        }
        
        // V-velocity scalar field
        file << "SCALARS v_velocity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << v_center[j][i] << "\n";
            }
        }
        
        // Velocity magnitude
        file << "SCALARS velocity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                const double mag = std::sqrt(u_center[j][i] * u_center[j][i] + 
                                           v_center[j][i] * v_center[j][i]);
                file << mag << "\n";
            }
        }
        
        // Pressure field
        file << "SCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << pressure[j][i] << "\n";
            }
        }
        
        // Vorticity calculation
        file << "SCALARS vorticity double 1\n";
        file << "LOOKUP_TABLE default\n";
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
        
        file.close();
        
        if (file.fail()) {
            throw std::runtime_error("Error writing to file: " + filename);
        }
    }
    
    /**
     * @brief Generate filename with proper ParaView time series convention
     * @param base_name Base filename
     * @param time_step Current time step
     * @return Formatted filename for ParaView time series
     */
    static std::string generate_filename(const std::string& base_name, 
                                       int time_step) {
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
 * @brief Main solver class for channel flow
 */
class ChannelSolver {
private:
    // Physical and numerical parameters
    static constexpr double LENGTH             = 3.0;    // Channel length
    static constexpr double HEIGHT             = 1.0;    // Channel height
    static constexpr int    NX_INT             = 93;     // Interior cells in x
    static constexpr int    NY_INT             = 31;     // Interior cells in y
    static constexpr double REYNOLDS_NUMBER    = 100.0;  // Re = inertial/viscous forces = (U*L) / nu 
    static constexpr double INLET_VELOCITY     = 1.0;    // Velocity of inlet
    static constexpr double DENSITY            = 1.0;    // Density of the fluid
    static constexpr double CFL                = 0.25;   // Stability factor
    static constexpr double final_time         = 10.0;   // Final time of simulation
    static constexpr double TOLERANCE_FACTOR   = 1e-7;   // Tolerance factor for SOR algorithm
    static constexpr double ABS_TOL            = 1e-10;  // Absolute tolerance for SOR
    static constexpr int    MAX_SOR_ITERS      = 10000;  // Maximum number of SOR sweeps
    static constexpr int    PRINT_INTERVAL     = 100;    // Interval for stats printed to terminal
    static constexpr int    SAVE_DATA_INTERVAL = 100;    // Interval for vtk snapshots/data export

    // Derived parameters
    const double kinematic_viscosity;
    const int    nx, ny;       // interior counts
    const double dx, dy;       // spacings
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
    ChannelSolver() 
        : kinematic_viscosity(INLET_VELOCITY * HEIGHT / REYNOLDS_NUMBER)
        , nx(NX_INT), ny(NY_INT)
        , dx(LENGTH / NX_INT), dy(HEIGHT / NY_INT)
        , optimal_omega(computeOptimalOmega2D(NX_INT, NY_INT))
        , time_step(CFL * std::min(0.25 * std::min(dx, dy) * std::min(dx, dy) / kinematic_viscosity, 
                                         std::min(dx, dy) / std::max(1e-12, INLET_VELOCITY)))
        , total_time_steps(static_cast<int>(final_time / time_step))
        , i_max(nx), j_max(ny)
    {
        validateParameters();
        allocateFields();
        setupOutputDirectory();
        printSimulationInfo();
        
        // Initialize velocities and apply boundary conditions
        applyBoundaryConditions();
        interpolateToCellCenters();
        exportData(0, 0.0);
    }

    /**
     * @brief Run the complete simulation
     */
    void run() {
        std::cout << GREEN
                  <<"Starting simulation...\n"
                  << RESET;
        
        for (int time_step_idx = 1; time_step_idx <= total_time_steps; ++time_step_idx) {
            const auto current_time = time_step_idx * time_step;
            
            computeTentativeVelocities();
            applyVelocityBC(u_tentative, v_tentative);
            
            buildSourceTerm();
            const auto [sor_iterations, residual] = solverPressurePoisson();
            
            applyPressureCorrection();
            applyBoundaryConditions();

            if (time_step_idx % PRINT_INTERVAL == 0 || time_step_idx == total_time_steps) {
                logStatistics(time_step_idx, current_time, sor_iterations, residual);
            }
            
            // Export data at specified intervals
            if (time_step_idx % SAVE_DATA_INTERVAL == 0 || time_step_idx == total_time_steps) {
                interpolateToCellCenters();
                exportData(time_step_idx, current_time);
            }
        }
        
        // Create ParaView collection file for animation
        createParaviewCollection();
        
        std::cout << GREEN
                  <<"Simulation completed successfully!\n"
                  <<"VTK files saved in directory: " << output_directory << "\n"
                  <<"Open '" << output_directory << "/channel_flow_animation.pvd' in ParaView for animation\n"
                  << RESET;
    }

private:
    /**
     * @brief Validate input parameters for physical consistency
     */
    void validateParameters() const {
        static_assert(NX_INT > 0, "Grid size must be positive!");
        static_assert(NY_INT > 0, "Grid size must be positive!");
        static_assert(REYNOLDS_NUMBER > 0, "Reynolds number must be positive!");
        static_assert(CFL > 0 && CFL < 1, "CFL number must be between 0 and 1!");
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
            pressure          = createField(j_max + 2, i_max + 2, 0.0);
            source_term       = createField(j_max + 2, i_max + 2, 0.0);
            poisson_residual  = createField(j_max + 2, i_max + 2, 0.0);
            u_tentative       = createField(j_max + 2, i_max + 1, 0.0);
            u_corrected       = createField(j_max + 2, i_max + 1, 0.0);
            u_center          = createField(j_max + 2, i_max + 2, 0.0);
            v_tentative       = createField(j_max + 1, i_max + 2, 0.0);
            v_corrected       = createField(j_max + 1, i_max + 2, 0.0);
            v_center          = createField(j_max + 2, i_max + 2, 0.0);
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
            const auto filename = VTKWriter::generate_filename("channel_flow", time_step_idx);
            const auto filepath = output_directory + "/" + filename;
            
            VTKWriter::writeStructuredGrid(filepath, u_center, v_center, pressure, 
                                            nx, ny, dx, dy, current_time);
            
            // Track exported files for collection
            exported_vtk_files.push_back(filename);
            exported_time_values.push_back(current_time);
            
            if (time_step_idx % PRINT_INTERVAL == 0 || time_step_idx == 0) {
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
            const auto collection_filepath = output_directory + "/channel_flow_animation.pvd";
            VTKWriter::write_paraview_collection(collection_filepath, exported_vtk_files, exported_time_values);
            
            std::cout << CYAN << "Created ParaView collection file: channel_flow_animation.pvd" << RESET << "\n";
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
                  << "=== Channel Flow Simulation ===\n"
                  << "Domain: " << LENGTH << "x" << HEIGHT << "\n"
                  << "Grid: " << nx << "x" << ny 
                  << " (dx=" << dx << ", dy=" << dy << ")\n"
                  << "Time: dt=" << time_step 
                  << ", steps=" << total_time_steps 
                  << ", final_time=" << final_time << "\n"
                  << "Reynolds=" << REYNOLDS_NUMBER 
                  << ", kinematic viscosity=" << kinematic_viscosity 
                  << ", CFL=" << CFL << "\n"
                  << "Relaxation factor=" << optimal_omega << "\n"
                  << "VTK export interval=" << SAVE_DATA_INTERVAL << " steps\n"
                  << "==========================================\n"
                  << RESET << "\n";
    }

    /**
     * @brief Apply boundary conditions using ghost cell method
     */
    void applyBoundaryConditions() noexcept {
        applyVelocityBC(u_corrected, v_corrected);
    }

    void applyVelocityBC(Field& uF, Field& vF) const noexcept {
        // Inlet (x=0): u=INLET_VELOCITY (Dirichlet at u-face), v=0
        for (int j = 1; j <= j_max; ++j) uF[j][0] = INLET_VELOCITY;
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

    /**
     * @brief Predictor step: compute tentative velocities without pressure gradient
     */
    void computeTentativeVelocities() noexcept {
        const double idx  = 1.0 / dx;
        const double idy  = 1.0 / dy;
        const double idx2 = 1.0 / (dx * dx);
        const double idy2 = 1.0 / (dy * dy);
        
        // Compute u* (tentative u-velocity)
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                // Viscous diffusion: ν∇²u
                const auto diffusion_term = kinematic_viscosity * (
                    (u_corrected[j][i+1] - 2.0*u_corrected[j][i] + u_corrected[j][i-1]) * idx2 +
                    (u_corrected[j+1][i] - 2.0*u_corrected[j][i] + u_corrected[j-1][i]) * idy2
                );
                
                // Convective flux: ∂(u²)/∂x
                const auto u_east       = 0.5 * (u_corrected[j][i]   + u_corrected[j][i+1]);
                const auto u_west       = 0.5 * (u_corrected[j][i-1] + u_corrected[j][i]);
                const auto convection_x = (u_east*u_east - u_west*u_west) * idx;
                
                // Cross-stream convective flux: ∂(vu)/∂y
                const auto v_north      = 0.5 * (v_corrected[j][i]   + v_corrected[j][i+1]);
                const auto v_south      = 0.5 * (v_corrected[j-1][i] + v_corrected[j-1][i+1]);
                const auto u_north      = 0.5 * (u_corrected[j+1][i] + u_corrected[j][i]);
                const auto u_south      = 0.5 * (u_corrected[j-1][i] + u_corrected[j][i]);
                const auto convection_y = (v_north*u_north - v_south*u_south) * idy;
                
                // Forward Euler update
                u_tentative[j][i] = u_corrected[j][i] + time_step * (diffusion_term - convection_x - convection_y);
            }
        }

        // Compute v* (tentative v-velocity)
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                // Viscous diffusion: ν∇²v
                const auto diffusion_term = kinematic_viscosity * (
                    (v_corrected[j][i+1] - 2.0*v_corrected[j][i] + v_corrected[j][i-1]) * idx2 +
                    (v_corrected[j+1][i] - 2.0*v_corrected[j][i] + v_corrected[j-1][i]) * idy2
                );

                // Convective flux: ∂(v²)/∂y
                const auto v_north      = 0.5 * (v_corrected[j][i]   + v_corrected[j+1][i]);
                const auto v_south      = 0.5 * (v_corrected[j-1][i] + v_corrected[j][i]);
                const auto convection_y = (v_north*v_north - v_south*v_south) * idy;

                // Cross-stream convective flux: ∂(uv)/∂x
                const auto u_east       = 0.5 * (u_corrected[j][i]   + u_corrected[j+1][i]);
                const auto u_west       = 0.5 * (u_corrected[j][i-1] + u_corrected[j+1][i-1]);
                const auto v_east       = 0.5 * (v_corrected[j][i]   + v_corrected[j][i+1]);
                const auto v_west       = 0.5 * (v_corrected[j][i-1] + v_corrected[j][i]);
                const auto convection_x = (u_east*v_east - u_west*v_west) * idx;

                // Forward Euler update
                v_tentative[j][i] = v_corrected[j][i] + time_step * (diffusion_term - convection_y - convection_x);
            }
        }
    }

    /**
     * @brief Build source term for pressure Poisson equation
     */
    void buildSourceTerm() noexcept {
        const double idx = 1.0 / dx, idy = 1.0 / dy;
        const double coeff = DENSITY / time_step;
        double max_source = 0.0;

        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                source_term[j][i] = coeff * ((u_tentative[j][i] - u_tentative[j][i-1]) * idx
                                          + (v_tentative[j][i] - v_tentative[j-1][i]) * idy);
                max_source = std::max(max_source, std::abs(source_term[j][i]));
            }
        }
        // Optional: remove tiny mean to aid convergence
        if (max_source > 0) {
            double mean_source = 0.0; int cnt = 0;
            for (int j = j_min; j <= j_max; ++j)
                for (int i = i_min; i <= i_max; ++i) { mean_source += source_term[j][i]; ++cnt; }
            mean_source /= static_cast<double>(cnt);
            for (int j = j_min; j <= j_max; ++j)
                for (int i = i_min; i <= i_max; ++i) source_term[j][i] -= mean_source;
        }
    }

    /**
     * @brief Solve pressure Poisson equation using SOR iteration
     * @return Pair of (iterations, final_residual)
     */
    [[nodiscard]] SolverResult solverPressurePoisson() {
        Field p_new = pressure; // start from previous p
        Field p_prev = p_new; // for GS east/north lookups
        const double idx2 = 1.0 / (dx * dx);
        const double idy2 = 1.0 / (dy * dy);
        const double denom = 2.0 * (idx2 + idy2);

        // tolerance based on source term magnitude
        double max_source = 0.0;
        for (int j = j_min; j <= j_max; ++j)
            for (int i = i_min; i <= i_max; ++i)
                max_source = std::max(max_source, std::abs(source_term[j][i]));
        const double tolerance = std::max(TOLERANCE_FACTOR * (max_source > 0 ? max_source : 1.0), ABS_TOL);

        double max_poisson_residual = tolerance + 1.0;
        int iteration_count = 0;

        while (max_poisson_residual > tolerance && iteration_count < MAX_SOR_ITERS) {
            ++iteration_count;
            p_prev = p_new; // capture state for GS east/north

            // Update interior with SOR (anisotropic 5-pt Laplacian, GS ordering)
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    const double pW = p_new[j][i-1];
                    const double pE = p_prev[j][i+1];
                    const double pS = p_new[j-1][i];
                    const double pN = p_prev[j+1][i];

                    const double sumNbrs = idx2 * (pE + pW) + idy2 * (pN + pS);
                    const double p_gs = (sumNbrs - source_term[j][i]) / denom;
                    p_new[j][i] = (1.0 - optimal_omega) * p_new[j][i] + optimal_omega * p_gs;
                }
            }
            // Refresh ghosts based on BCs
            applyPressureGhosts(p_new);

            // Residual (infty-norm of PPE)
            max_poisson_residual = 0.0;
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    const double lap = (p_new[j][i+1] - 2.0*p_new[j][i] + p_new[j][i-1]) * idx2
                                      + (p_new[j+1][i] - 2.0*p_new[j][i] + p_new[j-1][i]) * idy2;
                    poisson_residual[j][i] = lap - source_term[j][i];
                    max_poisson_residual = std::max(max_poisson_residual, std::abs(poisson_residual[j][i]));
                }
            }
        }
        if (iteration_count >= MAX_SOR_ITERS) {
            std::cerr << YELLOW << "Warning: PPE SOR hit max iterations, max_res=" << max_poisson_residual << RESET << "\n";
        }
        pressure.swap(p_new);
        return {iteration_count, max_poisson_residual};
    }

    /**
     * @brief Corrector step: apply pressure gradient to get final velocities
     */
    void applyPressureCorrection() noexcept {
        // u-correction on interior u-faces
        for (int j = j_min; j <= j_max; ++j)
            for (int i = i_min; i <= i_max - 1; ++i)
                u_corrected[j][i] = u_tentative[j][i] - (time_step / (DENSITY * dx)) * (pressure[j][i+1] - pressure[j][i]);
        // v-correction on interior v-faces
        for (int j = j_min; j <= j_max - 1; ++j)
            for (int i = i_min; i <= i_max;   ++i)
                v_corrected[j][i] = v_tentative[j][i] - (time_step / (DENSITY * dy)) * (pressure[j+1][i] - pressure[j][i]);
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
     * @param max_residual Maximum residual from pressure solver
     */
    void logStatistics(int step_number, double current_time, int sor_iterations, double max_residual) {
        interpolateToCellCenters();
        
        auto max_divergence = 0.0;
        auto total_kinetic_energy = 0.0;
        
        const auto dx_inv = 1.0 / dx;
        const auto dy_inv = 1.0 / dy;

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
                const auto divergence = (u_corrected[j][i] - u_corrected[j][i-1]) * dx_inv + 
                                       (v_corrected[j][i] - v_corrected[j-1][i]) * dy_inv;
                max_divergence = std::max(max_divergence, std::abs(divergence));
            }
        }
        
        const auto average_kinetic_energy = total_kinetic_energy / (nx * ny);
        
        // Print formatted statistics
        std::cout << "Step " << std::setw(6) << step_number << "/" << total_time_steps
                  << " | t=" << std::fixed << std::setprecision(3) << std::setw(8) << current_time 
                  << " | max(div)=" << std::scientific << std::setprecision(2) << std::setw(10) << max_divergence
                  << " | avg_KE=" << std::fixed << std::setprecision(6) << std::setw(10) << average_kinetic_energy
                  << " | PPE iters=" << std::setw(4) << sor_iterations
                  << " | res=" << std::scientific << std::setprecision(2) << std::setw(10) << max_residual
                  << "\n";
    }
};
}

/**
 * @brief Main function - entry point for the simulation
 */
int main() {
    try {
        ChannelFlow::ChannelSolver solver;
        solver.run();
        return 0;
    }
    catch (const std::exception& e) {
    std::cerr << RED << "Error: " << e.what() << RESET << "\n";
    return 1;
    }
    catch (...) {
        std::cerr << RED << "Unknown error occurred\n" << RESET;
        return 1;
    }
}