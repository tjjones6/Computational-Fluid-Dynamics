/**
 * @file    backwards_step-01.cpp
 * @brief   Fully-explicit, staggered-grid, projection method backwards step flow solver.
 *          We are using a velocity inlet and pressure outlet with a backwards step geometry
 *
 * @details
 *  * Time scheme:     Forward Euler  
 *  * Diffusion:       2nd-order central  
 *  * Convection:      1st-order central  
 *  * Pressure solver: SOR  
 *  * Output:          VTK files for ParaView with animation support
 *  * Geometry:        Backwards step with inlet height H1, outlet height H2 (H2 > H1)
 *
 * @author  Tyler Jones
 * @date    2025-08-04
 * @version 2.0
 * 
 * @todo Compute vorticity in the main solver
 * 
 *  g++ -std=c++17 -O2 -Wall backwards_step-01.cpp -o backwards_step
 *  ./backwards_step
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

namespace BackwardsStepFlow {

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
 * @brief VTK file writer for structured grid data with animation support and obstacle masking
 */
class VTKWriter {
public:
    /**
     * @brief Write flow field data to VTK file with time information and obstacle masking
     * @param filename Output filename
     * @param u_center U-velocity at cell centers
     * @param v_center V-velocity at cell centers
     * @param pressure Pressure field
     * @param is_fluid Mask indicating which cells are fluid (true) or solid (false)
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
                                    const std::vector<std::vector<bool>>& is_fluid,
                                    int nx, int ny,
                                    double dx, double dy,
                                    double time_value = 0.0) {
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // VTK header with time information
        file << "# vtk DataFile Version 3.0\n";
        file << "Backwards Step Flow Data - Time: " << std::fixed << std::setprecision(6) << time_value << "\n";
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
        
        // Fluid/solid mask
        file << "SCALARS FluidMask double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << (is_fluid[j][i] ? 1.0 : 0.0) << "\n";
            }
        }
        
        // Velocity vector field (masked)
        file << "VECTORS velocity double\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                if (is_fluid[j][i]) {
                    file << u_center[j][i] << " " << v_center[j][i] << " 0.0\n";
                } else {
                    file << "0.0 0.0 0.0\n";
                }
            }
        }
        
        // U-velocity scalar field (masked)
        file << "SCALARS u_velocity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << (is_fluid[j][i] ? u_center[j][i] : 0.0) << "\n";
            }
        }
        
        // V-velocity scalar field (masked)
        file << "SCALARS v_velocity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << (is_fluid[j][i] ? v_center[j][i] : 0.0) << "\n";
            }
        }
        
        // Velocity magnitude (masked)
        file << "SCALARS velocity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                if (is_fluid[j][i]) {
                    const double mag = std::sqrt(u_center[j][i] * u_center[j][i] + 
                                               v_center[j][i] * v_center[j][i]);
                    file << mag << "\n";
                } else {
                    file << "0.0\n";
                }
            }
        }
        
        // Pressure field (masked)
        file << "SCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                file << (is_fluid[j][i] ? pressure[j][i] : 0.0) << "\n";
            }
        }
        
        // Vorticity calculation (masked)
        file << "SCALARS vorticity double 1\n";
        file << "LOOKUP_TABLE default\n";
        const double idx = 1.0 / dx, idy = 1.0 / dy;
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                if (!is_fluid[j][i]) {
                    file << "0.0\n";
                    continue;
                }
                
                double dvdx = 0.0, dudy = 0.0;
                
                // Check neighboring cells for vorticity calculation
                bool can_compute_vort = true;
                if (i == 1 || i == nx || j == 1 || j == ny) can_compute_vort = false;
                if (can_compute_vort && (!is_fluid[j][i-1] || !is_fluid[j][i+1] || 
                    !is_fluid[j-1][i] || !is_fluid[j+1][i])) can_compute_vort = false;
                
                if (can_compute_vort) {
                    dvdx = 0.5 * (v_center[j][i+1] - v_center[j][i-1]) * idx;
                    dudy = 0.5 * (u_center[j+1][i] - u_center[j-1][i]) * idy;
                    file << (dvdx - dudy) << "\n";
                } else {
                    file << "0.0\n";
                }
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
 * @brief Main solver class for backwards step flow
 */
class BackwardsStepSolver {
private:
    // Physical and numerical parameters
    static constexpr double LENGTH             = 8.0;    // Total channel length
    static constexpr double HEIGHT_INLET       = 1.0;    // Height of inlet channel
    static constexpr double HEIGHT_TOTAL       = 2.0;    // Total height after step
    static constexpr double STEP_LOCATION      = 2.0;    // x-location of step
    static constexpr int    NX_INT             = 8*32;    // Interior cells in x
    static constexpr int    NY_INT             = 32;     // Interior cells in y
    static constexpr double REYNOLDS_NUMBER    = 100.0;  // Re = inertial/viscous forces = (U*L) / nu 
    static constexpr double INLET_VELOCITY     = 1.0;    // Velocity of inlet
    static constexpr double DENSITY            = 1.0;    // Density of the fluid
    static constexpr double CFL                = 0.2;    // Stability factor (reduced for complex geometry)
    static constexpr double final_time         = 15.0;   // Final time of simulation
    static constexpr double TOLERANCE_FACTOR   = 1e-7;   // Tolerance factor for SOR algorithm
    static constexpr double ABS_TOL            = 1e-10;  // Absolute tolerance for SOR
    static constexpr int    MAX_SOR_ITERS      = 10000;  // Maximum number of SOR sweeps
    static constexpr int    PRINT_INTERVAL     = 10;    // Interval for stats printed to terminal
    static constexpr int    SAVE_DATA_INTERVAL = 10;     // Interval for vtk snapshots/data export

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
    
    // Step geometry parameters
    const int step_i_location;  // Grid index where step occurs
    const int step_j_height;    // Grid index for step height

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
    
    // Geometry mask (true for fluid cells, false for solid)
    std::vector<std::vector<bool>> is_fluid;
    
    // Output directory and tracking
    const std::string output_directory = "vtk_output";
    std::vector<std::string> exported_vtk_files;
    std::vector<double> exported_time_values;

public:
    /**
     * @brief Constructor - initializes solver parameters and allocates memory
     */
    BackwardsStepSolver() 
        : kinematic_viscosity(INLET_VELOCITY * HEIGHT_INLET / REYNOLDS_NUMBER)
        , nx(NX_INT), ny(NY_INT)
        , dx(LENGTH / NX_INT), dy(HEIGHT_TOTAL / NY_INT)
        , optimal_omega(computeOptimalOmega2D(NX_INT, NY_INT))
        , time_step(CFL * std::min(0.25 * std::min(dx, dy) * std::min(dx, dy) / kinematic_viscosity, 
                                         std::min(dx, dy) / std::max(1e-12, INLET_VELOCITY)))
        , total_time_steps(static_cast<int>(final_time / time_step))
        , i_max(nx), j_max(ny)
        , step_i_location(static_cast<int>(STEP_LOCATION / dx))
        , step_j_height(static_cast<int>((HEIGHT_TOTAL - HEIGHT_INLET) / dy))
    {
        validateParameters();
        allocateFields();
        setupGeometry();
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
                  <<"Starting backwards step simulation...\n"
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
                  <<"Open '" << output_directory << "/backwards_step_animation.pvd' in ParaView for animation\n"
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
        static_assert(HEIGHT_TOTAL > HEIGHT_INLET, "Total height must be greater than inlet height!");
        static_assert(STEP_LOCATION > 0 && STEP_LOCATION < LENGTH, "Step location must be within domain!");
        
        if (time_step <= 0) {
            throw std::runtime_error("Computed time step is non-positive. Check physical parameters!");
        }
        
        if (step_i_location <= 0 || step_i_location >= nx) {
            throw std::runtime_error("Step location is outside computational domain!");
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
            
            // Initialize fluid mask
            is_fluid.resize(j_max + 2);
            for (int j = 0; j <= j_max + 1; ++j) {
                is_fluid[j].resize(i_max + 2, false);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate memory for flow fields: " + std::string(e.what()));
        }
    }
    
    /**
     * @brief Setup backwards step geometry
     */
    void setupGeometry() {
        const int inlet_j_max = static_cast<int>(HEIGHT_INLET / dy);
        
        std::cout << CYAN << "Setting up backwards step geometry:\n";
        std::cout << "  Step location: x = " << STEP_LOCATION << " (i = " << step_i_location << ")\n";
        std::cout << "  Inlet height: " << HEIGHT_INLET << " (j = 1 to " << inlet_j_max << ")\n";
        std::cout << "  Total height: " << HEIGHT_TOTAL << " (j = 1 to " << j_max << ")\n" << RESET;
        
        // Initialize all cells as solid
        for (int j = 0; j <= j_max + 1; ++j) {
            for (int i = 0; i <= i_max + 1; ++i) {
                is_fluid[j][i] = false;
            }
        }
        
        // Mark fluid regions
        for (int i = i_min; i <= i_max; ++i) {
            for (int j = j_min; j <= j_max; ++j) {
                if (i <= step_i_location) {
                    // Before step: only lower part is fluid
                    if (j <= inlet_j_max) {
                        is_fluid[j][i] = true;
                    }
                } else {
                    // After step: full height is fluid
                    is_fluid[j][i] = true;
                }
            }
        }
        
        // Count fluid cells for verification
        int fluid_count = 0;
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                if (is_fluid[j][i]) fluid_count++;
            }
        }
        
        std::cout << BLUE << "Geometry setup complete. Fluid cells: " << fluid_count 
                  << "/" << (nx * ny) << RESET << "\n";
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
            const auto filename = VTKWriter::generate_filename("backwards_step", time_step_idx);
            const auto filepath = output_directory + "/" + filename;
            
            VTKWriter::writeStructuredGrid(filepath, u_center, v_center, pressure, is_fluid,
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
            const auto collection_filepath = output_directory + "/backwards_step_animation.pvd";
            VTKWriter::write_paraview_collection(collection_filepath, exported_vtk_files, exported_time_values);
            
            std::cout << CYAN << "Created ParaView collection file: backwards_step_animation.pvd" << RESET << "\n";
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
                  << "=== Backwards Step Flow Simulation ===\n"
                  << "Domain: " << LENGTH << "x" << HEIGHT_TOTAL << "\n"
                  << "Step: height=" << (HEIGHT_TOTAL - HEIGHT_INLET) 
                  << ", location=" << STEP_LOCATION << "\n"
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
     * @brief Apply boundary conditions using ghost cell method with step geometry
     */
    void applyBoundaryConditions() noexcept {
        applyVelocityBC(u_corrected, v_corrected);
    }

    void applyVelocityBC(Field& uF, Field& vF) const noexcept {
        const int inlet_j_max = static_cast<int>(HEIGHT_INLET / dy);
        
        // Inlet (x=0): u=INLET_VELOCITY for inlet region, no-slip elsewhere
        for (int j = 1; j <= inlet_j_max; ++j) {
            uF[j][0] = INLET_VELOCITY;
        }
        for (int j = inlet_j_max + 1; j <= j_max; ++j) {
            uF[j][0] = 0.0;  // No-slip on solid wall
        }
        for (int j = 0; j <= j_max; ++j) {
            vF[j][0] = 0.0;
        }

        // Outlet (x=L): zero-gradient for u,v
        for (int j = 1; j <= j_max; ++j) {
            uF[j][i_max] = uF[j][i_max - 1];
        }
        for (int j = 0; j <= j_max; ++j) {
            vF[j][i_max+1] = vF[j][i_max];
        }

        // Bottom wall (y=0): no-slip
        for (int i = 1; i <= i_max; ++i) {
            vF[0][i] = 0.0;
        }
        for (int i = 0; i <= i_max; ++i) {
            uF[0][i] = -uF[1][i];  // Antisymmetric for no-slip
        }

        // Top wall (y=H): no-slip
        for (int i = 1; i <= i_max; ++i) {
            vF[j_max][i] = 0.0;
        }
        for (int i = 0; i <= i_max; ++i) {
            uF[j_max+1][i] = -uF[j_max][i];  // Antisymmetric for no-slip
        }
        
        // Step walls: no-slip conditions for internal solid boundaries
        for (int j = 1; j <= j_max; ++j) {
            for (int i = 1; i <= i_max; ++i) {
                // Apply no-slip on solid-fluid interfaces
                if (!is_fluid[j][i]) {
                    // This is a solid cell, enforce no-slip on adjacent fluid faces
                    
                    // Check east face (u-velocity)
                    if (i < i_max && is_fluid[j][i+1]) {
                        uF[j][i] = 0.0;  // u-face between solid and fluid
                    }
                    
                    // Check west face (u-velocity)  
                    if (i > 1 && is_fluid[j][i-1]) {
                        uF[j][i-1] = 0.0;  // u-face between fluid and solid
                    }
                    
                    // Check north face (v-velocity)
                    if (j < j_max && is_fluid[j+1][i]) {
                        vF[j][i] = 0.0;  // v-face between solid and fluid
                    }
                    
                    // Check south face (v-velocity)
                    if (j > 1 && is_fluid[j-1][i]) {
                        vF[j-1][i] = 0.0;  // v-face between fluid and solid
                    }
                }
            }
        }
    }

    void applyPressureGhosts(Field& pF) const noexcept {
        const int inlet_j_max = static_cast<int>(HEIGHT_INLET / dy);
        
        // Inlet: Neumann dp/dx = 0 for inlet region
        for (int j = 1; j <= inlet_j_max; ++j) {
            pF[j][0] = pF[j][1];
        }
        // Solid wall at inlet: pressure can be extrapolated
        for (int j = inlet_j_max + 1; j <= j_max; ++j) {
            pF[j][0] = pF[j][1];
        }
        
        // Outlet: Dirichlet p = 0 (reference pressure)
        for (int j = 1; j <= j_max; ++j) {
            pF[j][i_max+1] = 0.0;
        }
        
        // Bottom and top walls: Neumann dp/dy = 0
        for (int i = 1; i <= i_max; ++i) {
            pF[0][i] = pF[1][i];
            pF[j_max+1][i] = pF[j_max][i];
        }
        
        // Internal solid boundaries: extrapolate pressure from neighboring fluid cells
        for (int j = 1; j <= j_max; ++j) {
            for (int i = 1; i <= i_max; ++i) {
                if (!is_fluid[j][i]) {
                    // This is a solid cell, extrapolate pressure from fluid neighbors
                    double p_sum = 0.0;
                    int fluid_neighbors = 0;
                    
                    // Check all four neighbors
                    if (i > 1 && is_fluid[j][i-1]) {
                        p_sum += pF[j][i-1];
                        fluid_neighbors++;
                    }
                    if (i < i_max && is_fluid[j][i+1]) {
                        p_sum += pF[j][i+1];
                        fluid_neighbors++;
                    }
                    if (j > 1 && is_fluid[j-1][i]) {
                        p_sum += pF[j-1][i];
                        fluid_neighbors++;
                    }
                    if (j < j_max && is_fluid[j+1][i]) {
                        p_sum += pF[j+1][i];
                        fluid_neighbors++;
                    }
                    
                    if (fluid_neighbors > 0) {
                        pF[j][i] = p_sum / fluid_neighbors;
                    }
                }
            }
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
        
        // Compute u* (tentative u-velocity) - only for fluid cells
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                // Check if this u-face is between two fluid cells or at fluid boundary
                bool valid_u_face = (i == 0) || (i == i_max) || 
                                   (is_fluid[j][i] || is_fluid[j][i+1]);
                
                if (!valid_u_face) {
                    u_tentative[j][i] = 0.0;
                    continue;
                }
                
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

        // Compute v* (tentative v-velocity) - only for fluid cells
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                // Check if this v-face is between two fluid cells or at fluid boundary
                bool valid_v_face = (j == 0) || (j == j_max) || 
                                   (is_fluid[j][i] || is_fluid[j+1][i]);
                
                if (!valid_v_face) {
                    v_tentative[j][i] = 0.0;
                    continue;
                }
                
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
     * @brief Build source term for pressure Poisson equation (only in fluid cells)
     */
    void buildSourceTerm() noexcept {
        const double idx = 1.0 / dx, idy = 1.0 / dy;
        const double coeff = DENSITY / time_step;
        double max_source = 0.0;

        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                if (!is_fluid[j][i]) {
                    source_term[j][i] = 0.0;
                    continue;
                }
                
                source_term[j][i] = coeff * ((u_tentative[j][i] - u_tentative[j][i-1]) * idx
                                          + (v_tentative[j][i] - v_tentative[j-1][i]) * idy);
                max_source = std::max(max_source, std::abs(source_term[j][i]));
            }
        }
        
        // Optional: remove mean from fluid cells to aid convergence
        if (max_source > 0) {
            double mean_source = 0.0; 
            int fluid_count = 0;
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    if (is_fluid[j][i]) {
                        mean_source += source_term[j][i]; 
                        fluid_count++;
                    }
                }
            }
            if (fluid_count > 0) {
                mean_source /= static_cast<double>(fluid_count);
                for (int j = j_min; j <= j_max; ++j) {
                    for (int i = i_min; i <= i_max; ++i) {
                        if (is_fluid[j][i]) {
                            source_term[j][i] -= mean_source;
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Solve pressure Poisson equation using SOR iteration (only in fluid cells)
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
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                if (is_fluid[j][i]) {
                    max_source = std::max(max_source, std::abs(source_term[j][i]));
                }
            }
        }
        const double tolerance = std::max(TOLERANCE_FACTOR * (max_source > 0 ? max_source : 1.0), ABS_TOL);

        double max_poisson_residual = tolerance + 1.0;
        int iteration_count = 0;

        while (max_poisson_residual > tolerance && iteration_count < MAX_SOR_ITERS) {
            ++iteration_count;
            p_prev = p_new; // capture state for GS east/north

            // Update interior fluid cells with SOR
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    if (!is_fluid[j][i]) continue;
                    
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

            // Residual (infty-norm of PPE in fluid cells only)
            max_poisson_residual = 0.0;
            for (int j = j_min; j <= j_max; ++j) {
                for (int i = i_min; i <= i_max; ++i) {
                    if (!is_fluid[j][i]) {
                        poisson_residual[j][i] = 0.0;
                        continue;
                    }
                    
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
     * @brief Corrector step: apply pressure gradient to get final velocities (only in fluid cells)
     */
    void applyPressureCorrection() noexcept {
        // u-correction on interior u-faces (only where meaningful)
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max - 1; ++i) {
                // Apply correction only if u-face is between fluid cells or at boundary
                bool valid_u_face = (i == 0) || (i == i_max - 1) || 
                                   (is_fluid[j][i] || is_fluid[j][i+1]);
                
                if (valid_u_face) {
                    u_corrected[j][i] = u_tentative[j][i] - 
                        (time_step / (DENSITY * dx)) * (pressure[j][i+1] - pressure[j][i]);
                } else {
                    u_corrected[j][i] = 0.0;  // No flow through solid boundaries
                }
            }
        }
        
        // v-correction on interior v-faces (only where meaningful)
        for (int j = j_min; j <= j_max - 1; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                // Apply correction only if v-face is between fluid cells or at boundary
                bool valid_v_face = (j == 0) || (j == j_max - 1) || 
                                   (is_fluid[j][i] || is_fluid[j+1][i]);
                
                if (valid_v_face) {
                    v_corrected[j][i] = v_tentative[j][i] - 
                        (time_step / (DENSITY * dy)) * (pressure[j+1][i] - pressure[j][i]);
                } else {
                    v_corrected[j][i] = 0.0;  // No flow through solid boundaries
                }
            }
        }
    }

    /**
     * @brief Interpolate staggered velocities to cell centers (only in fluid cells)
     */
    [[nodiscard]] VelocityFields interpolateToCellCenters() noexcept {
        // Initialize all centers to zero
        for (int j = 0; j <= j_max + 1; ++j) {
            for (int i = 0; i <= i_max + 1; ++i) {
                u_center[j][i] = 0.0;
                v_center[j][i] = 0.0;
            }
        }
        
        // Interpolate u-velocity to fluid cell centers
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                if (is_fluid[j][i]) {
                    u_center[j][i] = 0.5 * (u_corrected[j][i-1] + u_corrected[j][i]);
                }
            }
        }
        
        // Interpolate v-velocity to fluid cell centers
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                if (is_fluid[j][i]) {
                    v_center[j][i] = 0.5 * (v_corrected[j-1][i] + v_corrected[j][i]);
                }
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
        int fluid_cell_count = 0;
        
        const auto dx_inv = 1.0 / dx;
        const auto dy_inv = 1.0 / dy;

        // Compute velocity statistics and kinetic energy (fluid cells only)
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                if (is_fluid[j][i]) {
                    total_kinetic_energy += 0.5 * (u_center[j][i] * u_center[j][i] + 
                                                  v_center[j][i] * v_center[j][i]);
                    fluid_cell_count++;
                }
            }
        }
        
        // Divergence check using staggered grid velocities (fluid cells only)
        for (int j = j_min; j <= j_max; ++j) {
            for (int i = i_min; i <= i_max; ++i) {
                if (is_fluid[j][i]) {
                    const auto divergence = (u_corrected[j][i] - u_corrected[j][i-1]) * dx_inv + 
                                           (v_corrected[j][i] - v_corrected[j-1][i]) * dy_inv;
                    max_divergence = std::max(max_divergence, std::abs(divergence));
                }
            }
        }
        
        const auto average_kinetic_energy = (fluid_cell_count > 0) ? 
            total_kinetic_energy / fluid_cell_count : 0.0;
        
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
        BackwardsStepFlow::BackwardsStepSolver solver;
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