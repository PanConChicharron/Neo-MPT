cmake_minimum_required(VERSION 3.10)
project(acados_interface_example)

set(CMAKE_CXX_STANDARD 14)

include_directories(/home/arjunram/Workspace/acados/include/hpipm/include)
include_directories(/home/arjunram/Workspace/acados/include/blasfeo/include)

include_directories(/home/arjunram/Workspace/acados/include)
include_directories(/home/arjunram/Workspace/acados/include/acados)
include_directories(/home/arjunram/Workspace/acados/include/acados_c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find ROS2 packages
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(autoware_internal_debug_msgs REQUIRED)
find_package(autoware_planning_msgs REQUIRED)

# Add Autoware install include path so headers like autoware_internal_debug_msgs are found (provided by user workspace)
include_directories(/home/arjunram/Workspace/autoware-MPT/install/autoware_internal_debug_msgs/include)
include_directories(/home/arjunram/Workspace/autoware-MPT/install/autoware_internal_debug_msgs/include/autoware_internal_debug_msgs)
include_directories(/home/arjunram/Workspace/autoware-MPT/install/autoware_planning_msgs/include)
include_directories(/home/arjunram/Workspace/autoware-MPT/install/autoware_planning_msgs/include/autoware_planning_msgs)

add_library(acados_interface
    include/acados_interface.hpp
    src/acados_interface.cpp
)

add_library(acados_generated_code
    ../c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.h
    ../c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.c
)

# Make sure the generated-code target can find acados headers
target_include_directories(acados_generated_code
    PUBLIC
    /home/arjunram/Workspace/acados/include
    /home/arjunram/Workspace/acados/include/acados
    /home/arjunram/Workspace/acados/include/acados_c
    /home/arjunram/Workspace/acados/include/blasfeo/include
    /home/arjunram/Workspace/acados/include/hpipm/include
)

# also include the generated model external function sources
target_sources(acados_generated_code PRIVATE
    ../c_generated_code/curvilinear_bicycle_model_spatial_model/curvilinear_bicycle_model_spatial_expl_ode_fun.c
    ../c_generated_code/curvilinear_bicycle_model_spatial_model/curvilinear_bicycle_model_spatial_expl_vde_adj.c
    ../c_generated_code/curvilinear_bicycle_model_spatial_model/curvilinear_bicycle_model_spatial_expl_vde_forw.c
    ../c_generated_code/curvilinear_bicycle_model_spatial_model/curvilinear_bicycle_model_spatial_model.h
)


target_link_libraries(acados_interface
    acados_generated_code
)

# example executable that uses the C++ AcadosInterface wrapper
add_executable(spline_interface
    src/spline_interface.cpp
)

# Ensure ament is available and pull in ROS2 include/link flags
find_package(ament_cmake REQUIRED)
ament_target_dependencies(spline_interface rclcpp std_msgs geometry_msgs autoware_internal_debug_msgs autoware_planning_msgs)

# Link application against local acados interface libs and acados runtime libs
target_link_libraries(spline_interface
    acados_interface
    acados_generated_code
    m
    /home/arjunram/Workspace/acados/lib/libacados.so
    /home/arjunram/Workspace/acados/lib/libblasfeo.so
    /home/arjunram/Workspace/acados/lib/libhpipm.so
)

# Ensure runtime linker can find acados shared libs when running the executable from build dir
set_target_properties(spline_interface PROPERTIES
    INSTALL_RPATH "/home/arjunram/Workspace/acados/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
