cmake_minimum_required(VERSION 3.10)
project(acados_interface_example)

set(CMAKE_CXX_STANDARD 14)

include_directories(/home/arjunram/acados/include/hpipm/include)
include_directories(/home/arjunram/acados/include/blasfeo/include)

include_directories(/home/arjunram/acados/include)
include_directories(/home/arjunram/acados/include/acados)
include_directories(/home/arjunram/acados/include/acados_c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(acados_interface
    include/acados_interface.hpp
    src/acados_interface.cpp
)

add_library(acados_generated_code
    ../c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.h
    ../c_generated_code/acados_solver_curvilinear_bicycle_model_spatial.c
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

target_link_libraries(spline_interface
    PRIVATE
    acados_interface
    acados_generated_code
    m
)

# Link against acados runtime libraries (adjust paths if acados is installed elsewhere)
target_link_libraries(spline_interface
    PRIVATE
    /home/arjunram/acados/lib/libacados.so
    /home/arjunram/acados/lib/libblasfeo.so
    /home/arjunram/acados/lib/libhpipm.so
)

# Ensure runtime linker can find acados shared libs when running the executable from build dir
set_target_properties(spline_interface PROPERTIES
    INSTALL_RPATH "/home/arjunram/acados/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
