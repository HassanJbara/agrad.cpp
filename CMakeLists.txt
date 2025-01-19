cmake_minimum_required(VERSION 3.14)
project(neural_network VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR})

# add googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add matplotplusplus
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master
) 
FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


# Add source files and create static libraries
# add_library(lina STATIC lina/lina.cpp)
# add_library(matrix STATIC lina/Matrix.cpp)
# add_library(numeria STATIC numeria/numeria.cpp)
add_library(agrad STATIC agrad/Value.cpp)

# Add the executable
add_executable(main main.cpp)
add_executable(tmp tmp.cpp)

target_link_libraries(main agrad matplot)
target_link_libraries(tmp agrad matplot)

# Custom target for cleaning up build files
add_custom_target(deep_clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMENT "Removing build directory..."
)

add_custom_target(build_tmp
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target tmp
    COMMENT "Building tmp..."
)

add_custom_target(run_tmp
    COMMAND ${CMAKE_BINARY_DIR}/tmp
    COMMENT "Running tmp..."
)

enable_testing()

# Create main library target
add_library(nn_lib INTERFACE)
target_include_directories(nn_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Create test executable
add_executable(nn_tests
    agrad/tests.cpp
    nn/tests.cpp
    nn/test/NeuronTest.cpp
    nn/test/LayerTest.cpp
    nn/test/MLPTest.cpp
)

target_link_libraries(nn_tests
    PRIVATE
    nn_lib
    agrad
    GTest::gtest
    GTest::gtest_main
)

# Register tests
include(GoogleTest)
gtest_discover_tests(nn_tests)