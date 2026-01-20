cmake_minimum_required(VERSION 3.22.1)

project(CNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-frtti) 

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization system)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)


# Chercher les headers Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


# Pour CMake

# # Recherche d'OpenMP si demandé
# if(USE_OPENMP)
#     find_package(OpenMP)
#     if(OpenMP_CXX_FOUND)
#         message(STATUS "OpenMP trouvé: ${OpenMP_CXX_VERSION}")
#         set(HAVE_OPENMP 1)
#     else()
#         message(WARNING "OpenMP non trouvé - compilation sans parallelisation")
#         set(HAVE_OPENMP 0)
#     endif()
# else()
#     message(STATUS "OpenMP désactivé par l'utilisateur")
#     set(HAVE_OPENMP 0)
# endif()

# # if(NOT PYTHONLIBS_FOUND)
#     # message(FATAL_ERROR "Bibliothèque Python non trouvée")


# # Configuration OpenMP pour les targets (sera utilisé dans src/CMakeLists.txt)
# if(USE_OPENMP AND OpenMP_CXX_FOUND)
#     set(OPENMP_LIBRARIES OpenMP::OpenMP_CXX)
#     add_compile_definitions(HAVE_OPENMP)
#     message(STATUS "OpenMP activé pour le projet CNN")
# else()
#     set(OPENMP_LIBRARIES "")
#     message(STATUS "OpenMP désactivé - compilation séquentielle")
# endif()

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${PYTHON_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}

)

# Définitions globales pour tout le projet
add_compile_definitions(
    $<$<BOOL:${HAVE_OPENMP}>:HAVE_OPENMP>
)

add_subdirectory(src)
add_subdirectory(test)


