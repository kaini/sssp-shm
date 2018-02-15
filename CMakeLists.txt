cmake_minimum_required (VERSION 3.5)

# Project setup
###############
project (sssp-shm)

# Compiler setup
################
set (CMAKE_CXX_STANDARD 14)
# TODO With cmake 3.10+ this if can be removed.
if (MSVC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++${CMAKE_CXX_STANDARD}")
	add_definitions (-D_USE_MATH_DEFINES -D_WIN32_WINNT=0x0A00 -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
endif ()

# Find external libraries
#########################
add_library (openmp INTERFACE)
find_package (OpenMP REQUIRED)
target_compile_options (openmp INTERFACE ${OpenMP_CXX_FLAGS})
if (NOT MSVC)
	# TODO this is a cmake hack, nicer possible in 3.10.
	target_link_libraries (openmp INTERFACE "${OpenMP_CXX_FLAGS}")
endif ()

add_library (boost INTERFACE)
find_package (Boost 1.65.0 REQUIRED COMPONENTS date_time)
target_include_directories (boost SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
target_link_libraries (boost INTERFACE ${Boost_LIBRARIES})
if (MSVC)
	target_compile_definitions (boost INTERFACE -DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE)
endif ()

# sssp-mpi executable
#####################
function (add_variant NAME)
	string (TOUPPER ${NAME} UPPER_NAME)
	set (HEADERS
	)
	set (SOURCES
		main.cpp
	)
	add_executable (sssp-shm-${NAME} ${HEADERS} ${SOURCES})
	target_link_libraries (sssp-shm-${NAME} PRIVATE boost openmp)
	target_compile_definitions (sssp-shm-${NAME} PRIVATE -D${UPPER_NAME})
endfunction ()

add_variant (dijkstra)
add_variant (crauser)
