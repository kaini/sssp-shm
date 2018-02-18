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
else ()
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
	set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -g3 -march=native")
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

add_library (numa INTERFACE)
find_path (NUMA_INCLUDE_DIRECTORY numa.h)
find_library (NUMA_LIBRARY numa)
target_include_directories (numa INTERFACE ${NUMA_INCLUDE_DIRECTORY})
target_link_libraries (numa INTERFACE ${NUMA_LIBRARY})
target_compile_definitions (numa INTERFACE -DNUMA)
if (NUMA_INCLUDE_DIRECTORY)
	set (HAS_NUMA YES)
else ()
	set (HAS_NUMA NO)
endif ()

# sssp-mpi executable
#####################
function (add_variant_impl NAME NUMA)
	string (TOUPPER ${NAME} UPPER_NAME)
	if (NUMA)
		set (NUMASTR "-numa")
	else ()
		set (NUMASTR "")
	endif ()
	set (HEADERS
		array_slice.hpp
		buddy_allocator.hpp
		carray.hpp
		thread_local_allocator.hpp
	)
	set (SOURCES
		buddy_allocator.cpp
		main.cpp
		thread_local_allocator.cpp
	)
	add_executable (sssp-shm-${NAME}${NUMASTR} ${HEADERS} ${SOURCES})
	target_link_libraries (sssp-shm-${NAME}${NUMASTR} PRIVATE boost openmp)
	if (NUMA)
		target_link_libraries (sssp-shm-${NAME}${NUMASTR} PRIVATE numa)
	endif ()
	target_compile_definitions (sssp-shm-${NAME}${NUMASTR} PRIVATE -D${UPPER_NAME})
endfunction ()

function (add_variant NAME)
	add_variant_impl (${NAME} NO)
	if (HAS_NUMA)
		add_variant_impl (${NAME} YES)
	endif ()
endfunction ()

add_variant (dijkstra)
add_variant (crauser)
