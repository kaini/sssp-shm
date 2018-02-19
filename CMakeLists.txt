cmake_minimum_required (VERSION 3.5)

# Project setup
###############
project (sssp-shm)
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE Release)
endif ()

# Compiler setup
################
set (CMAKE_CXX_STANDARD 14)
if (MSVC)
	# TODO With cmake 3.10+ this if can be removed.
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++${CMAKE_CXX_STANDARD}")
	add_definitions (-D_USE_MATH_DEFINES -D_WIN32_WINNT=0x0A00 -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
else ()
	set (CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -m64")
	set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
	set (CMAKE_CXX_FLAGS_RELEASE "-O3")
	set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3")
	if (CMAKE_SYSTEM_NAME STREQUAL "SunOS")
		link_libraries (atomic)
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
		set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -mcpu=native")
	else ()
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
		set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")
	endif ()
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
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	find_path (NUMA_INCLUDE_DIRECTORY numa.h)
	target_include_directories (numa SYSTEM INTERFACE ${NUMA_INCLUDE_DIRECTORY})
	target_link_libraries (numa INTERFACE -lnuma)
	target_compile_definitions (numa INTERFACE -DNUMA_LIBNUMA=1)
elseif (CMAKE_SYSTEM_NAME STREQUAL "SunOS")
	find_path (LGRP_INCLUDE_DIRECTORY sys/lgrp_user.h)
	target_include_directories (numa SYSTEM INTERFACE ${LGRP_INCLUDE_DIRECTORY})
	target_link_libraries (numa INTERFACE -llgrp)
	target_compile_definitions (numa INTERFACE -DNUMA_LGRP=1)
else ()
	target_compile_definitions (numa INTERFACE -DNUMA_NONE=1)
endif ()

add_library (hwloc INTERFACE)
find_path (HWLOC_INCLUDE_DIRECTORY hwloc.h)
find_library (HWLOC_LIBRARY hwloc)
target_include_directories (hwloc SYSTEM INTERFACE ${HWLOC_INCLUDE_DIRECTORY})
target_link_libraries (hwloc INTERFACE ${HWLOC_LIBRARY})

# sssp-mpi executable
#####################
function (add_variant NAME)
	string (TOUPPER ${NAME} UPPER_NAME)
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
	target_link_libraries (sssp-shm-${NAME}${NUMASTR} PRIVATE boost openmp hwloc)
	target_compile_definitions (sssp-shm-${NAME}${NUMASTR} PRIVATE -D${UPPER_NAME})
endfunction ()

add_variant (dijkstra)
add_variant (crauser)
