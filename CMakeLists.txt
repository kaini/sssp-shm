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
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
	set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
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
add_library (boost INTERFACE)
find_package (Boost 1.65.0 REQUIRED COMPONENTS date_time thread system)
target_include_directories (boost SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
target_link_libraries (boost INTERFACE ${Boost_LIBRARIES})
if (MSVC)
	target_compile_definitions (boost INTERFACE -DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE)
endif ()

add_library (hwloc INTERFACE)
find_path (HWLOC_INCLUDE_DIRECTORY hwloc.h)
find_library (HWLOC_LIBRARY hwloc)
target_include_directories (hwloc SYSTEM INTERFACE ${HWLOC_INCLUDE_DIRECTORY})
target_link_libraries (hwloc INTERFACE ${HWLOC_LIBRARY})
 
add_library (tbbmalloc INTERFACE)
if (CMAKE_SYSTEM_NAME STREQUAL "SunOS")
	# ignore this library
else ()
	find_library (Tbb_MALLOC_LIBRARY tbbmalloc_proxy)
	target_link_libraries (tbbmalloc INTERFACE ${Tbb_MALLOC_LIBRARY})
endif ()

# sssp-mpi executable
#####################
function (ADD_VARIANT)
	cmake_parse_arguments (ADD_VARIANT "" "NAME" "DEFINES" ${ARGN})
	set (HEADERS
		array_slice.hpp
		buddy_allocator.hpp
		by_edges_sssp.hpp
		carray.hpp
		collective_functions.hpp
		dijkstra.hpp
		graph.hpp
		own_queues_sssp.hpp
		relaxed_vector.hpp
		thread_local_allocator.hpp
	)
	set (SOURCES
		buddy_allocator.cpp
		by_edges_sssp.cpp
		dijkstra.cpp
		main.cpp
		own_queues_sssp.cpp
		thread_local_allocator.cpp
	)
	if (MSVC)
		set (NATVIS carray.natvis)
	else ()
		set (NATVIS)
	endif ()
	add_executable (sssp-shm-${ADD_VARIANT_NAME} ${HEADERS} ${SOURCES} ${NATVIS})
	target_link_libraries (sssp-shm-${ADD_VARIANT_NAME} PRIVATE boost hwloc tbbmalloc)
	target_compile_definitions (sssp-shm-${ADD_VARIANT_NAME} PRIVATE ${ADD_VARIANT_DEFINES})
endfunction ()

add_variant (NAME by-edges-crauser-out                DEFINES -DBY_EDGES -DCRAUSER_OUT)

add_variant (NAME by-nodes-crauser-in                 DEFINES -DBY_NODES -DCRAUSER_IN)
add_variant (NAME by-nodes-crauser-out                DEFINES -DBY_NODES -DCRAUSER_OUT)
add_variant (NAME by-nodes-crauser-indyn              DEFINES -DBY_NODES -DCRAUSER_INDYN)
add_variant (NAME by-nodes-crauser-outdyn             DEFINES -DBY_NODES -DCRAUSER_OUTDYN)

add_variant (NAME by-nodes-crauser-in-out             DEFINES -DBY_NODES -DCRAUSER_IN -DCRAUSER_OUT)
add_variant (NAME by-nodes-crauser-indyn-out          DEFINES -DBY_NODES -DCRAUSER_INDYN -DCRAUSER_OUT)
add_variant (NAME by-nodes-crauser-in-outdyn          DEFINES -DBY_NODES -DCRAUSER_IN -DCRAUSER_OUTDYN)
add_variant (NAME by-nodes-crauser-indyn-outdyn       DEFINES -DBY_NODES -DCRAUSER_INDYN -DCRAUSER_OUTDYN)
