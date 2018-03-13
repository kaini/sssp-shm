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
	add_definitions (-D_USE_MATH_DEFINES -D_WIN32_WINNT=0x0A00 -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -D_ENABLE_ATOMIC_ALIGNMENT_FIX)
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
set (HEADERS
	array_slice.hpp
	buddy_allocator.hpp
	carray.hpp
	collective_functions.hpp
	dijkstra.hpp
	graph.hpp
	perf_counter.hpp
	relaxed_vector.hpp
	thread_local_allocator.hpp
)
set (SOURCES
	buddy_allocator.cpp
	carray.cpp
	dijkstra.cpp
	perf_counter.cpp
	thread_local_allocator.cpp
)
set (NATVIS carray.natvis)
add_library (common STATIC ${HEADERS} ${SOURCES} ${NATVIS})
target_link_libraries (common PUBLIC boost hwloc tbbmalloc)

function (ADD_VARIANT)
	set (MULTIVALUE_ARGUMENTS DEFINES FILES)
	cmake_parse_arguments (ADD_VARIANT "" "NAME" "${MULTIVALUE_ARGUMENTS}" ${ARGN})
	add_executable (sssp-shm-${ADD_VARIANT_NAME} main.cpp ${ADD_VARIANT_FILES})
	target_link_libraries (sssp-shm-${ADD_VARIANT_NAME} PRIVATE common)
	target_compile_definitions (sssp-shm-${ADD_VARIANT_NAME} PRIVATE ${ADD_VARIANT_DEFINES})
endfunction ()

foreach (QUEUE array heap)
	string (TOUPPER "${QUEUE}" DQUEUE)

	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-in
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_IN
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-out
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_OUT
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-indyn
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_INDYN
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-outdyn
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_OUTDYN
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)

	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-in-out
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_IN -DCRAUSER_OUT
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-in-out-traff
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_IN -DCRAUSER_OUT -DTRAFF
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-indyn-out
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_INDYN -DCRAUSER_OUT
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (
		NAME ${QUEUE}-by-nodes-crauser-in-outdyn
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_IN -DCRAUSER_OUTDYN
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (NAME ${QUEUE}-by-nodes-crauser-indyn-outdyn
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_INDYN -DCRAUSER_OUTDYN
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
	add_variant (NAME ${QUEUE}-by-nodes-crauser-indyn-outdyn-traff
		DEFINES -DQ_${DQUEUE} -DBY_NODES -DCRAUSER_INDYN -DCRAUSER_OUTDYN -DTRAFF
		FILES own_queues_sssp.hpp own_queues_sssp.cpp
	)
endforeach ()

add_variant (
	NAME heap-by-edges-crauser-out
	DEFINES -DQ_HEAP -DBY_EDGES -DCRAUSER_OUT
	FILES by_edges_sssp.hpp by_edges_sssp.cpp
)
add_variant (
	NAME delta-stepping
	DEFINES -DDELTASTEPPING
	FILES delta_stepping.hpp delta_stepping.cpp
)
