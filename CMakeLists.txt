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
 
add_library (tbb INTERFACE)
find_path (Tbb_INCLUDE_DIR tbb/parallel_for.h)
target_include_directories (tbb INTERFACE ${Tbb_INCLUDE_DIR})
find_library (Tbb_LIBRARY tbb)
find_library (Tbb_MALLOC_LIBRARY tbbmalloc_proxy)
if (MSVC)
	find_library (Tbb_Debug_LIBRARY tbb_debug)
else ()
	set (Tbb_Debug_LIBRARY ${Tbb_LIBRARY})
endif ()
target_link_libraries (tbb
	INTERFACE optimized ${Tbb_LIBRARY} ${Tbb_MALLOC_LIBRARY}
	INTERFACE debug ${Tbb_Debug_LIBRARY} ${Tbb_MALLOC_LIBRARY}
)

# sssp-mpi executable
#####################
function (add_variant NAME)
	string (TOUPPER ${NAME} UPPER_NAME)
	set (HEADERS
		array_slice.hpp
		buddy_allocator.hpp
		carray.hpp
		collective_functions.hpp
		graph.hpp
		own_queues_sssp.hpp
		thread_local_allocator.hpp
	)
	set (SOURCES
		buddy_allocator.cpp
		main.cpp
		own_queues_sssp.cpp
		thread_local_allocator.cpp
	)
	add_executable (sssp-shm-${NAME}${NUMASTR} ${HEADERS} ${SOURCES})
	target_link_libraries (sssp-shm-${NAME}${NUMASTR} PRIVATE boost hwloc tbb)
	target_compile_definitions (sssp-shm-${NAME}${NUMASTR} PRIVATE -D${UPPER_NAME})
endfunction ()

add_variant (crauser)
add_variant (crauserdyn)
