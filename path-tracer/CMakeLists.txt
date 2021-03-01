# CMakeList.txt : CMake project for path-tracer, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_executable (path-tracer "path-tracer.c"  "win32-threading.h" "c11-threading.h")

if (NOT MSVC)
    target_link_libraries(path-tracer m pthread)
endif()
