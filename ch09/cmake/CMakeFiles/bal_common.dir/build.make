# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yj/slambook2/ch9

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yj/slambook2/ch9/cmake

# Include any dependencies generated for this target.
include CMakeFiles/bal_common.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bal_common.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bal_common.dir/flags.make

CMakeFiles/bal_common.dir/common.cpp.o: CMakeFiles/bal_common.dir/flags.make
CMakeFiles/bal_common.dir/common.cpp.o: ../common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yj/slambook2/ch9/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bal_common.dir/common.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bal_common.dir/common.cpp.o -c /home/yj/slambook2/ch9/common.cpp

CMakeFiles/bal_common.dir/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bal_common.dir/common.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yj/slambook2/ch9/common.cpp > CMakeFiles/bal_common.dir/common.cpp.i

CMakeFiles/bal_common.dir/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bal_common.dir/common.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yj/slambook2/ch9/common.cpp -o CMakeFiles/bal_common.dir/common.cpp.s

# Object files for target bal_common
bal_common_OBJECTS = \
"CMakeFiles/bal_common.dir/common.cpp.o"

# External object files for target bal_common
bal_common_EXTERNAL_OBJECTS =

libbal_common.a: CMakeFiles/bal_common.dir/common.cpp.o
libbal_common.a: CMakeFiles/bal_common.dir/build.make
libbal_common.a: CMakeFiles/bal_common.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yj/slambook2/ch9/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libbal_common.a"
	$(CMAKE_COMMAND) -P CMakeFiles/bal_common.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bal_common.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bal_common.dir/build: libbal_common.a

.PHONY : CMakeFiles/bal_common.dir/build

CMakeFiles/bal_common.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bal_common.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bal_common.dir/clean

CMakeFiles/bal_common.dir/depend:
	cd /home/yj/slambook2/ch9/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yj/slambook2/ch9 /home/yj/slambook2/ch9 /home/yj/slambook2/ch9/cmake /home/yj/slambook2/ch9/cmake /home/yj/slambook2/ch9/cmake/CMakeFiles/bal_common.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bal_common.dir/depend
