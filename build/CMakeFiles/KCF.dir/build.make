# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jason/Project/KCFcpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason/Project/KCFcpp/build

# Include any dependencies generated for this target.
include CMakeFiles/KCF.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/KCF.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/KCF.dir/flags.make

CMakeFiles/KCF.dir/src/fhog.cpp.o: CMakeFiles/KCF.dir/flags.make
CMakeFiles/KCF.dir/src/fhog.cpp.o: ../src/fhog.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/Project/KCFcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/KCF.dir/src/fhog.cpp.o"
	/usr/bin/g++-5   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KCF.dir/src/fhog.cpp.o -c /home/jason/Project/KCFcpp/src/fhog.cpp

CMakeFiles/KCF.dir/src/fhog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KCF.dir/src/fhog.cpp.i"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jason/Project/KCFcpp/src/fhog.cpp > CMakeFiles/KCF.dir/src/fhog.cpp.i

CMakeFiles/KCF.dir/src/fhog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KCF.dir/src/fhog.cpp.s"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jason/Project/KCFcpp/src/fhog.cpp -o CMakeFiles/KCF.dir/src/fhog.cpp.s

CMakeFiles/KCF.dir/src/fhog.cpp.o.requires:

.PHONY : CMakeFiles/KCF.dir/src/fhog.cpp.o.requires

CMakeFiles/KCF.dir/src/fhog.cpp.o.provides: CMakeFiles/KCF.dir/src/fhog.cpp.o.requires
	$(MAKE) -f CMakeFiles/KCF.dir/build.make CMakeFiles/KCF.dir/src/fhog.cpp.o.provides.build
.PHONY : CMakeFiles/KCF.dir/src/fhog.cpp.o.provides

CMakeFiles/KCF.dir/src/fhog.cpp.o.provides.build: CMakeFiles/KCF.dir/src/fhog.cpp.o


CMakeFiles/KCF.dir/src/kcftracker.cpp.o: CMakeFiles/KCF.dir/flags.make
CMakeFiles/KCF.dir/src/kcftracker.cpp.o: ../src/kcftracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/Project/KCFcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/KCF.dir/src/kcftracker.cpp.o"
	/usr/bin/g++-5   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KCF.dir/src/kcftracker.cpp.o -c /home/jason/Project/KCFcpp/src/kcftracker.cpp

CMakeFiles/KCF.dir/src/kcftracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KCF.dir/src/kcftracker.cpp.i"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jason/Project/KCFcpp/src/kcftracker.cpp > CMakeFiles/KCF.dir/src/kcftracker.cpp.i

CMakeFiles/KCF.dir/src/kcftracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KCF.dir/src/kcftracker.cpp.s"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jason/Project/KCFcpp/src/kcftracker.cpp -o CMakeFiles/KCF.dir/src/kcftracker.cpp.s

CMakeFiles/KCF.dir/src/kcftracker.cpp.o.requires:

.PHONY : CMakeFiles/KCF.dir/src/kcftracker.cpp.o.requires

CMakeFiles/KCF.dir/src/kcftracker.cpp.o.provides: CMakeFiles/KCF.dir/src/kcftracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/KCF.dir/build.make CMakeFiles/KCF.dir/src/kcftracker.cpp.o.provides.build
.PHONY : CMakeFiles/KCF.dir/src/kcftracker.cpp.o.provides

CMakeFiles/KCF.dir/src/kcftracker.cpp.o.provides.build: CMakeFiles/KCF.dir/src/kcftracker.cpp.o


CMakeFiles/KCF.dir/src/roiSelector.cpp.o: CMakeFiles/KCF.dir/flags.make
CMakeFiles/KCF.dir/src/roiSelector.cpp.o: ../src/roiSelector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/Project/KCFcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/KCF.dir/src/roiSelector.cpp.o"
	/usr/bin/g++-5   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KCF.dir/src/roiSelector.cpp.o -c /home/jason/Project/KCFcpp/src/roiSelector.cpp

CMakeFiles/KCF.dir/src/roiSelector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KCF.dir/src/roiSelector.cpp.i"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jason/Project/KCFcpp/src/roiSelector.cpp > CMakeFiles/KCF.dir/src/roiSelector.cpp.i

CMakeFiles/KCF.dir/src/roiSelector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KCF.dir/src/roiSelector.cpp.s"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jason/Project/KCFcpp/src/roiSelector.cpp -o CMakeFiles/KCF.dir/src/roiSelector.cpp.s

CMakeFiles/KCF.dir/src/roiSelector.cpp.o.requires:

.PHONY : CMakeFiles/KCF.dir/src/roiSelector.cpp.o.requires

CMakeFiles/KCF.dir/src/roiSelector.cpp.o.provides: CMakeFiles/KCF.dir/src/roiSelector.cpp.o.requires
	$(MAKE) -f CMakeFiles/KCF.dir/build.make CMakeFiles/KCF.dir/src/roiSelector.cpp.o.provides.build
.PHONY : CMakeFiles/KCF.dir/src/roiSelector.cpp.o.provides

CMakeFiles/KCF.dir/src/roiSelector.cpp.o.provides.build: CMakeFiles/KCF.dir/src/roiSelector.cpp.o


CMakeFiles/KCF.dir/src/runtracker.cpp.o: CMakeFiles/KCF.dir/flags.make
CMakeFiles/KCF.dir/src/runtracker.cpp.o: ../src/runtracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/Project/KCFcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/KCF.dir/src/runtracker.cpp.o"
	/usr/bin/g++-5   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KCF.dir/src/runtracker.cpp.o -c /home/jason/Project/KCFcpp/src/runtracker.cpp

CMakeFiles/KCF.dir/src/runtracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KCF.dir/src/runtracker.cpp.i"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jason/Project/KCFcpp/src/runtracker.cpp > CMakeFiles/KCF.dir/src/runtracker.cpp.i

CMakeFiles/KCF.dir/src/runtracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KCF.dir/src/runtracker.cpp.s"
	/usr/bin/g++-5  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jason/Project/KCFcpp/src/runtracker.cpp -o CMakeFiles/KCF.dir/src/runtracker.cpp.s

CMakeFiles/KCF.dir/src/runtracker.cpp.o.requires:

.PHONY : CMakeFiles/KCF.dir/src/runtracker.cpp.o.requires

CMakeFiles/KCF.dir/src/runtracker.cpp.o.provides: CMakeFiles/KCF.dir/src/runtracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/KCF.dir/build.make CMakeFiles/KCF.dir/src/runtracker.cpp.o.provides.build
.PHONY : CMakeFiles/KCF.dir/src/runtracker.cpp.o.provides

CMakeFiles/KCF.dir/src/runtracker.cpp.o.provides.build: CMakeFiles/KCF.dir/src/runtracker.cpp.o


# Object files for target KCF
KCF_OBJECTS = \
"CMakeFiles/KCF.dir/src/fhog.cpp.o" \
"CMakeFiles/KCF.dir/src/kcftracker.cpp.o" \
"CMakeFiles/KCF.dir/src/roiSelector.cpp.o" \
"CMakeFiles/KCF.dir/src/runtracker.cpp.o"

# External object files for target KCF
KCF_EXTERNAL_OBJECTS =

KCF: CMakeFiles/KCF.dir/src/fhog.cpp.o
KCF: CMakeFiles/KCF.dir/src/kcftracker.cpp.o
KCF: CMakeFiles/KCF.dir/src/roiSelector.cpp.o
KCF: CMakeFiles/KCF.dir/src/runtracker.cpp.o
KCF: CMakeFiles/KCF.dir/build.make
KCF: /usr/local/lib/libopencv_dnn.so.4.0.0
KCF: /usr/local/lib/libopencv_ml.so.4.0.0
KCF: /usr/local/lib/libopencv_objdetect.so.4.0.0
KCF: /usr/local/lib/libopencv_shape.so.4.0.0
KCF: /usr/local/lib/libopencv_stitching.so.4.0.0
KCF: /usr/local/lib/libopencv_superres.so.4.0.0
KCF: /usr/local/lib/libopencv_videostab.so.4.0.0
KCF: /usr/local/lib/libopencv_photo.so.4.0.0
KCF: /usr/local/lib/libopencv_video.so.4.0.0
KCF: /usr/local/lib/libopencv_calib3d.so.4.0.0
KCF: /usr/local/lib/libopencv_features2d.so.4.0.0
KCF: /usr/local/lib/libopencv_flann.so.4.0.0
KCF: /usr/local/lib/libopencv_highgui.so.4.0.0
KCF: /usr/local/lib/libopencv_videoio.so.4.0.0
KCF: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
KCF: /usr/local/lib/libopencv_imgproc.so.4.0.0
KCF: /usr/local/lib/libopencv_core.so.4.0.0
KCF: CMakeFiles/KCF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jason/Project/KCFcpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable KCF"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KCF.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/KCF.dir/build: KCF

.PHONY : CMakeFiles/KCF.dir/build

CMakeFiles/KCF.dir/requires: CMakeFiles/KCF.dir/src/fhog.cpp.o.requires
CMakeFiles/KCF.dir/requires: CMakeFiles/KCF.dir/src/kcftracker.cpp.o.requires
CMakeFiles/KCF.dir/requires: CMakeFiles/KCF.dir/src/roiSelector.cpp.o.requires
CMakeFiles/KCF.dir/requires: CMakeFiles/KCF.dir/src/runtracker.cpp.o.requires

.PHONY : CMakeFiles/KCF.dir/requires

CMakeFiles/KCF.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/KCF.dir/cmake_clean.cmake
.PHONY : CMakeFiles/KCF.dir/clean

CMakeFiles/KCF.dir/depend:
	cd /home/jason/Project/KCFcpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason/Project/KCFcpp /home/jason/Project/KCFcpp /home/jason/Project/KCFcpp/build /home/jason/Project/KCFcpp/build /home/jason/Project/KCFcpp/build/CMakeFiles/KCF.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/KCF.dir/depend

