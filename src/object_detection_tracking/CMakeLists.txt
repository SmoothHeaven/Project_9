cmake_minimum_required(VERSION 3.8)
project(object_detection_tracking)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(OpenCV_DIR "/home/me492-ros/opencv_4_9/install/lib/cmake/opencv4")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# include directories
include_directories(${OpenCV_INCLUDE_DIRS})


# add executables and link libraries
add_executable(object_detection_tracking src/hopeanddream.cpp)
add_executable(bench src/testDNN.cpp)

target_link_libraries(object_detection_tracking ${OpenCV_LIBS})
target_link_libraries(bench ${OpenCV_LIBS})


ament_target_dependencies(
  object_detection_tracking
  "sensor_msgs"
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "geometry_msgs"
  "tf2_ros"
  "tf2_geometry_msgs"
)

ament_target_dependencies(
  bench
  "sensor_msgs"
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "geometry_msgs"
)

install(TARGETS object_detection_tracking
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS bench
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
