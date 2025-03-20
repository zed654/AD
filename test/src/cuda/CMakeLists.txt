cmake_minimum_required(VERSION 3.8)
project(unified_memory_ros2)

# CUDA 설정
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
enable_language(CUDA)

# ROS2 설정
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# CUDA 실행 파일 추가
add_executable(unified_memory_node src/main.cu)
target_include_directories(unified_memory_node PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(unified_memory_node ${CUDA_LIBRARIES})

# ROS2 패키지 빌드 설정
ament_target_dependencies(unified_memory_node rclcpp std_msgs)

# 실행 파일 설치
install(TARGETS
  unified_memory_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
