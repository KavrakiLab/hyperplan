cmake_minimum_required(VERSION 3.1.3)
project(hyperplan)

find_package(catkin REQUIRED)
find_package(robowflex_ompl)
catkin_python_setup()
catkin_package()

if(robowflex_ompl_FOUND)
    add_executable(robowflex_helper src/robowflex_helper.cpp)
    target_include_directories(robowflex_helper PRIVATE ${robowflex_ompl_INCLUDE_DIRS})
    target_link_libraries(robowflex_helper ${robowflex_ompl_LIBRARIES})
    install(PROGRAMS bin/robowflex_helper
            DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()

install(PROGRAMS bin/${PROJECT_NAME}_cmdline.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

find_program(DOCKER docker NO_CMAKE_SYSTEM_PATH)
if(DOCKER)
    add_custom_target(docker-hyperplan
        COMMAND ${DOCKER} build -t "${PROJECT_NAME}" -f Dockerfile "${CMAKE_SOURCE_DIR}")
endif()
