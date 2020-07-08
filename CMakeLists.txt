cmake_minimum_required(VERSION 3.1.3)
project(mphpo)

find_package(catkin REQUIRED)
find_package(robowflex_library)
catkin_python_setup()
catkin_package()

if(robowflex_library_FOUND)
    add_executable(robowflex_helper src/robowflex_helper.cpp)
    target_include_directories(robowflex_helper PRIVATE ${robowflex_library_INCLUDE_DIRS})
    target_link_libraries(robowflex_helper ${robowflex_library_LIBRARIES})
    install(PROGRAMS bin/robowflex_helper
            DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()

install(PROGRAMS bin/${PROJECT_NAME}_cmdline.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})