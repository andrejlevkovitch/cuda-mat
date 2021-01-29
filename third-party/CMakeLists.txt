# third-party

find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)


option(CATCH_BUILD_TESTING OFF)
add_subdirectory(Catch2)
