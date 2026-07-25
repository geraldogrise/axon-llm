# Fetch GoogleTest via FetchContent (no system-wide install needed).
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
# Avoid overriding the project's runtime settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(axon_tests
    test_tensor.cpp
    test_ops.cpp
    test_autograd.cpp
    test_activations.cpp
    test_transformer.cpp
    test_recurrent.cpp
    test_extra.cpp
    test_serialize.cpp
    test_nn.cpp
)
target_link_libraries(axon_tests PRIVATE axon GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(axon_tests)
