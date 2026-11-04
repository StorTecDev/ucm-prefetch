if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES CONFIGURE_DEPENDS "./case/*.cc")
    list(FILTER UCMSHARED_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]router_perf\\.cc$")
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        trans
        metrics
        infra_logger
        buffer_pool
        router
        gtest_main
        gtest
    )
    gtest_discover_tests(ucmshared.test)

    add_executable(router_perf case/router/router_perf.cc)
    target_link_libraries(router_perf PRIVATE router)
endif()

if(BUILD_UNIT_TESTS AND RUNTIME_ENVIRONMENT STREQUAL "simu")
    foreach(variant IN ITEMS available unavailable)
        add_executable(ascend_pin_only_${variant}.test
            ascend_pin_only/pin_only_api_test.cc
            ../trans/ascend/ascend_host_registration.cc)
        target_include_directories(ascend_pin_only_${variant}.test BEFORE PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/ascend_pin_only
            ${CMAKE_CURRENT_SOURCE_DIR}/..)
        target_link_libraries(ascend_pin_only_${variant}.test PRIVATE infra_status gtest_main)
        if(variant STREQUAL "unavailable")
            target_compile_definitions(ascend_pin_only_${variant}.test PRIVATE UCM_TEST_NO_PIN_API)
        endif()
        gtest_discover_tests(ascend_pin_only_${variant}.test TEST_PREFIX "${variant}.")
    endforeach()
endif()
