if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    if(NOT BUILD_UCM_ASU)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]asu[/\\\\].*")
    endif()
    if(NOT TARGET dramstore)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]dram[/\\\\].*")
    endif()
    if(NOT TARGET yuanrongstore)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*/yuanrong/.*\\.cc$")
    endif()
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
        set_source_files_properties(case/dram/kv_protocol_test.cc
            PROPERTIES COMPILE_OPTIONS "-Wno-nonnull")
    endif()
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore fakestore pcstore healthbreakerstore
        gtest_main gtest gmock
    )
    if(TARGET dramstore AND TARGET drampool_core)
        target_link_libraries(ucmstore.test PRIVATE
            dramstore
            drampool_core
        )
    endif()
    if(TARGET yuanrongstore)
        target_compile_definitions(ucmstore.test PRIVATE UCM_YUANRONG_TESTING)
        target_link_libraries(ucmstore.test PRIVATE yuanrongstore)
    endif()
    if(BUILD_UCM_ASU)
        target_link_libraries(ucmstore.test PRIVATE
            kv_client
            buffer_pool
            trans
        )
        target_compile_definitions(ucmstore.test PRIVATE ASU_BUILD_TESTS)
    endif()
    target_link_libraries(ucmstore.test PRIVATE delegator delegator_store)
    gtest_discover_tests(ucmstore.test)
endif()

if(BUILD_UNIT_TESTS AND RUNTIME_ENVIRONMENT STREQUAL "simu")
    target_compile_definitions(ucmstore.test PRIVATE UCM_HOST_REGISTRATION_SIMU_TEST)
endif()

if(BUILD_UNIT_TESTS AND RUNTIME_ENVIRONMENT STREQUAL "simu")
    add_executable(cache_a5_host_memory.test
        ascend_a5/host_memory_config_test.cc
        ../cache/cc/host_memory_config.cc)
    target_compile_definitions(cache_a5_host_memory.test PRIVATE UCM_ASCEND_PINNED_SHARED_L1=1)
    target_include_directories(cache_a5_host_memory.test PRIVATE ../cache/cc)
    target_link_libraries(cache_a5_host_memory.test PRIVATE storeintf gtest_main)
    gtest_discover_tests(cache_a5_host_memory.test)
endif()
