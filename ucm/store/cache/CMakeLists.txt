file(GLOB_RECURSE UCM_CACHE_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(cachestore SHARED ${UCM_CACHE_STORE_CC_SOURCE_FILES})
target_include_directories(cachestore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(cachestore PUBLIC storeintf trans infra_logger metrics)
set_target_properties(cachestore PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../shared/metrics"
)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS cachestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)

# Keep the compatible A5 default; pin-only registration is an explicit opt-in.
if(RUNTIME_ENVIRONMENT STREQUAL "ascend-a5")
    target_compile_definitions(cachestore PRIVATE UCM_ASCEND_PAGEABLE_SHARED_L1=1)
endif()

if(BUILD_UNIT_TESTS)
    target_compile_definitions(cachestore PUBLIC UCM_ENABLE_TEST_HOOKS)
endif()
