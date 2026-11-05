cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(torch)

if(NOT COMMAND add_mim_plugin)
    find_package(mim REQUIRED)
endif()

add_mim_plugin(torch
    SOURCES
        src/torch.cpp
        src/normalizers.cpp
        src/phase/lower.cpp
)

# `src/phase/lower.cpp` reads the runtime plugin's annexes. add_mim_plugin keeps each plugin's
# `include/` private, so reach the sibling checkout under `extra/` directly; an installed MimIR
# already provides the header through its include directory.
cmake_path(SET TORCH_RUNTIME_INCLUDE NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../runtime/include")
if(EXISTS "${TORCH_RUNTIME_INCLUDE}")
    target_include_directories(mim_torch PRIVATE "${TORCH_RUNTIME_INCLUDE}")
endif()

# `torch.mim` splices `helper.mim` with a path import, which is resolved relative to the importing
# file. The staged and installed `torch.mim` therefore needs `helper.mim` next to it; add_mim_plugin
# only stages the plugin's own `.mim`.
if(NOT DEFINED MIM_PLUGIN_DIR)
    set(MIM_PLUGIN_DIR "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/mim")
endif()
set(TORCH_HELPER_MIM     ${CMAKE_CURRENT_LIST_DIR}/helper.mim)
set(OUT_TORCH_HELPER_MIM ${MIM_PLUGIN_DIR}/helper.mim)
add_custom_command(
    OUTPUT  ${OUT_TORCH_HELPER_MIM}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_HELPER_MIM} ${OUT_TORCH_HELPER_MIM}
    DEPENDS ${TORCH_HELPER_MIM}
    COMMENT "Copy '${TORCH_HELPER_MIM}' to '${OUT_TORCH_HELPER_MIM}'"
    VERBATIM
)
add_custom_target(mim_internal_torch_helper DEPENDS ${OUT_TORCH_HELPER_MIM})
add_dependencies(mim_torch mim_internal_torch_helper)
install(FILES ${OUT_TORCH_HELPER_MIM} DESTINATION ${CMAKE_INSTALL_LIBDIR}/mim)
