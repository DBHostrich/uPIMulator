# Copyright 2020 UPMEM. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR dpu)

set(triple dpu-upmem-dpurte)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_COMPILER_TARGET ${triple})
set(CMAKE_ASM_COMPILER clang)
set(CMAKE_ASM_COMPILER_TARGET ${triple})
set(CMAKE_AR llvm-ar)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_COMPILER_FORCED 1)

set(CHIP_VERSION "v1A")  # Default value
if(EXISTS "/sys/class/dpu_rank/dpu_rank0/dpu_chip_id")
    file(READ "/sys/class/dpu_rank/dpu_rank0/dpu_chip_id" CHIP_ID_NUMBER)
    if(CHIP_ID_NUMBER GREATER 8)
        set(CHIP_VERSION "v1B")
    endif()
endif()

set(CMAKE_C_FLAGS_INIT "-mcpu=${CHIP_VERSION} -g")
