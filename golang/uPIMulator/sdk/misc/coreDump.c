/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

/*
 * The "core dump" program, used by debugging processes to fetch each
 * runtime register and the atomic bits.
 * The program should be booted once on runtime 0.
 *
 * The output in WRAM has the following form:
 *  - byte 0..255 = atomic bits : each bit is stored into an individual byte
 *  - byte 256..2559 = work registers
 *  - byte 2560..2555 = flags
 *
 * Only the runtime 0 fills in the atomic bits part of the output.
 */

#include "restore_carry_and_zero_flag.h"

void __attribute__((naked, used, section(".text.__bootstrap"))) __bootstrap()
{
    /* clang-format off */
    __asm__ volatile(
        "  sd id8, " "256, d0\n"
        "  or r0, zero, 0, ?xnz, no_z_flag\n"
        "  or r0, r0, 0x2\n"
        "  no_z_flag:\n"
        "  addc r0, r0, 0\n"
        "  sw id4, " "2560, r0\n"
        "  jeq id, " __STR(NR_THREADS) " - 1, .+2\n"
        "  boot id, 1\n"
        "  sd id8, " "448, d2\n"
        "  sd id8, " "640, d4\n"
        "  sd id8, " "832, d6\n"
        "  sd id8, " "1024, d8\n"
        "  sd id8, " "1216, d10\n"
        "  sd id8, " "1408, d12\n"
        "  sd id8, " "1600, d14\n"
        "  sd id8, " "1792, d16\n"
        "  sd id8, " "1984, d18\n"
        "  sd id8, " "2176, d20\n"
        "  sd id8, " "2368, d22\n"
        "  jnz id, atomic_done\n"
        "  move r0, " __STR(NR_ATOMIC_BITS) " - 1\n"
        "atomic_loop:\n"
        "  sb r0, 0, 0xFF\n"
        "  acquire r0, 0, nz, atomic_next\n"
        "  sb r0, 0, 0x00\n"
        "  release r0, 0, nz, atomic_next\n"
        "atomic_next:\n"
        "  add r0, r0, -1, pl, atomic_loop\n"
        "atomic_done:\n"
        RESTORE_CARRY_AND_ZERO_FLAG
    );
    /* clang-format on */
}
