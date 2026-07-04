/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

/*
 * The "restore registers" program, is used by debugging processes to restore every registers of every runtime.
 * The program should be booted once on runtime 0.
 */

#include "restore_carry_and_zero_flag.h"

void __attribute__((naked, used, section(".text.__bootstrap"))) __bootstrap()
{
    /* clang-format off */
    __asm__ volatile(
        "  jeq id, " __STR(NR_THREADS) " - 1, .+2\n"
        "  boot id, 1\n"
        "  ld d2,  id8, " "448\n"
        "  ld d4,  id8, " "640\n"
        "  ld d6,  id8, " "832\n"
        "  ld d8,  id8, " "1024\n"
        "  ld d10, id8, " "1216\n"
        "  ld d12, id8, " "1408\n"
        "  ld d14, id8, " "1600\n"
        "  ld d16, id8, " "1792\n"
        "  ld d18, id8, " "1984\n"
        "  ld d20, id8, " "2176\n"
        "  ld d22, id8, " "2368\n"
        "  jnz id, atomic_done\n"
        "  move r0, " __STR(NR_ATOMIC_BITS) " - 1\n"
        "atomic_loop:\n"
        "  lbu r1, r0, 0\n"
        "  jz r1, atomic_release\n"
        "  acquire r0, 0, true, atomic_next\n"
        "atomic_release:\n"
        "  release r0, 0, nz, atomic_next\n"
        "atomic_next:\n"
        "  add r0, r0, -1, pl, atomic_loop\n"
        "atomic_done:\n"
        RESTORE_CARRY_AND_ZERO_FLAG
    );
    /* clang-format on */
}
