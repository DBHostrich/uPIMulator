/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>

#include <dpu_error.h>
#include <verbose_control.h>
#include <dpu_types.h>
#include <dpu_attributes.h>
#include <dpu_instruction_encoder.h>
#include <dpu_predef_programs.h>
#include <dpu_management.h>
#include <dpu_memory.h>
#include <dpu_config.h>
#include <dpu_debug.h>
#include <dpu_runner.h>

#include <dpu_rank.h>
#include <dpu_internals.h>
#include <dpu_api_log.h>
#include <dpu_mask.h>
#include <dpu_fifo.h>

static dpu_error_t
copy_from_mrams_using_dpu_program(struct dpu_rank_t *rank, const struct dpu_transfer_matrix *transfer_matrix);
static dpu_error_t
copy_to_mrams_using_dpu_program(struct dpu_rank_t *rank, const struct dpu_transfer_matrix *transfer_matrix);
static dpu_error_t
access_mram_using_dpu_program_individual(struct dpu_t *dpu,
    const struct dpu_transfer_matrix *transfer,
    dpu_transfer_type_t transfer_type,
    dpuinstruction_t *program,
    dpuinstruction_t *iram_save,
    iram_size_t nr_instructions,
    dpuword_t *wram_save,
    wram_size_t wram_size);

static inline uint32_t
_transfer_matrix_index(struct dpu_t *dpu)
{
    return get_transfer_matrix_index(dpu->rank, dpu->slice_id, dpu->dpu_id);
}

/*
 * Smart short-wait helper for the input-FIFO retry path. usleep() on Linux has
 * ~50-100 us granularity due to the scheduler quantum, so usleep(10) actually
 * sleeps ~50-100 us. That amplification is the dominant mechanism behind the
 * V1A 1-rank WRAM-FIFO pathology (per-DPU retry loop accumulating ~50us per
 * retry). For sub-50us requested waits we route through sched_yield() instead,
 * which gives other runnable threads a turn on the CPU without paying the
 * scheduler-quantum oversleep. Waits >= 50us still use usleep() (the legacy
 * behavior) since the requested duration is meaningful at that granularity.
 *
 * 2026-05-26 PIMulator optimization; default time_for_retry was simultaneously
 * lowered from 10 to 1 in init_fifo (see api/dpu_fifo.c).
 */
static inline void
dpu_fifo_retry_wait(uint32_t time_us)
{
    if (time_us < 50) {
        sched_yield();
    } else {
        usleep(time_us);
    }
}

__API_SYMBOL__ dpu_error_t
dpu_copy_to_iram_for_matrix(struct dpu_rank_t *rank, struct dpu_transfer_matrix *matrix)
{
    uint32_t iram_instruction_index = matrix->offset;
    uint32_t nb_of_instructions = matrix->size;
    LOG_RANK(DEBUG, rank, "%p, %u, %u", matrix, iram_instruction_index, nb_of_instructions);

    verify_iram_access(iram_instruction_index, nb_of_instructions, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_to_irams_matrix)(rank, matrix);
    dpu_unlock_rank(rank);

    return status;
}

__API_SYMBOL__ dpu_error_t
dpu_copy_from_iram_for_matrix(struct dpu_rank_t *rank, struct dpu_transfer_matrix *matrix)
{
    uint32_t iram_instruction_index = matrix->offset;
    uint32_t nb_of_instructions = matrix->size;
    LOG_RANK(DEBUG, rank, "%u, %u", iram_instruction_index, nb_of_instructions);

    verify_iram_access(iram_instruction_index, nb_of_instructions, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_from_irams_matrix)(rank, matrix);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_to_iram_for_rank(struct dpu_rank_t *rank,
    iram_addr_t iram_instruction_index,
    const dpuinstruction_t *source,
    iram_size_t nb_of_instructions)
{
    LOG_RANK(DEBUG, rank, "%u, %u", iram_instruction_index, nb_of_instructions);

    verify_iram_access(iram_instruction_index, nb_of_instructions, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_to_irams_rank)(rank, iram_instruction_index, source, nb_of_instructions);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_to_iram_for_dpu(struct dpu_t *dpu,
    iram_addr_t iram_instruction_index,
    const dpuinstruction_t *source,
    iram_size_t nb_of_instructions)
{
    LOG_DPU(DEBUG, dpu, "%u, %u", iram_instruction_index, nb_of_instructions);

    if (!dpu->enabled) {
        return DPU_ERR_DPU_DISABLED;
    }

    struct dpu_rank_t *rank = dpu_get_rank(dpu);
    verify_iram_access(iram_instruction_index, nb_of_instructions, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_to_irams_dpu)(dpu, iram_instruction_index, source, nb_of_instructions);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_from_iram_for_dpu(struct dpu_t *dpu,
    dpuinstruction_t *destination,
    iram_addr_t iram_instruction_index,
    iram_size_t nb_of_instructions)
{
    LOG_DPU(DEBUG, dpu, "%u, %u", iram_instruction_index, nb_of_instructions);

    if (!dpu->enabled) {
        return DPU_ERR_DPU_DISABLED;
    }

    struct dpu_rank_t *rank = dpu_get_rank(dpu);
    verify_iram_access(iram_instruction_index, nb_of_instructions, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_from_irams_dpu)(dpu, destination, iram_instruction_index, nb_of_instructions);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_to_wram_for_rank(struct dpu_rank_t *rank, wram_addr_t wram_word_offset, const dpuword_t *source, wram_size_t nb_of_words)
{
    LOG_RANK(DEBUG, rank, "%u, %u", wram_word_offset, nb_of_words);

    verify_wram_access(wram_word_offset, nb_of_words, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_to_wrams_rank)(rank, wram_word_offset, source, nb_of_words);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_to_wram_for_dpu(struct dpu_t *dpu, wram_addr_t wram_word_offset, const dpuword_t *source, wram_size_t nb_of_words)
{
    LOG_DPU(DEBUG, dpu, "%u, %u", wram_word_offset, nb_of_words);

    if (!dpu->enabled) {
        return DPU_ERR_DPU_DISABLED;
    }

    struct dpu_rank_t *rank = dpu_get_rank(dpu);
    verify_wram_access(wram_word_offset, nb_of_words, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_to_wrams_dpu)(dpu, wram_word_offset, source, nb_of_words);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_to_wram_for_matrix(struct dpu_rank_t *rank, struct dpu_transfer_matrix *transfer_matrix)
{
    uint32_t wram_word_offset = transfer_matrix->offset;
    uint32_t nb_of_words = transfer_matrix->size;
    LOG_RANK(DEBUG, rank, "%p, %u, %u", transfer_matrix, wram_word_offset, nb_of_words);

    verify_wram_access(wram_word_offset, nb_of_words, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_to_wrams_matrix)(rank, transfer_matrix);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_from_wram_for_dpu(struct dpu_t *dpu, dpuword_t *destination, wram_addr_t wram_word_offset, wram_size_t nb_of_words)
{
    LOG_DPU(DEBUG, dpu, "%u, %u", wram_word_offset, nb_of_words);

    if (!dpu->enabled) {
        return DPU_ERR_DPU_DISABLED;
    }

    struct dpu_rank_t *rank = dpu_get_rank(dpu);
    verify_wram_access(wram_word_offset, nb_of_words, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_from_wrams_dpu)(dpu, destination, wram_word_offset, nb_of_words);
    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_from_wram_for_matrix(struct dpu_rank_t *rank, struct dpu_transfer_matrix *transfer_matrix)
{
    uint32_t wram_word_offset = transfer_matrix->offset;
    uint32_t nb_of_words = transfer_matrix->size;
    LOG_RANK(DEBUG, rank, "%u, %u", wram_word_offset, nb_of_words);

    verify_wram_access(wram_word_offset, nb_of_words, rank);

    dpu_lock_rank(rank);
    dpu_error_t status = RANK_FEATURE(rank, copy_from_wrams_matrix)(rank, transfer_matrix);
    dpu_unlock_rank(rank);

    return status;
}

static bool
broadcast_possible(struct dpu_rank_t *rank, struct dpu_fifo_rank_t *fifo, uint8_t *wr_address)
{

    bool first = true;
    uint8_t nr_of_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
    uint8_t nr_of_cis = rank->description->hw.topology.nr_of_control_interfaces;
    for (dpu_slice_id_t each_ci = 0; each_ci < nr_of_cis; ++each_ci) {
        for (dpu_member_id_t each_dpu = 0; each_dpu < nr_of_dpus_per_ci; ++each_dpu) {
            struct dpu_t *dpu = DPU_GET_UNSAFE(rank, each_ci, each_dpu);

            if (!dpu || !dpu->enabled)
                continue;

            // check if the fifo is not full and the FIFOs write addresses the same
            if (is_fifo_full(fifo, dpu)) {
                return false;
            } else if (first) {
                first = false;
                *wr_address = get_fifo_wr_ptr(fifo, dpu);
            } else if (*wr_address != get_fifo_wr_ptr(fifo, dpu)) {
                return false;
            }
        }
    }
    return true;
}

static struct dpu_t *
get_first_enabled_dpu(struct dpu_rank_t *rank)
{

    uint8_t nr_of_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
    uint8_t nr_of_cis = rank->description->hw.topology.nr_of_control_interfaces;

    struct dpu_t *dpu = NULL;
    for (dpu_slice_id_t each_ci = 0; each_ci < nr_of_cis; ++each_ci) {
        for (dpu_member_id_t each_dpu = 0; each_dpu < nr_of_dpus_per_ci; ++each_dpu) {
            dpu = DPU_GET_UNSAFE(rank, each_ci, each_dpu);
            if (!dpu || !dpu->enabled)
                continue;
            else {
                return dpu;
            }
        }
    }
    return NULL;
}

// increment the write pointer for all DPUs in the rank
// this assumes that all DPUs have the same write pointer
static dpu_error_t
increment_fifo_wr_pointer(struct dpu_rank_t *rank, struct dpu_fifo_rank_t *fifo)
{

    // update the input fifo pointers write pointers
    // just take the pointer of dpu 0, they should all be the same

    struct dpu_t *dpu = get_first_enabled_dpu(rank);
    if (!dpu)
        return DPU_ERR_INTERNAL;
    uint64_t wr_ptr = get_fifo_abs_wr_ptr(fifo, dpu);
    ++wr_ptr;
    // +2 because the DPU structure contains first read pointer then write pointer
    return dpu_copy_to_wram_for_rank(rank, fifo->fifo_pointers_matrix.offset + 2, (void *)&wr_ptr, 2);
}

static dpu_error_t
send_data_one_dpu(struct dpu_t *dpu, struct dpu_fifo_rank_t *fifo, void *data)
{

    dpu_error_t status = DPU_OK;
    uint8_t input_fifo_wr_ptr = get_fifo_wr_ptr(fifo, dpu);
    status = (dpu_copy_to_wram_for_dpu(
        dpu, fifo->dpu_fifo_address + input_fifo_wr_ptr * fifo->dpu_fifo_data_size / 4, data, fifo->dpu_fifo_data_size / 4));

    return status;
}

static dpu_error_t
increment_fifo_wr_pointer_one_dpu(struct dpu_t *dpu, struct dpu_fifo_rank_t *fifo)
{

    uint64_t wr_ptr = get_fifo_abs_wr_ptr(fifo, dpu);
    wr_ptr++;
    return dpu_copy_to_wram_for_dpu(dpu, fifo->fifo_pointers_matrix.offset + 2, (void *)&wr_ptr, 2);
}

/* Patch F (2026-05-26): worst-case free slots across enabled DPUs as of last
 * rd_ptr refresh. FIFO depth is 1 << ptr_size; the extra pointer bit in
 * is_fifo_full's wrap-detect trick only distinguishes full vs empty, it
 * does NOT double capacity. Off-by-2x here caused silent overwrite of
 * unread slots at 1-DPU M19 (see tools/calibration/data/sdk_patches_2026-05-2x/patchf_v1_buggy/). */
static uint32_t
compute_min_free_slots(struct dpu_rank_t *rank, struct dpu_fifo_rank_t *fifo)
{
    uint64_t fifo_size = (uint64_t)1 << fifo->dpu_fifo_ptr_size;
    uint32_t min_free = (uint32_t)fifo_size;
    uint8_t nr_of_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
    uint8_t nr_of_cis = rank->description->hw.topology.nr_of_control_interfaces;
    for (dpu_slice_id_t each_ci = 0; each_ci < nr_of_cis; ++each_ci) {
        for (dpu_member_id_t each_dpu = 0; each_dpu < nr_of_dpus_per_ci; ++each_dpu) {
            struct dpu_t *dpu = DPU_GET_UNSAFE(rank, each_ci, each_dpu);
            if (!dpu || !dpu->enabled)
                continue;
            uint64_t rd = get_fifo_abs_rd_ptr(fifo, dpu);
            uint64_t wr = get_fifo_abs_wr_ptr(fifo, dpu);
            uint64_t occ = wr - rd;
            if (occ >= fifo_size) {
                return 0;
            }
            uint32_t free = (uint32_t)(fifo_size - occ);
            if (free < min_free)
                min_free = free;
        }
    }
    return min_free;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_to_wram_fifo(struct dpu_rank_t *rank, struct dpu_fifo_rank_t *fifo, struct dpu_transfer_matrix *transfer_matrix)
{

    bool loop = false;
    dpu_error_t status = DPU_OK;

    dpu_bitfield_t dpu_xfer_done_mask[DPU_MAX_NR_CIS] = { 0 };
    bool retry = false;
    uint64_t n_retries = 0;
    do {
        // will loop if one of the input FIFO is full and retry until the push is successful
        // allowing a maximum number of retries after which a DPU_ERR_WRAM_FIFO_FULL error is returned.
        loop = false;

        // Refresh DPU-owned rd_ptr only when our cached free-slot budget is
        // exhausted, or when we're in the retry path (need fresh DPU state to
        // detect when the DPU has consumed enough to make room). Otherwise
        // skip the read entirely: cached state strictly over-estimates real
        // occupancy.
        if (retry || fifo->min_free_budget == 0) {
            uint32_t saved_matrix_size = fifo->fifo_pointers_matrix.size;
            fifo->fifo_pointers_matrix.size = 2;
            status = dpu_copy_from_wram_for_matrix(rank, &(fifo->fifo_pointers_matrix));
            fifo->fifo_pointers_matrix.size = saved_matrix_size;
            if (status != DPU_OK)
                goto end;
            fifo->min_free_budget = compute_min_free_slots(rank, fifo);
        }

        // when data is sent at the same pace to all DPUs, the input FIFO write pointer for all of them
        // is the same, unless one of the DPU has its FIFO full and cannot take data anymore.
        // Hence for this case, the broadcast transfer is possible
        uint8_t wr_address = 0;
        // TODO support broadcast on retry ?
        // The broadcast path is the steady-state common case (~99% of pushes in
        // typical workloads); mark it likely so the compiler keeps the hot
        // instruction stream contiguous. 2026-05-26 PIMulator optimization.
        if (__builtin_expect(!retry && broadcast_possible(rank, fifo, &wr_address), 1)) {

            // transfer at the same address (divided by 4 to get address in words)
            transfer_matrix->offset = fifo->dpu_fifo_address + wr_address * fifo->dpu_fifo_data_size / 4;
            transfer_matrix->size = fifo->dpu_fifo_data_size >> 2;
            FF(dpu_copy_to_wram_for_matrix(rank, transfer_matrix));

            FF(increment_fifo_wr_pointer(rank, fifo));

            if (fifo->min_free_budget > 0)
                fifo->min_free_budget--;
        } else {

            uint8_t nr_of_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
            uint8_t nr_of_cis = rank->description->hw.topology.nr_of_control_interfaces;
            for (dpu_slice_id_t each_ci = 0; each_ci < nr_of_cis; ++each_ci) {
                for (dpu_member_id_t each_dpu = 0; each_dpu < nr_of_dpus_per_ci; ++each_dpu) {
                    struct dpu_t *dpu = DPU_GET_UNSAFE(rank, each_ci, each_dpu);

                    if (!dpu || !dpu->enabled)
                        continue;

                    void *data = dpu_transfer_matrix_get_ptr(dpu, transfer_matrix);

                    // check if the fifo is not full and if there is remaining data to be sent
                    if (!dpu_mask_is_selected(dpu_xfer_done_mask[each_ci], each_dpu) && data) {
                        if (!is_fifo_full(fifo, dpu)) {

                            // send data
                            send_data_one_dpu(dpu, fifo, data);

                            // update the input fifo write pointer
                            FF(increment_fifo_wr_pointer_one_dpu(dpu, fifo));

                            dpu_xfer_done_mask[each_ci] = dpu_mask_select(dpu_xfer_done_mask[each_ci], each_dpu);
                        } else {
                            loop = true;
                            retry = true;
                        }
                    }
                }
            }
            // Per-DPU path diverges wr_ptrs; budget accounting no longer
            // applies. Force refresh on next entry.
            fifo->min_free_budget = 0;
            if (retry) {
                n_retries++;
                // Route short waits through sched_yield to avoid usleep's
                // ~50-100us scheduler-quantum oversleep (V1A 1-rank pathology
                // mitigation). See dpu_fifo_retry_wait above.
                dpu_fifo_retry_wait(fifo->time_for_retry);
                // stop retries after a threshold
                if (n_retries > fifo->max_retries) {
                    loop = false;
                    status = DPU_ERR_WRAM_FIFO_FULL;
                }
            }
        }
    } while (loop);

end:
    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_from_wram_fifo(struct dpu_rank_t *rank, struct dpu_fifo_rank_t *fifo, struct dpu_transfer_matrix *transfer_matrix)
{
    dpu_error_t status = DPU_OK;

    // retrieve the value of FIFO pointers from DPUs
    FF(dpu_copy_from_wram_for_matrix(rank, &(fifo->fifo_pointers_matrix)));

    // copy the content of DPU output FIFO to the local buffer
    // TODO we could optimize here to transfer only the max size of the FIFOs.
    // Also need to determine how the user knows which elements are valid and not from the FIFO
    // With the size of the transfer, we will transfer garbage in the invalid elements
    // if(broadcast_possible)
    transfer_matrix->offset = fifo->dpu_fifo_address;
    // TODO cleanup and check if optimization possible when no wrap-around
    uint16_t sz = get_fifo_max_size(rank, fifo);
    if (sz) {
        // O(K) output flush (PIMulator 2026-06-06), RUNTIME-gated by the
        // DPU_FIFO_OKFLUSH env var (cached on first call). Stock vs patched is
        // therefore a runtime toggle on the SAME lib -- no recompile, and not
        // dependent on CMAKE_C_FLAGS reaching this translation unit (an #ifdef
        // build flag did NOT propagate to dpu_memory.c on upmemcloud5).
        //
        // Stock behavior (env unset): transfer the WHOLE ring
        // (data_size << ptr_size) on every pull regardless of occupancy --
        // O(capacity), the dominant cost in tools/calibration/data/
        // wram_output_fifo_*. Enabled (env set) AND the occupied region does
        // not wrap for ANY enabled DPU: transfer only the union span
        // [min_rd, max_wr) across the rank -- O(K). Falls back to the full
        // transfer on wrap/full, so correctness is never at risk.
        //
        // SEMANTIC NOTE: when the optimized path fires, ring element min_rd
        // lands at host-buffer offset 0 (not at its absolute ring index). For
        // synchronized DPUs min_rd == rd, so buf[0..K) are exactly the valid
        // elements; a caller relying on the stock absolute-ring-index layout
        // must offset reads by min_rd.
        static int okflush_env = -1;
        if (okflush_env < 0)
            okflush_env = (getenv("DPU_FIFO_OKFLUSH") != NULL) ? 1 : 0;

        bool used_optimized = false;
        if (okflush_env) {
            uint32_t words_per_elem = fifo->dpu_fifo_data_size >> 2;
            uint32_t capacity = 1u << fifo->dpu_fifo_ptr_size;
            uint32_t mask = capacity - 1u;
            bool okflush = true;
            uint32_t min_rd = capacity;
            uint32_t max_wr = 0;
            uint8_t nr_of_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
            uint8_t nr_of_cis = rank->description->hw.topology.nr_of_control_interfaces;
            for (dpu_slice_id_t each_ci = 0; each_ci < nr_of_cis && okflush; ++each_ci) {
                for (dpu_member_id_t each_dpu = 0; each_dpu < nr_of_dpus_per_ci; ++each_dpu) {
                    struct dpu_t *dpu = DPU_GET_UNSAFE(rank, each_ci, each_dpu);
                    if (!dpu || !dpu->enabled)
                        continue;
                    uint64_t abs_rd = get_fifo_abs_rd_ptr(fifo, dpu);
                    uint64_t abs_wr = get_fifo_abs_wr_ptr(fifo, dpu);
                    uint64_t occ = abs_wr - abs_rd;
                    if (occ == 0)
                        continue; // empty: contributes nothing to the span
                    if (occ >= capacity) {
                        okflush = false; // full
                        break;
                    }
                    uint32_t rd = (uint32_t)(abs_rd & mask);
                    uint32_t wr = (uint32_t)(abs_wr & mask);
                    if (rd < wr) {
                        if (rd < min_rd)
                            min_rd = rd;
                        if (wr > max_wr)
                            max_wr = wr;
                    } else {
                        okflush = false; // occupied region wraps the ring
                        break;
                    }
                }
            }
            if (okflush && max_wr > min_rd) {
                transfer_matrix->offset = fifo->dpu_fifo_address + min_rd * words_per_elem;
                transfer_matrix->size = (max_wr - min_rd) * words_per_elem;
                used_optimized = true;
            }
        }
        if (!used_optimized) {
            // stock: full ring (circular-buffer wrap means invalid slots carry
            // garbage, but we avoid transferring when size is zero).
            transfer_matrix->offset = fifo->dpu_fifo_address;
            transfer_matrix->size = (fifo->dpu_fifo_data_size << fifo->dpu_fifo_ptr_size) >> 2;
        }
        FF(dpu_copy_from_wram_for_matrix(rank, transfer_matrix));
    }

    // We need to update the read pointer with the write pointer on the DPU
    // But we dont want to update it on the host (to retrieve correct size of the FIFO)
    // Hence after copying on the DPU we swap again the read and write to go back to
    // original state
    swap_fifo_rd_wr_ptr(fifo);

    // Note: we should never erase the FIFO write pointer. For an output FIFO, this is owned by the DPU
    // We only want to write the read pointer, so decrease the matrix size to 2 word for this transfer
    fifo->fifo_pointers_matrix.size = 2;
    status = dpu_copy_to_wram_for_matrix(rank, &(fifo->fifo_pointers_matrix));
    swap_fifo_rd_wr_ptr(fifo);
    fifo->fifo_pointers_matrix.size = 4;

end:
    return status;
}

__API_SYMBOL__ dpu_error_t
dpu_copy_to_mram(struct dpu_t *dpu, mram_addr_t mram_byte_offset, const uint8_t *source, mram_size_t nb_of_bytes)
{
    if (!dpu->enabled) {
        return DPU_ERR_DPU_DISABLED;
    }

    if (!nb_of_bytes) {
        return DPU_OK;
    }

    dpu_error_t status;
    struct dpu_rank_t *rank = dpu_get_rank(dpu);
    struct dpu_transfer_matrix transfer_matrix;
    dpu_transfer_matrix_clear_all(rank, &transfer_matrix);

    dpu_transfer_matrix_add_dpu(dpu, &transfer_matrix, (void *)source);

    transfer_matrix.size = nb_of_bytes;
    transfer_matrix.offset = mram_byte_offset;
    status = dpu_copy_to_mrams(rank, &transfer_matrix);

    return status;
}

__API_SYMBOL__ dpu_error_t
dpu_copy_from_mram(struct dpu_t *dpu, uint8_t *destination, mram_addr_t mram_byte_offset, mram_size_t nb_of_bytes)
{
    if (!dpu->enabled) {
        return DPU_ERR_DPU_DISABLED;
    }

    if (!nb_of_bytes) {
        return DPU_OK;
    }

    dpu_error_t status;
    struct dpu_rank_t *rank = dpu_get_rank(dpu);
    struct dpu_transfer_matrix transfer_matrix;
    dpu_transfer_matrix_clear_all(rank, &transfer_matrix);

    dpu_transfer_matrix_add_dpu(dpu, &transfer_matrix, (void *)destination);

    transfer_matrix.size = nb_of_bytes;
    transfer_matrix.offset = mram_byte_offset;
    status = dpu_copy_from_mrams(rank, &transfer_matrix);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_to_mrams(struct dpu_rank_t *rank, struct dpu_transfer_matrix *transfer_matrix)
{
    LOG_RANK(VERBOSE, rank, "%p", transfer_matrix);
    dpu_error_t status;

    uint32_t offset = transfer_matrix->offset;
    uint32_t size = transfer_matrix->size;

    verify_mram_access_offset_and_size(offset, size, rank);

    dpu_lock_rank(rank);

    if (rank->runtime.run_context.nb_dpu_running > 0) {
        LOG_RANK(WARNING,
            rank,
            "Host does not have access to the MRAM because %u DPU%s running.",
            rank->runtime.run_context.nb_dpu_running,
            rank->runtime.run_context.nb_dpu_running > 1 ? "s are" : " is");
        dpu_unlock_rank(rank);
        return DPU_ERR_MRAM_BUSY;
    }

    if (rank->description->configuration.mram_access_by_dpu_only) {
        status = copy_to_mrams_using_dpu_program(rank, transfer_matrix);
    } else {
        status = RANK_FEATURE(rank, copy_to_mrams)(rank, transfer_matrix);
    }

    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ dpu_error_t
dpu_copy_from_mrams(struct dpu_rank_t *rank, struct dpu_transfer_matrix *transfer_matrix)
{
    LOG_RANK(VERBOSE, rank, "%p", transfer_matrix);
    dpu_error_t status;

    uint32_t offset = transfer_matrix->offset;
    uint32_t size = transfer_matrix->size;

    verify_mram_access_offset_and_size(offset, size, rank);

    dpu_lock_rank(rank);

    if (rank->runtime.run_context.nb_dpu_running > 0) {
        LOG_RANK(WARNING,
            rank,
            "Host does not have access to the MRAM because %u DPU%s running.",
            rank->runtime.run_context.nb_dpu_running,
            rank->runtime.run_context.nb_dpu_running > 1 ? "s are" : " is");
        dpu_unlock_rank(rank);
        return DPU_ERR_MRAM_BUSY;
    }

    if (rank->description->configuration.mram_access_by_dpu_only) {
        status = copy_from_mrams_using_dpu_program(rank, transfer_matrix);
    } else {
        status = RANK_FEATURE(rank, copy_from_mrams)(rank, transfer_matrix);
    }

    dpu_unlock_rank(rank);

    return status;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ void
dpu_transfer_matrix_add_dpu(struct dpu_t *dpu, struct dpu_transfer_matrix *transfer_matrix, void *buffer)
{
    LOG_DPU(DEBUG, dpu, "%p, %p", transfer_matrix, buffer);

    transfer_matrix->ptr[_transfer_matrix_index(dpu)] = buffer;
}

__PERF_PROFILING_SYMBOL__ dpu_error_t
dpu_transfer_matrix_add_dpu_block(struct dpu_t *dpu, struct dpu_transfer_matrix *transfer_matrix, void *buffer, uint32_t length)
{
    LOG_DPU(DEBUG, dpu, "%p, %p", transfer_matrix, buffer);

    if (transfer_matrix->type == DPU_SG_XFER_MATRIX) {
        struct sg_xfer_buffer *sg_ptr = transfer_matrix->sg_ptr[_transfer_matrix_index(dpu)];
        if (sg_ptr->nr_blocks >= sg_ptr->max_nr_blocks)
            return DPU_ERR_INTERNAL;
        sg_ptr->blocks_addr[sg_ptr->nr_blocks] = buffer;
        sg_ptr->blocks_length[sg_ptr->nr_blocks] = length;
        sg_ptr->nr_blocks++;
    } else {
        return DPU_ERR_INTERNAL;
    }

    return DPU_OK;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ void
dpu_transfer_matrix_set_all(struct dpu_rank_t *rank, struct dpu_transfer_matrix *transfer_matrix, void *buffer)
{
    LOG_RANK(DEBUG, rank, "%p, %p", transfer_matrix, buffer);
    uint8_t nr_of_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
    uint8_t nr_of_cis = rank->description->hw.topology.nr_of_control_interfaces;

    for (dpu_slice_id_t each_ci = 0; each_ci < nr_of_cis; ++each_ci) {
        for (dpu_member_id_t each_dpu = 0; each_dpu < nr_of_dpus_per_ci; ++each_dpu) {
            struct dpu_t *dpu = DPU_GET_UNSAFE(rank, each_ci, each_dpu);

            dpu_transfer_matrix_add_dpu(dpu, transfer_matrix, dpu->enabled ? buffer : NULL);
        }
    }
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ void
dpu_transfer_matrix_clear_dpu(struct dpu_t *dpu, struct dpu_transfer_matrix *transfer_matrix)
{
    LOG_DPU(DEBUG, dpu, "%p", __func__, dpu_get_id(dpu), transfer_matrix);

    transfer_matrix->ptr[_transfer_matrix_index(dpu)] = NULL;
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ void
dpu_transfer_matrix_clear_all(struct dpu_rank_t *rank, struct dpu_transfer_matrix *transfer_matrix)
{
    LOG_RANK(DEBUG, rank, "%p", transfer_matrix);

    memset(transfer_matrix, 0, sizeof(struct dpu_transfer_matrix));
}

__PERF_PROFILING_SYMBOL__ __API_SYMBOL__ void
dpu_transfer_matrix_copy(struct dpu_transfer_matrix *dst, struct dpu_transfer_matrix *src)
{
    LOG_FN(DEBUG, "%p %p", dst, src);

    memcpy(dst, src, sizeof(struct dpu_transfer_matrix));
}

__API_SYMBOL__ void *
dpu_transfer_matrix_get_ptr(struct dpu_t *dpu, struct dpu_transfer_matrix *transfer_matrix)
{
    LOG_DPU(DEBUG, dpu, "%p", transfer_matrix);

    if (!dpu->enabled) {
        LOG_DPU(WARNING, dpu, "dpu is disabled");

        return NULL;
    }

    uint32_t dpu_index = _transfer_matrix_index(dpu);

    return transfer_matrix->ptr[dpu_index];
}

static dpu_error_t
copy_from_mrams_using_dpu_program(struct dpu_rank_t *rank, const struct dpu_transfer_matrix *transfer_matrix)
{
    LOG_RANK(VERBOSE, rank, "%p", transfer_matrix);
    uint8_t nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
    uint8_t nr_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
    int idx = 0;
    dpu_error_t status = DPU_OK;
    bool arrays_initialized = false;
    dpuinstruction_t *program = NULL;
    dpuinstruction_t *iram_save = NULL;
    dpuword_t *wram_save = NULL;
    iram_size_t nr_saved_instructions = 0;
    wram_size_t wram_size_in_words = 0;
    dpuword_t *mram_buffer = NULL;

    for (dpu_member_id_t dpu_id = 0; dpu_id < nr_dpus_per_ci; ++dpu_id) {
        for (dpu_slice_id_t slice_id = 0; slice_id < nr_cis; ++slice_id, ++idx) {
            void *ptr = transfer_matrix->ptr[idx];
            const struct dpu_transfer_matrix *actual_transfer;
            struct dpu_transfer_matrix aligned_transfer;
            bool copy_needed;
            dpu_transfer_matrix_clear_all(rank, &aligned_transfer);

            if (ptr == NULL) {
                continue;
            }

            struct dpu_t *dpu = DPU_GET_UNSAFE(rank, slice_id, dpu_id);

            if (!dpu->enabled) {
                continue;
            }

            if (!arrays_initialized) {
                program = fetch_mram_access_program(&nr_saved_instructions);

                if (program == NULL) {
                    status = DPU_ERR_SYSTEM;
                    goto free_buffer;
                }

                uint32_t nr_saved_iram_bytes = nr_saved_instructions * sizeof(dpuinstruction_t);

                if ((iram_save = malloc(nr_saved_iram_bytes)) == NULL) {
                    status = DPU_ERR_SYSTEM;
                    free(program);
                    goto free_buffer;
                }

                wram_size_in_words = rank->description->hw.memories.wram_size;

                if ((wram_save = malloc(wram_size_in_words * sizeof(*wram_save))) == NULL) {
                    status = DPU_ERR_SYSTEM;
                    free(iram_save);
                    free(program);
                    goto free_buffer;
                }

                arrays_initialized = true;
            }

            if (((((unsigned long)ptr) & 3l) != 0) || ((transfer_matrix->offset & 7) != 0)
                || ((transfer_matrix->size & 7) != 0)) {
                uint32_t aligned_size = (transfer_matrix->size + (transfer_matrix->offset & 7) + 7) & ~7;

                if ((mram_buffer = realloc(mram_buffer, aligned_size)) == NULL) {
                    status = DPU_ERR_SYSTEM;
                    goto free_arrays;
                }

                aligned_transfer.size = aligned_size;
                aligned_transfer.offset = transfer_matrix->offset & ~7;
                aligned_transfer.ptr[0] = mram_buffer;
                actual_transfer = &aligned_transfer;
                copy_needed = true;
            } else {
                actual_transfer = transfer_matrix;
                copy_needed = false;
            }

            if ((status = access_mram_using_dpu_program_individual(dpu,
                     actual_transfer,
                     DPU_TRANSFER_FROM_MRAM,
                     program,
                     iram_save,
                     nr_saved_instructions,
                     wram_save,
                     wram_size_in_words))
                != DPU_OK) {
                goto free_arrays;
            }

            if (copy_needed) {
                memcpy(ptr, ((uint8_t *)mram_buffer) + (transfer_matrix->offset & 7), transfer_matrix->size);
            }
        }
    }

free_arrays:
    if (arrays_initialized) {
        free(program);
        free(iram_save);
        free(wram_save);
    }
free_buffer:
    if (mram_buffer != NULL) {
        free(mram_buffer);
    }
    return status;
}

static dpu_error_t
copy_to_mrams_using_dpu_program(struct dpu_rank_t *rank, const struct dpu_transfer_matrix *transfer_matrix)
{
    uint8_t nr_cis = rank->description->hw.topology.nr_of_control_interfaces;
    uint8_t nr_dpus_per_ci = rank->description->hw.topology.nr_of_dpus_per_control_interface;
    int idx = 0;
    dpu_error_t status = DPU_OK;
    bool arrays_initialized = false;
    dpuinstruction_t *program = NULL;
    dpuinstruction_t *iram_save = NULL;
    dpuword_t *wram_save = NULL;
    iram_size_t nr_saved_instructions = 0;
    wram_size_t wram_size_in_words = 0;
    dpuword_t *mram_buffer = NULL;

    for (dpu_member_id_t dpu_id = 0; dpu_id < nr_dpus_per_ci; ++dpu_id) {
        for (dpu_slice_id_t slice_id = 0; slice_id < nr_cis; ++slice_id, ++idx) {
            void *ptr = transfer_matrix->ptr[idx];
            const struct dpu_transfer_matrix *actual_transfer;
            struct dpu_transfer_matrix aligned_transfer;
            dpu_transfer_matrix_clear_all(rank, &aligned_transfer);

            if (ptr == NULL) {
                continue;
            }

            struct dpu_t *dpu = DPU_GET_UNSAFE(rank, slice_id, dpu_id);

            if (!dpu->enabled) {
                continue;
            }

            if (!arrays_initialized) {
                program = fetch_mram_access_program(&nr_saved_instructions);

                if (program == NULL) {
                    status = DPU_ERR_SYSTEM;
                    goto free_buffer;
                }

                uint32_t nr_saved_iram_bytes = nr_saved_instructions * sizeof(dpuinstruction_t);

                if ((iram_save = malloc(nr_saved_iram_bytes)) == NULL) {
                    status = DPU_ERR_SYSTEM;
                    free(program);
                    goto free_buffer;
                }

                wram_size_in_words = rank->description->hw.memories.wram_size;

                if ((wram_save = malloc(wram_size_in_words * sizeof(*wram_save))) == NULL) {
                    status = DPU_ERR_SYSTEM;
                    free(iram_save);
                    free(program);
                    goto free_buffer;
                }

                arrays_initialized = true;
            }

            if (((((unsigned long)ptr) & 3l) != 0) || ((transfer_matrix->offset & 7) != 0)
                || ((transfer_matrix->size & 7) != 0)) {
                uint32_t aligned_size = (transfer_matrix->size + (transfer_matrix->offset & 7) + 7) & ~7;

                if ((mram_buffer = realloc(mram_buffer, aligned_size)) == NULL) {
                    status = DPU_ERR_SYSTEM;
                    goto free_arrays;
                }

                if ((transfer_matrix->offset & 7) != 0) {
                    aligned_transfer.offset = transfer_matrix->offset & ~7;
                    aligned_transfer.size = 8;
                    aligned_transfer.ptr[0] = mram_buffer;

                    if ((status = access_mram_using_dpu_program_individual(dpu,
                             &aligned_transfer,
                             DPU_TRANSFER_FROM_MRAM,
                             program,
                             iram_save,
                             nr_saved_instructions,
                             wram_save,
                             wram_size_in_words))
                        != DPU_OK) {
                        goto free_arrays;
                    }
                }

                if (((transfer_matrix->size + (transfer_matrix->offset & 7)) & 7) != 0) {
                    aligned_transfer.offset = transfer_matrix->offset + aligned_size - 8;
                    aligned_transfer.size = 8;
                    aligned_transfer.ptr[0] = ((uint8_t *)mram_buffer) + aligned_size - 8;

                    if ((status = access_mram_using_dpu_program_individual(dpu,
                             &aligned_transfer,
                             DPU_TRANSFER_FROM_MRAM,
                             program,
                             iram_save,
                             nr_saved_instructions,
                             wram_save,
                             wram_size_in_words))
                        != DPU_OK) {
                        goto free_arrays;
                    }
                }

                memcpy(((uint8_t *)mram_buffer) + (transfer_matrix->offset & 7), ptr, transfer_matrix->size);

                aligned_transfer.offset = transfer_matrix->offset & ~7;
                aligned_transfer.size = aligned_size;
                aligned_transfer.ptr[0] = mram_buffer;
                actual_transfer = &aligned_transfer;
            } else {
                actual_transfer = transfer_matrix;
            }

            if ((status = access_mram_using_dpu_program_individual(dpu,
                     actual_transfer,
                     DPU_TRANSFER_TO_MRAM,
                     program,
                     iram_save,
                     nr_saved_instructions,
                     wram_save,
                     wram_size_in_words))
                != DPU_OK) {
                goto free_arrays;
            }
        }
    }

free_arrays:
    if (arrays_initialized) {
        free(program);
        free(iram_save);
        free(wram_save);
    }
free_buffer:
    if (mram_buffer != NULL) {
        free(mram_buffer);
    }

    return status;
}

static dpu_error_t
access_mram_using_dpu_program_individual(struct dpu_t *dpu,
    const struct dpu_transfer_matrix *transfer,
    dpu_transfer_type_t transfer_type,
    dpuinstruction_t *program,
    dpuinstruction_t *iram_save,
    iram_size_t nr_instructions,
    dpuword_t *wram_save,
    wram_size_t wram_size)
{
    /*
     * Preconditions:
     *  - transfer->ptr[0]          must be aligned on 4 bytes
     *  - transfer->size            must be aligned on 8 bytes
     *  - transfer->offset          must be aligned on 8 bytes
     */

    if (!dpu->enabled) {
        return DPU_ERR_DPU_DISABLED;
    }

#define PROGRAM_CONTEXT_WRAM_SIZE 8
    dpu_error_t status = DPU_OK;
    struct dpu_rank_t *rank = dpu_get_rank(dpu);
    struct dpu_context_t context;
    bool context_init = false;

    uint32_t ptr_offset;

    dpu_lock_rank(rank);

    if (transfer->size == 0) {
        goto end;
    }

    FF(dpu_custom_for_dpu(dpu, DPU_COMMAND_EVENT_START, (dpu_custom_command_args_t)DPU_EVENT_MRAM_ACCESS_PROGRAM));

#define DMA_ACCESS_IDX_1 11
#define DMA_ACCESS_IDX_2 19
#define DMA_ACCESS_WRAM_REG REG_R2
#define DMA_ACCESS_MRAM_REG REG_R0
#define DMA_ACCESS_LENGTH_1 255
#define DMA_ACCESS_LENGTH_2 0

    // Patch DMA access instructions
    if (transfer_type == DPU_TRANSFER_TO_MRAM) {
        program[DMA_ACCESS_IDX_1] = SDMArri(DMA_ACCESS_WRAM_REG, DMA_ACCESS_MRAM_REG, DMA_ACCESS_LENGTH_1);
        program[DMA_ACCESS_IDX_2] = SDMArri(DMA_ACCESS_WRAM_REG, DMA_ACCESS_MRAM_REG, DMA_ACCESS_LENGTH_2);
    } else {
        program[DMA_ACCESS_IDX_1] = LDMArri(DMA_ACCESS_WRAM_REG, DMA_ACCESS_MRAM_REG, DMA_ACCESS_LENGTH_1);
        program[DMA_ACCESS_IDX_2] = LDMArri(DMA_ACCESS_WRAM_REG, DMA_ACCESS_MRAM_REG, DMA_ACCESS_LENGTH_2);
    }

    wram_size_t wram_save_size;

    if (((transfer->size / sizeof(dpuword_t)) + PROGRAM_CONTEXT_WRAM_SIZE) >= wram_size) {
        wram_save_size = wram_size;
    } else {
        wram_save_size = (transfer->size / sizeof(dpuword_t)) + PROGRAM_CONTEXT_WRAM_SIZE;
    }

    FF(dpu_context_fill_from_rank(&context, rank));
    context.nr_of_running_threads = 0;
    for (dpu_thread_t each_thread = 0; each_thread < rank->description->hw.dpu.nr_of_threads; ++each_thread) {
        context.scheduling[each_thread] = 0xff;
    }
    context_init = true;
    FF(dpu_initialize_fault_process_for_dpu(dpu, &context, (mram_addr_t)0 /*nullptr*/));
    FF(dpu_extract_pcs_for_dpu(dpu, &context));
    FF(dpu_extract_context_for_dpu(dpu, &context));

    // Saving IRAM & WRAM
    FF(dpu_copy_from_iram_for_dpu(dpu, iram_save, 0, nr_instructions));
    FF(dpu_copy_from_wram_for_dpu(dpu, wram_save, 0, wram_save_size));

    // Loading DPU program in IRAM
    FF(dpu_copy_to_iram_for_dpu(dpu, 0, program, nr_instructions));

    wram_size_t transfer_size = wram_save_size - PROGRAM_CONTEXT_WRAM_SIZE;
    wram_addr_t transfer_start = PROGRAM_CONTEXT_WRAM_SIZE;

    uint32_t transfer_size_in_bytes = transfer_size * sizeof(dpuword_t);
    uint32_t nr_complete_iterations = transfer->size / transfer_size_in_bytes;
    uint32_t remaining = transfer->size % transfer_size_in_bytes;

    // Write context
    dpuword_t program_context[3] = {
        transfer->offset, // MRAM offset
        transfer_size * sizeof(dpuword_t), // Buffer size
        ((nr_complete_iterations == 0) || ((nr_complete_iterations == 1) && (remaining == 0))) ? 1 : 0 // Last transfer marker
    };

    FF(dpu_copy_to_wram_for_dpu(dpu, 5, program_context, 3));
    if (transfer_type == DPU_TRANSFER_TO_MRAM) {
        // Loading WRAM
        FF(dpu_copy_to_wram_for_dpu(dpu, transfer_start, transfer->ptr[0], transfer_size));
    }

    // Program execution
    FF(dpu_thread_boot_safe_for_dpu(dpu, 0, NULL, false));

    // Polling
    bool run, fault;
    do {
        FF(dpu_poll_dpu(dpu, &run, &fault));
    } while (run && !fault);

    if (fault) {
        LOG_DPU(WARNING, dpu, "MRAM copy program has faulted");
        status = DPU_ERR_INTERNAL;
        goto end;
    }

    if (transfer_type == DPU_TRANSFER_FROM_MRAM) {
        // Fetching WRAM
        FF(dpu_copy_from_wram_for_dpu(dpu, transfer->ptr[0], transfer_start, transfer_size));
    }
    ptr_offset = transfer_size * sizeof(dpuword_t);

    for (uint32_t each_iteration = 1; each_iteration < nr_complete_iterations; ++each_iteration) {
        if (transfer_type == DPU_TRANSFER_TO_MRAM) {
            // Loading WRAM
            FF(dpu_copy_to_wram_for_dpu(dpu, transfer_start, transfer->ptr[0] + ptr_offset, transfer_size));
        }

        if ((each_iteration == (nr_complete_iterations - 1)) && (remaining == 0)) {
            dpuword_t last_transfer_marker = 1;
            FF(dpu_copy_to_wram_for_dpu(dpu, 7, &last_transfer_marker, 1));
        }

        FF(dpu_thread_boot_safe_for_dpu(dpu, 0, NULL, true));

        // Polling
        do {
            FF(dpu_poll_dpu(dpu, &run, &fault));
        } while (run && !fault);

        if (fault) {
            LOG_DPU(WARNING, dpu, "MRAM copy program has faulted");
            status = DPU_ERR_INTERNAL;
            goto end;
        }

        if (transfer_type == DPU_TRANSFER_FROM_MRAM) {
            // Fetching WRAM
            FF(dpu_copy_from_wram_for_dpu(dpu, transfer->ptr[0] + ptr_offset, transfer_start, transfer_size));
        }
        ptr_offset += transfer_size * sizeof(dpuword_t);
    }

    if (remaining > 0) {
        if (nr_complete_iterations > 0) {
            // Changing buffer size in the DPU program
            transfer_size = remaining / sizeof(dpuword_t);
            program_context[1] = remaining;
            program_context[2] = 1;

            FF(dpu_copy_to_wram_for_dpu(dpu, 6, program_context + 1, 2));

            if (transfer_type == DPU_TRANSFER_TO_MRAM) {
                // Loading WRAM
                FF(dpu_copy_to_wram_for_dpu(dpu, transfer_start, transfer->ptr[0] + ptr_offset, transfer_size));
            }

            FF(dpu_thread_boot_safe_for_dpu(dpu, 0, NULL, true));

            // Polling
            do {
                FF(dpu_poll_dpu(dpu, &run, &fault));
            } while (run && !fault);

            if (fault) {
                LOG_DPU(WARNING, dpu, "MRAM copy program has faulted");
                status = DPU_ERR_INTERNAL;
                goto end;
            }

            if (transfer_type == DPU_TRANSFER_FROM_MRAM) {
                // Fetching WRAM
                FF(dpu_copy_from_wram_for_dpu(dpu, transfer->ptr[0] + ptr_offset, transfer_start, transfer_size));
            }
        }
    }

    // Restoring IRAM & WRAM
    FF(dpu_copy_to_iram_for_dpu(dpu, 0, iram_save, nr_instructions));
    FF(dpu_copy_to_wram_for_dpu(dpu, 0, wram_save, wram_save_size));

    // Restoring context
    FF(dpu_restore_context_for_dpu(dpu, &context));
    FF(dpu_finalize_fault_process_for_dpu(dpu, &context));

    FF(dpu_custom_for_dpu(dpu, DPU_COMMAND_EVENT_END, (dpu_custom_command_args_t)DPU_EVENT_MRAM_ACCESS_PROGRAM));

end:
    if (context_init) {
        dpu_free_dpu_context(&context);
    }
    dpu_unlock_rank(rank);
    return status;
}
