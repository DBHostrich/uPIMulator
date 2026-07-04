/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libudev.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "dpu_description.h"
#include "dpu_attributes.h"

/* Header shared with driver */
#include "dpu_region_address_translation.h"
#include "hw_dpu_sysfs.h"

#include "dpu_log_utils.h"
#include "static_verbose.h"

/* maximum possible number of ranks on an icelake server
 * with 2 sockets and 8 channels per socket */
#define MAX_NR_DEVICES 64

static struct verbose_control *this_vc;
static struct verbose_control *
__vc()
{
    if (this_vc == NULL) {
        this_vc = get_verbose_control_for("hw");
    }
    return this_vc;
}

struct udev_cache_t {
    pthread_mutex_t lock;
    struct udev *udev;
    struct udev_enumerate *enumerate;
    struct udev_device *rank_devices[MAX_NR_DEVICES];
    struct udev_device *dax_devices[MAX_NR_DEVICES];
    size_t devices_count;
    bool valid;
};

static struct udev_cache_t *
gbl_udev_cache()
{
    static struct udev_cache_t gbl_udev_cache = { .valid = false, .lock = PTHREAD_MUTEX_INITIALIZER };
    return &gbl_udev_cache;
}

#define dpu_sys_get_integer_sysattr(name, udev, type, format, default_value)                                                     \
    const char *str;                                                                                                             \
    type value;                                                                                                                  \
                                                                                                                                 \
    str = udev_device_get_sysattr_value(rank_fs->udev, name);                                                                    \
    if (str == NULL)                                                                                                             \
        return (type)(default_value);                                                                                            \
                                                                                                                                 \
    (void)sscanf(str, format, &value);                                                                                           \
                                                                                                                                 \
    return value;

#define dpu_sys_set_integer_sysattr(name, udev, value, format)                                                                   \
    char str[32];                                                                                                                \
                                                                                                                                 \
    (void)sprintf(str, format, value);                                                                                           \
                                                                                                                                 \
    return udev_device_set_sysattr_value(rank_fs->udev, name, str) < 0;

#define dpu_sys_get_string_sysattr(name, udev)                                                                                   \
    const char *str;                                                                                                             \
                                                                                                                                 \
    str = udev_device_get_sysattr_value(rank_fs->udev, name);                                                                    \
    if (str == NULL)                                                                                                             \
        return NULL;                                                                                                             \
                                                                                                                                 \
    return str;

/* In this case, sscanf is preferred over ato* functions because it is more generic. */
// NOLINTBEGIN(cert-err34-c)
uint64_t
dpu_sysfs_get_region_size(struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("size", udev_dax, uint64_t, "%" SCNu64, 0) }

uint8_t
    dpu_sysfs_get_channel_id(struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("channel_id", udev, uint8_t, "%hhu", -1) }

uint8_t dpu_sysfs_get_rank_id(struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("rank_id", udev, uint8_t, "%hhu", 0) }

uint8_t
    dpu_sysfs_get_backend_id(struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("backend_id", udev, uint8_t, "%hhu", 0) }

uint8_t dpu_sysfs_get_dpu_chip_id(
    struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("dpu_chip_id", udev, uint8_t, "%hhu", 0) }

uint8_t dpu_sysfs_get_nb_ci(struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("nb_ci", udev, uint8_t, "%hhu", 0) }

uint8_t dpu_sysfs_get_ci_mask(struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("ci_mask", udev, uint8_t, "%hhu", 0) }

uint8_t dpu_sysfs_get_nb_dpus_per_ci(
    struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("nb_dpus_per_ci", udev, uint8_t, "%hhu", 0) }

uint32_t
    dpu_sysfs_get_mram_size(struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("mram_size", udev, uint32_t, "%u", 0) }

uint64_t dpu_sysfs_get_capabilities(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_integer_sysattr("capabilities", udev, uint64_t, "%" SCNx64, 0)
}

int
dpu_sysfs_set_reset_ila(struct dpu_rank_fs *rank_fs, uint8_t val) { dpu_sys_set_integer_sysattr("reset_ila", udev, val, "%hhu") }

uint32_t dpu_sysfs_get_activate_ila(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_integer_sysattr("activate_ila", udev, uint8_t, "%hhu", 0)
}

int
dpu_sysfs_set_activate_ila(struct dpu_rank_fs *rank_fs,
    uint8_t val) { dpu_sys_set_integer_sysattr("activate_ila", udev, val, "%hhu") }

uint32_t dpu_sysfs_get_activate_filtering_ila(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_integer_sysattr("activate_filtering_ila", udev, uint8_t, "%hhu", 0)
}

int
dpu_sysfs_set_activate_filtering_ila(struct dpu_rank_fs *rank_fs,
    uint8_t val) { dpu_sys_set_integer_sysattr("activate_filtering_ila", udev, val, "%hhu") }

uint32_t dpu_sysfs_get_activate_mram_bypass(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_integer_sysattr("activate_mram_bypass", udev, uint8_t, "%hhu", 0)
}

int
dpu_sysfs_set_activate_mram_bypass(struct dpu_rank_fs *rank_fs,
    uint8_t val) { dpu_sys_set_integer_sysattr("activate_mram_bypass", udev, val, "%hhu") }

uint32_t dpu_sysfs_get_mram_refresh_emulation_period(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_integer_sysattr("mram_refresh_emulation_period", udev, uint32_t, "%u", 0)
}

int
dpu_sysfs_set_mram_refresh_emulation_period(struct dpu_rank_fs *rank_fs, uint32_t val)
{
    dpu_sys_set_integer_sysattr("mram_refresh_emulation_period", udev, val, "%u")
}

int
dpu_sysfs_set_inject_faults(struct dpu_rank_fs *rank_fs,
    uint8_t val) { dpu_sys_set_integer_sysattr("inject_faults", udev, val, "%hhu") }

uint32_t dpu_sysfs_get_fck_frequency(
    struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("fck_frequency", udev, uint32_t, "%u", 0) }

uint32_t dpu_sysfs_get_clock_division(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_integer_sysattr("clock_division", udev, uint32_t, "%u", 0)
}

int
dpu_sysfs_get_numa_node(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_integer_sysattr("numa_node", udev_dax, int, "%d", 0)
}

const char *
dpu_sysfs_get_byte_order(struct dpu_rank_fs *rank_fs)
{
    dpu_sys_get_string_sysattr("byte_order", udev);
}

uint8_t
dpu_sysfs_get_rank_index(
    struct dpu_rank_fs *rank_fs) { dpu_sys_get_integer_sysattr("rank_index", udev, uint8_t, "%hhu", DPU_INVALID_RANK_INDEX) }
// NOLINTEND(cert-err34-c)

__API_SYMBOL__ void invalidate_udev_cache(struct udev_cache_t *udev_cache)
{
    pthread_mutex_lock(&udev_cache->lock);
    for (size_t i = 0; i < udev_cache->devices_count; i++) {
        if (udev_cache->rank_devices[i]) {
            udev_device_unref(udev_cache->rank_devices[i]);
            udev_cache->rank_devices[i] = NULL;
        }
        if (udev_cache->dax_devices[i]) {
            udev_device_unref(udev_cache->dax_devices[i]);
            udev_cache->dax_devices[i] = NULL;
        }
    }
    if (udev_cache->enumerate) {
        udev_enumerate_unref(udev_cache->enumerate);
        udev_cache->enumerate = NULL;
    }
    if (udev_cache->udev) {
        udev_unref(udev_cache->udev);
        udev_cache->udev = NULL;
    }
    udev_cache->valid = false;
    pthread_mutex_unlock(&udev_cache->lock);
}

__attribute__((destructor)) static void
clear_udev_cache()
{
    invalidate_udev_cache(gbl_udev_cache());
}

static int
init_cached_udev_enumerates(struct udev_cache_t *udev_cache)
{
    udev_cache->udev = udev_new();
    if (!udev_cache->udev) {
        LOG_FN(WARNING, "can't create udev");
        goto err;
    }
    udev_cache->enumerate = udev_enumerate_new(udev_cache->udev);
    if (!udev_cache->enumerate) {
        LOG_FN(WARNING, "can't create udev enumerate");
        goto cleanup_udev;
    }
    udev_enumerate_add_match_subsystem(udev_cache->enumerate, "dpu_rank");
    udev_enumerate_add_match_subsystem(udev_cache->enumerate, "dpu_dax");

    if (udev_enumerate_scan_devices(udev_cache->enumerate)) {
        LOG_FN(WARNING, "can't scan devices");
        goto cleanup_enumerate;
    }
    struct udev_list_entry *devices = udev_enumerate_get_list_entry(udev_cache->enumerate);
    if (!devices) {
        LOG_FN(WARNING, "can't get devices list");
        goto cleanup_enumerate;
    }

    // Populate the device array
    struct udev_list_entry *entry = NULL;
    size_t index_ranks = 0;
    size_t index_dax = 0;
    udev_list_entry_foreach(entry, devices)
    {
        assert(index_ranks < MAX_NR_DEVICES && "too many devices, adjust MAX_NR_DEVICES in source code");
        const char *path = udev_list_entry_get_name(entry);
        struct udev_device *device = udev_device_new_from_syspath(udev_cache->udev, path);
        if (!device) {
            LOG_FN(WARNING, "can't create udev device from syspath");
            goto cleanup_devices;
        }
        const char *device_subsystem = udev_device_get_subsystem(device);
        if (!device_subsystem) {
            LOG_FN(WARNING, "can't get subsystem of udev device");
            udev_device_unref(device);
            goto cleanup_devices;
        }
        if (strcmp(device_subsystem, "dpu_rank") == 0) {
            udev_cache->rank_devices[index_ranks] = device;
            index_ranks++;
        } else if (strcmp(device_subsystem, "dpu_dax") == 0) {
            udev_cache->dax_devices[index_dax] = device;
            index_dax++;
        } else {
            assert(false && "unexpected subsystem");
        }
    }

    if (index_ranks != index_dax && index_dax != 0) {
        LOG_FN(WARNING, "number of dpu ranks and dax devices do not match");
        goto cleanup_devices;
    }

    udev_cache->devices_count = index_ranks;
    udev_cache->valid = true;
    return 0;

cleanup_devices:
    for (size_t i = 0; i < index_ranks; i++) {
        udev_device_unref(udev_cache->rank_devices[i]);
        udev_cache->rank_devices[i] = NULL;
    }
    for (size_t i = 0; i < index_dax; i++) {
        udev_device_unref(udev_cache->dax_devices[i]);
        udev_cache->dax_devices[i] = NULL;
    }
cleanup_enumerate:
    udev_enumerate_unref(udev_cache->enumerate);
    udev_cache->enumerate = NULL;
cleanup_udev:
    udev_unref(udev_cache->udev);
    udev_cache->udev = NULL;
err:
    return -1;
}

static struct udev_cache_t *
init_udev_enumerator(struct udev_cache_t *udev_cache)
{
    /* double-checked locking */
    if (!udev_cache->valid) {
        pthread_mutex_lock(&udev_cache->lock);
        if (!udev_cache->valid) {
            assert(udev_cache->udev == NULL && "cache should be uninitialized");
            if (init_cached_udev_enumerates(udev_cache) < 0) {
                LOG_FN(WARNING, "can't create udev enumerate cache");
            }
        }
        pthread_mutex_unlock(&udev_cache->lock);
    }
    return udev_cache;
}

static bool
has_specified_parent(struct udev_device *device, const char *specified_parent_sysname)
{
    struct udev_device *parent = udev_device_get_parent(device);
    if (parent == NULL) {
        LOG_FN(WARNING, "No parent for device");
        return false;
    }
    const char *actual_parent_sysname = udev_device_get_sysname(parent);
    bool result = strcmp(actual_parent_sysname, specified_parent_sysname) == 0;
    return result;
}

static int
dpu_sysfs_try_to_allocate_rank(const char *dev_rank_path, struct dpu_rank_fs *rank_fs)
{
    /* Whatever the mode, we keep an fd to dpu_rank so that
     * we have infos about how/who uses the rank
     */
    rank_fs->fd_rank = open(dev_rank_path, O_RDWR);
    if (rank_fs->fd_rank < 0) {
        return -errno;
    }

    /* udev_device_get_parent does not take a reference as stated in header */
    struct udev_device *udev_parent = udev_device_get_parent(rank_fs->udev);
    const char *parent_sysname = udev_device_get_sysname(udev_parent);

    /* Dax device only exists if backend supports PERF mode */
    uint64_t capabilities = dpu_sysfs_get_capabilities(rank_fs);

    if (capabilities & CAP_PERF) {
        /* There's only one dax device associated to the region,
         * we scan dax devices once and filter on the parent
         * sysname to get the right one.
         */
        struct udev_cache_t *udev_cache = init_udev_enumerator(gbl_udev_cache());
        if (!udev_cache->valid) {
            LOG_FN(WARNING, "Error initializing udev enumerator");
            goto err;
        }

        for (size_t i = 0; i < udev_cache->devices_count; i++) {
            const char *dev_dax_path;

            rank_fs->udev_dax = udev_cache->dax_devices[i];
            assert(udev_device_get_is_initialized(rank_fs->udev_dax) && "dax device is not initialized");
            if (!has_specified_parent(rank_fs->udev_dax, parent_sysname)) {
                continue;
            }
            dev_dax_path = udev_device_get_devnode(rank_fs->udev_dax);

            rank_fs->fd_dax = open(dev_dax_path, O_RDWR);
            if (rank_fs->fd_dax >= 0) {
                return 0;
            }

            LOG_FN(WARNING, "Error (%d: '%s') opening dax device '%s'", errno, MT_SAFE_STRERROR(errno), dev_dax_path);
        }
    } else {
        return 0;
    }

err:
    close(rank_fs->fd_rank);

    return -EINVAL;
}

void
dpu_sysfs_free_rank_fs(struct dpu_rank_fs *rank_fs)
{
    if (rank_fs->fd_dax) {
        close(rank_fs->fd_dax);
    }

    close(rank_fs->fd_rank);
}

void
dpu_sysfs_free_rank(struct dpu_rank_fs *rank_fs)
{
    dpu_sysfs_free_rank_fs(rank_fs);
}

uint8_t
dpu_sysfs_get_nb_physical_ranks()
{
    struct udev_cache_t *udev_cache = init_udev_enumerator(gbl_udev_cache());
    if (!udev_cache->valid) {
        LOG_FN(WARNING, "Error initializing udev enumerator");
        return 0;
    }

    return udev_cache->devices_count;
}

// TODO allocation must be smarter than just allocating rank "one by one":
// it is better (at memory bandwidth point of view) to allocate ranks
// from unused channels rather than allocating ranks of a same channel
// (memory bandwidth is limited at a channel level)
int
dpu_sysfs_get_available_rank(const char *rank_path, struct dpu_rank_fs *rank_fs)
{
    int eacces_count = 0;

    struct udev_cache_t *udev_cache = init_udev_enumerator(gbl_udev_cache());
    if (!udev_cache->valid) {
        LOG_FN(WARNING, "Error initializing udev enumerator");
        goto end;
    }

    for (size_t i = 0; i < udev_cache->devices_count; i++) {
        rank_fs->udev = udev_cache->rank_devices[i];
        assert(udev_device_get_is_initialized(rank_fs->udev) && "rank device is not initialized");
        const char *dev_rank_path = udev_device_get_devnode(rank_fs->udev);

        if (strlen(rank_path)) {
            if (!strcmp(dev_rank_path, rank_path)) {
                if (!dpu_sysfs_try_to_allocate_rank(dev_rank_path, rank_fs)) {
                    strcpy(rank_fs->rank_path, dev_rank_path);
                    return 0;
                }
                LOG_FN(WARNING, "Allocation of requested %s rank failed", rank_path);
                break;
            }
        } else {
            int res = dpu_sysfs_try_to_allocate_rank(dev_rank_path, rank_fs);
            if (!res) {
                strcpy(rank_fs->rank_path, dev_rank_path);
                return 0;
            }
            /* record whether we have found something that we cannot access */
            if (res == -EACCES) {
                eacces_count++;
            }
        }
    }

end:
    return eacces_count ? -EACCES : -ENODEV;
}

// We assume there is only one chip id per machine: so we just need to get that info
// from the first rank.
int
dpu_sysfs_get_hardware_chip_id(uint8_t *chip_id)
{
    struct dpu_rank_fs rank_fs;

    struct udev_cache_t *udev_cache = init_udev_enumerator(gbl_udev_cache());
    if (!udev_cache->valid) {
        LOG_FN(WARNING, "Error initializing udev enumerator");
        goto end;
    }

    rank_fs.udev = udev_cache->rank_devices[0];
    assert(udev_device_get_is_initialized(rank_fs.udev) && "rank device is not initialized");

    /* Get the chip id from the driver */
    *chip_id = dpu_sysfs_get_dpu_chip_id(&rank_fs);

    return 0;

end:
    return -1;
}

// We assume the topology is identical for all the ranks: so we just need to get that info
// from the first rank.
int
dpu_sysfs_get_hardware_description(dpu_description_t description, uint8_t *capabilities_mode)
{
    struct dpu_rank_fs rank_fs;

    struct udev_cache_t *udev_cache = init_udev_enumerator(gbl_udev_cache());
    if (!udev_cache->valid) {
        LOG_FN(WARNING, "Error initializing udev enumerator");
        goto end;
    }

    rank_fs.udev = udev_cache->rank_devices[0];
    assert(udev_device_get_is_initialized(rank_fs.udev) && "rank device is not initialized");

    /* Get the real topology from the driver */
    uint8_t clock_division = 0;
    uint32_t fck_frequency_in_mhz = 0;
    description->hw.topology.nr_of_dpus_per_control_interface = dpu_sysfs_get_nb_dpus_per_ci(&rank_fs);
    description->hw.topology.nr_of_control_interfaces = dpu_sysfs_get_nb_ci(&rank_fs);
    description->hw.memories.mram_size = dpu_sysfs_get_mram_size(&rank_fs);
    clock_division = dpu_sysfs_get_clock_division(&rank_fs);
    fck_frequency_in_mhz = dpu_sysfs_get_fck_frequency(&rank_fs);
    /* Keep clock_division and fck_frequency_in_mhz default value if sysfs returns 0 */
    if (clock_division) {
        description->hw.timings.clock_division = clock_division;
    }
    if (fck_frequency_in_mhz) {
        description->hw.timings.fck_frequency_in_mhz = fck_frequency_in_mhz;
    }

    *capabilities_mode = dpu_sysfs_get_capabilities(&rank_fs);

    return 0;

end:
    return -1;
}

int
dpu_sysfs_get_kernel_module_version(unsigned int *major, unsigned int *minor)
{
    FILE *fp = NULL;

    if ((fp = fopen("/sys/module/dpu/version", "r")) == NULL) {
        return -errno;
    }

    if (fscanf(fp, "%u.%u", major, minor) != 2) {
        int err = errno;
        (void)fclose(fp);
        return (err != 0) ? -err : -1; // errno can be 0 is there is no matching character
    }

    if (fclose(fp) == EOF) {
        return -errno;
    }
    return 0;
}
