/* test_fuzz_serialize.c — feed malformed data to the serialization
   layer and verify it returns errors, never crashes.
   every test writes a crafted binary file and calls ax_model_load
   or ax_tensor_load. success = no crash + error returned. */

#include "test.h"
#include "axiom/axiom.h"
#include <stdio.h>
#include <string.h>

static void write_bytes(const char *path, const void *data, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(data, 1, len, f); fclose(f); }
}

/* truncated file: just the magic, nothing else */
static void test_fuzz_truncated_model(void)
{
    uint32_t magic = 0x41584F4E; /* AXON */
    write_bytes("/tmp/ax_fuzz_trunc.axm", &magic, sizeof(magic));

    ax_model_t *m = ax_model_load("/tmp/ax_fuzz_trunc.axm");
    AX_TEST_ASSERT(m == NULL, "truncated model should fail to load");
    remove("/tmp/ax_fuzz_trunc.axm");
}

/* wrong magic number */
static void test_fuzz_bad_magic(void)
{
    uint8_t data[64];
    memset(data, 0xFF, sizeof(data));
    write_bytes("/tmp/ax_fuzz_magic.axm", data, sizeof(data));

    ax_model_t *m = ax_model_load("/tmp/ax_fuzz_magic.axm");
    AX_TEST_ASSERT(m == NULL, "bad magic should fail");
    remove("/tmp/ax_fuzz_magic.axm");
}

/* version too high */
static void test_fuzz_future_version(void)
{
    uint32_t header[3] = { 0x41584F4E, 9999, 1 }; /* magic, version, 1 layer */
    write_bytes("/tmp/ax_fuzz_ver.axm", header, sizeof(header));

    ax_model_t *m = ax_model_load("/tmp/ax_fuzz_ver.axm");
    AX_TEST_ASSERT(m == NULL, "future version should fail");
    remove("/tmp/ax_fuzz_ver.axm");
}

/* absurd layer count */
static void test_fuzz_huge_layer_count(void)
{
    uint32_t header[3] = { 0x41584F4E, 4, 99999 }; /* 99999 layers */
    write_bytes("/tmp/ax_fuzz_layers.axm", header, sizeof(header));

    ax_model_t *m = ax_model_load("/tmp/ax_fuzz_layers.axm");
    AX_TEST_ASSERT(m == NULL, "huge layer count should fail");
    remove("/tmp/ax_fuzz_layers.axm");
}

/* tensor file with negative ndim */
static void test_fuzz_negative_ndim(void)
{
    struct {
        uint32_t magic;
        uint32_t dtype;
        uint32_t ndim;
    } __attribute__((packed)) header = { 0x41585430, 0, (uint32_t)-5 };
    write_bytes("/tmp/ax_fuzz_ndim.axt", &header, sizeof(header));

    ax_tensor_t *t = ax_tensor_load("/tmp/ax_fuzz_ndim.axt");
    AX_TEST_ASSERT(t == NULL, "negative ndim should fail");
    remove("/tmp/ax_fuzz_ndim.axt");
}

/* tensor with huge shape that would overflow numel */
static void test_fuzz_overflow_shape(void)
{
    struct {
        uint32_t magic;
        uint32_t dtype;
        uint32_t ndim;
        int64_t shape[2];
    } __attribute__((packed)) header = {
        0x41585430, 0, 2,
        { (int64_t)1 << 40, (int64_t)1 << 40 }
    };
    write_bytes("/tmp/ax_fuzz_overflow.axt", &header, sizeof(header));

    ax_tensor_t *t = ax_tensor_load("/tmp/ax_fuzz_overflow.axt");
    AX_TEST_ASSERT(t == NULL, "overflow shape should fail");
    remove("/tmp/ax_fuzz_overflow.axt");
}

/* tensor with negative shape dimension */
static void test_fuzz_negative_shape(void)
{
    struct {
        uint32_t magic;
        uint32_t dtype;
        uint32_t ndim;
        int64_t shape[2];
    } __attribute__((packed)) header = {
        0x41585430, 0, 2,
        { -1, 10 }
    };
    write_bytes("/tmp/ax_fuzz_negshape.axt", &header, sizeof(header));

    ax_tensor_t *t = ax_tensor_load("/tmp/ax_fuzz_negshape.axt");
    AX_TEST_ASSERT(t == NULL, "negative shape should fail");
    remove("/tmp/ax_fuzz_negshape.axt");
}

/* zero-byte file */
static void test_fuzz_empty_file(void)
{
    write_bytes("/tmp/ax_fuzz_empty.axm", "", 0);
    ax_model_t *m = ax_model_load("/tmp/ax_fuzz_empty.axm");
    AX_TEST_ASSERT(m == NULL, "empty file should fail");
    remove("/tmp/ax_fuzz_empty.axm");

    write_bytes("/tmp/ax_fuzz_empty.axt", "", 0);
    ax_tensor_t *t = ax_tensor_load("/tmp/ax_fuzz_empty.axt");
    AX_TEST_ASSERT(t == NULL, "empty tensor file should fail");
    remove("/tmp/ax_fuzz_empty.axt");
}

/* nonexistent file */
static void test_fuzz_missing_file(void)
{
    ax_model_t *m = ax_model_load("/tmp/ax_fuzz_doesnt_exist_12345.axm");
    AX_TEST_ASSERT(m == NULL, "missing model file should fail");

    ax_tensor_t *t = ax_tensor_load("/tmp/ax_fuzz_doesnt_exist_12345.axt");
    AX_TEST_ASSERT(t == NULL, "missing tensor file should fail");
}

/* valid magic but garbage payload */
static void test_fuzz_garbage_payload(void)
{
    uint8_t data[4096];
    uint32_t magic = 0x41584F4E;
    memcpy(data, &magic, 4);
    /* fill rest with random-ish bytes */
    for (int i = 4; i < 4096; i++) data[i] = (uint8_t)(i * 37 + 17);
    write_bytes("/tmp/ax_fuzz_garbage.axm", data, sizeof(data));

    ax_model_t *m = ax_model_load("/tmp/ax_fuzz_garbage.axm");
    /* may or may not load, but must not crash */
    if (m) ax_model_destroy(m);
    AX_TEST_ASSERT(1, "garbage payload did not crash");
    remove("/tmp/ax_fuzz_garbage.axm");
}

/* tensor with valid header but data truncated */
static void test_fuzz_truncated_tensor_data(void)
{
    struct {
        uint32_t magic;
        uint32_t dtype;
        uint32_t ndim;
        int64_t shape[1];
        float data[2]; /* only 2 floats but shape says 1000 */
    } __attribute__((packed)) file = {
        0x41585430, 0, 1,
        { 1000 },
        { 1.0f, 2.0f }
    };
    write_bytes("/tmp/ax_fuzz_truncdata.axt", &file, sizeof(file));

    ax_tensor_t *t = ax_tensor_load("/tmp/ax_fuzz_truncdata.axt");
    AX_TEST_ASSERT(t == NULL, "truncated tensor data should fail");
    remove("/tmp/ax_fuzz_truncdata.axt");
}

int main(void)
{
    ax_init();

    printf("=== serialization fuzz tests ===\n");
    AX_RUN_TEST(test_fuzz_truncated_model);
    AX_RUN_TEST(test_fuzz_bad_magic);
    AX_RUN_TEST(test_fuzz_future_version);
    AX_RUN_TEST(test_fuzz_huge_layer_count);
    AX_RUN_TEST(test_fuzz_negative_ndim);
    AX_RUN_TEST(test_fuzz_overflow_shape);
    AX_RUN_TEST(test_fuzz_negative_shape);
    AX_RUN_TEST(test_fuzz_empty_file);
    AX_RUN_TEST(test_fuzz_missing_file);
    AX_RUN_TEST(test_fuzz_garbage_payload);
    AX_RUN_TEST(test_fuzz_truncated_tensor_data);

    ax_shutdown();
    AX_TEST_SUMMARY();
}
