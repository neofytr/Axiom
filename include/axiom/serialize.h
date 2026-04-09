/* axiom/serialize.h — model serialization.

   the binary format is designed for embedded deployment:
   - no external dependencies (no protobuf, no json)
   - fixed-size header, simple structure
   - little-endian throughout
   - can be read incrementally (no need to load entire file at once)

   format overview:
     [header]              magic + version + metadata
     [layer descriptors]   type, shape, flags for each layer
     [parameter data]      raw weight/bias bytes, densely packed */

#ifndef AX_SERIALIZE_H
#define AX_SERIALIZE_H

#include "layer.h"
#include "model.h"

/* magic bytes to identify axiom model files */
#define AX_MAGIC 0x41584F4E  /* "AXON" in ascii */
#define AX_FORMAT_VERSION 1

/* save a sequential model to a binary file.
   writes architecture + all parameter data.
   returns AX_OK on success. */
ax_status_t ax_model_save(ax_model_t *model, const char *path);

/* load a model from a binary file.
   reconstructs the layer tree and loads all weights.
   returns NULL on failure (check ax_err_last_message). */
ax_model_t *ax_model_load(const char *path);

/* save a single tensor to a file (raw binary with a small header).
   useful for exporting weights or data. */
ax_status_t ax_tensor_save(ax_tensor_t *t, const char *path);

/* load a tensor from a file saved by ax_tensor_save. */
ax_tensor_t *ax_tensor_load(const char *path);

#endif /* AX_SERIALIZE_H */
