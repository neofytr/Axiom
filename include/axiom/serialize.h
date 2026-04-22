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
#define AX_FORMAT_VERSION 4

/* save a sequential model to a binary file. writes architecture + all
   parameter data.
   ownership: model is borrowed (caller still owns it).
   returns AX_OK on success, AX_ERR_IO on file open/write errors,
   AX_ERR_INVALID for unsupported layer types.
   thread-safety: not concurrent-safe with mutations to the model;
   external locking required. */
ax_status_t ax_model_save(ax_model_t *model, const char *path);

/* load a model from a binary file. reconstructs the layer tree and
   loads all weights.
   ownership: caller owns the returned model; release with
   ax_model_destroy.
   returns NULL on read/parse failure or unknown format version;
   ax_err_last_message describes the cause.
   thread-safety: concurrent-safe across distinct paths. */
ax_model_t *ax_model_load(const char *path);

/* save a single tensor to a file (raw binary with a small header).
   useful for exporting weights or data.
   ownership: t is borrowed.
   returns AX_OK on success, AX_ERR_IO on filesystem errors. */
ax_status_t ax_tensor_save(ax_tensor_t *t, const char *path);

/* load a tensor from a file saved by ax_tensor_save.
   ownership: caller owns the returned tensor; release with
   ax_tensor_destroy.
   returns NULL on failure; check ax_err_last_message. */
ax_tensor_t *ax_tensor_load(const char *path);

#endif /* AX_SERIALIZE_H */
