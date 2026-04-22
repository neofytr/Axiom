/* axiom/device.h — generic device management api.

   device-class-agnostic wrappers that route through the backend
   registered as the owner of a given ax_device_t. cuda-specific
   helpers in cuda.h now forward here.

   ownership: queries only — no allocation, no destruction.
   error handling: queries return defaults for missing backends rather
   than failing (count == 0, available == 0, synchronize == no-op);
   no error state is set.
   thread-safety: all queries are concurrent-safe. */

#ifndef AX_DEVICE_H
#define AX_DEVICE_H

#include "types.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* number of physical devices of the given class that can be used.
   returns 0 for AX_DEVICE_CPU (not meaningful), or when the owning
   backend is not compiled in. */
int  ax_device_count(ax_device_t device);

/* block until all pending work on the given device completes.
   no-op for AX_DEVICE_CPU and for devices whose backend is absent. */
void ax_device_synchronize(ax_device_t device);

/* returns non-zero if a backend claims ownership of the given device.
   used by core code to decide whether non-cpu tensor paths can run. */
int  ax_device_is_available(ax_device_t device);

#ifdef __cplusplus
}
#endif

#endif /* AX_DEVICE_H */
