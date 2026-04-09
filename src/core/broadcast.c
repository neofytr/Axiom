/* broadcast.c — numpy-style broadcast shape computation */

#include "axiom/broadcast.h"
#include "axiom/error.h"

ax_status_t ax_broadcast_shape(
    const int64_t *a_shape, int a_ndim,
    const int64_t *b_shape, int b_ndim,
    int64_t *out_shape, int *out_ndim)
{
    int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    if (max_ndim > AX_MAX_DIMS)
    {
        ax_err_set(AX_ERR_INVALID_SHAPE, "broadcast result exceeds max dims");
        return AX_ERR_INVALID_SHAPE;
    }

    *out_ndim = max_ndim;

    /* walk right to left, comparing dimensions */
    for (int i = 0; i < max_ndim; i++)
    {
        int a_idx = a_ndim - 1 - i;
        int b_idx = b_ndim - 1 - i;
        int64_t da = (a_idx >= 0) ? a_shape[a_idx] : 1;
        int64_t db = (b_idx >= 0) ? b_shape[b_idx] : 1;

        if (da == db)
        {
            out_shape[max_ndim - 1 - i] = da;
        }
        else if (da == 1)
        {
            out_shape[max_ndim - 1 - i] = db;
        }
        else if (db == 1)
        {
            out_shape[max_ndim - 1 - i] = da;
        }
        else
        {
            ax_err_set(AX_ERR_SHAPE_MISMATCH,
                       "cannot broadcast dim %ld with %ld", da, db);
            return AX_ERR_SHAPE_MISMATCH;
        }
    }
    return AX_OK;
}

bool ax_broadcast_compatible(
    const int64_t *a_shape, int a_ndim,
    const int64_t *b_shape, int b_ndim)
{
    int64_t tmp[AX_MAX_DIMS];
    int tmp_ndim;
    return ax_broadcast_shape(a_shape, a_ndim, b_shape, b_ndim,
                              tmp, &tmp_ndim) == AX_OK;
}
