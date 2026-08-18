# TT-SVD (Quantized MPS / Tensor Train)

Oseledets 2011, SIAM J. Sci. Comput. We use ``tensorly``'s
``tensor_train`` decomposition at a fixed maximum bond dimension. The
paper config is ``bond_dim = 9``, which gives ``~68.6x`` compression at
``1024^3``.

We compress each velocity / pressure channel independently and concatenate
the per-channel TT cores into a single payload. Requires ``tensorly``.
