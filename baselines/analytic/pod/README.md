# POD (Proper Orthogonal Decomposition)

Lumley 1967. Fits an orthonormal basis ``U`` of shape ``(K, 4 * D * H * W)``
on the training split via randomised SVD. Encoding: ``z = U^T x`` is
``K`` floats. Decoding: ``x_hat = U z``.

The paper uses ``K = 2048`` for the ``64x`` tier comparison; the
per-frame ratio is reported alongside the tier label in Table 1.

## Fitting the basis

```bash
python -m baselines.analytic.pod.compute_basis \
    --data_dir data/jhtdb/ \
    --stats   data/jhtdb/stats.npz \
    --K 2048 \
    --out checkpoints/pod_basis_K2048.npy
```

The output file is large (the ``K * 4 * D^3`` matrix takes ~32 GB at
``D = 256``, ~8 TB at ``D = 1024``). For the open-source release we
recommend fitting at ``D = 256`` (after the stride-4 down-sample); the
``1024^3`` reconstruction can be done patch-wise with the same basis.
