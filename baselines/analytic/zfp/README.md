# ZFP

Lindstrom 2014, IEEE TVCG. Streaming block-floating-point compression
with deterministic rate ``r`` bits per voxel. We use ``zfpy`` to call
the C library; ``rate = 32 / ratio`` reaches the target compression
``ratio`` (e.g. ``rate = 0.5`` for ``64x``).

The byte-representation packs the four per-channel ZFP streams into a
single ``numpy.npz`` archive. Requires ``zfpy``.
