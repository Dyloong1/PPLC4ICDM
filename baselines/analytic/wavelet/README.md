# Wavelet (db4 level 3) baseline

Daubechies 1988. Per-channel 3D ``wavedecn`` with the ``db4`` mother
wavelet at ``level = 3``. After the decomposition we magnitude-sort all
coefficients across all sub-bands and channels, keep the top ``K`` so
the kept-count matches the target compression ratio, and inverse-transform
to get the reconstruction.

For the ``64x`` tier we keep ``4 * D^3 / 64`` coefficients (~67 M for
``D = 1024``). The byte-representation stores the kept ``(value, index)``
pairs plus a small shape map needed to rebuild the nested ``wavedecn``
coefficient tree.

Requires `PyWavelets` (``pywt``).
