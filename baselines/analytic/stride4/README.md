# Stride-4 + trilinear up-sample

The "null" compressor: average every ``4^3`` block of voxels (the
stride-4 ``avg_pool3d``) and store the down-sampled field. Reconstruct
by trilinear interpolation. The ratio is exactly ``64x`` (``4^3``).

This is the floor any learned method must beat to justify the cost of
training. Its physics signature is unambiguous: every wavenumber above
``N / 8 = k = 128`` is zeroed out, so the inertial-range slope and the
dissipation rate both suffer.

There is no checkpoint to ship; the decoder is parameter-free.
