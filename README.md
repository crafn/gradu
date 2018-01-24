# gradu
???

## TODO

- z2 cuda simulation
	- limit dim_block and increase dim_grid automatically
- reverse field memory layout: should be same as multidim C arrays
- .qft extension to .lc, .fc, .qc or something (lattice C, field C)
- simulation.qft -> diffusion.qft
- command line options format: -option=value -> -opt value
- full support for ordinary arrays (declaring etc.)
- acceleratedness to field type, e.g. "device int field(2) data;"
- #define support
- #include support
- change containers to "non-templated" but still type-safe
- optimize matrix & field access by reducing unnecessary index calculations