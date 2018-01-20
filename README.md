# gradu
???

## TODO

- z2 cuda simulation
	- limit dim_block and increase dim_grid
	- modified cuda vars to pointers that are cudaMalloced and transferred <=> original
	- accumulation
	- even/odd
	- random
- QC_AST_NODE -> QC_B
- reverse field memory layout: should be same as multidim C arrays
- .qft extension to .lc, .fc, .qc or something (lattice C, field C)
- simulation.qft -> diffusion.qft
- command line options format: -option=value -> -opt value
- full support for ordinary arrays (declaring etc.)
- acceleratedness to field type, e.g. "device int field(2) data;"
- #define support
- #include support
- change containers to "non-templated" but still type-safe
