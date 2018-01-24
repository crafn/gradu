# gradu
???

## TODO

- cuda
	- limit dim_block and increase dim_grid automatically
- acceleratedness to field type, e.g. "device int field(2) data;"
- optimize matrix & field access by reducing unnecessary index calculations
- overloading test
- test for correct error messages
- error recovery -- don't stop parsing at first error
- handle (preserve) C preprocessor
- change containers to "non-templated" but still type-safe
- fix intentional memory leak in qc_lift_types_and_funcs_to_global_scope