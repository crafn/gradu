# gradu
???

## TODO

- cuda
	- limit dim_block and increase dim_grid automatically
- run with valgrind and fix (some memory bug manifesting occasionally with z2.c generation, rand_data type changes from float to int or intfield5)
- acceleratedness to field type, e.g. "device int field(2) data;"
- optimize matrix & field access by reducing unnecessary index calculations
- overloading test
- test for correct error messages
- error recovery -- don't stop parsing at first error
- handle (preserve) C preprocessor
- change containers to "non-templated" but still type-safe
