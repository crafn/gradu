#include "backend_cuda.h"

Array(char) gen_cuda_code(AST_Scope *root)
{
	Array(char) arr = create_array(char)(0);
	append_str(&arr, "@todo CUDA generation\n");
	memset(root, 0, 0);
	return arr;
}
