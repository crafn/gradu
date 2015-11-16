#include "backend_cuda.h"
#include "backend_c.h"

Array(char) gen_cuda_code(AST_Scope *root)
{
	Array(char) gen_src = create_array(char)(0);
	AST_Scope *modified_ast = (AST_Scope*)copy_ast_tree(AST_BASE(root));

	lift_types_and_funcs_to_global_scope(modified_ast);
	add_builtin_c_decls_to_global_scope(modified_ast, false);
	ast_to_c_str(&gen_src, 0, AST_BASE(modified_ast));

	destroy_ast_tree(modified_ast);
	return gen_src;
}
