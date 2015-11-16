#include "backend_cuda.h"
#include "backend_c.h"

INTERNAL void apply_cuda_operator_overloading(AST_Scope *root)
{
	int i, k;
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) replace_list_old = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) replace_list_new = create_array(AST_Node_Ptr)(0);
	push_subnodes(&subnodes, AST_BASE(root), false);

	for (i = 0; i < subnodes.size; ++i) {
		/* Handle matrix "operator overloading" */
		if (subnodes.data[i]->type == AST_biop) {
			CASTED_NODE(AST_Biop, biop, subnodes.data[i]);
			AST_Type type;
			Builtin_Type bt;

			if (!expr_type(&type, AST_BASE(biop)))
				continue;

			if (!biop->is_top_level)
				continue;
			if (!type.base_type_decl->is_builtin)
				continue;

			bt = type.base_type_decl->builtin_type;
			if (!bt.is_matrix)
				continue;

			{ /* Create cuda call */
				AST_Scope *scope = create_scope_node();
				Array(AST_Node_Ptr) var_accesses = create_array(AST_Node_Ptr)(0);

				Array(AST_Node_Ptr) var_decls = create_array(AST_Node_Ptr)(0);
				Array(AST_Node_Ptr) malloc_calls = create_array(AST_Node_Ptr)(0);
				Array(AST_Node_Ptr) memcpy_to_device_calls = create_array(AST_Node_Ptr)(0);
				Array(AST_Node_Ptr) memcpy_from_device_calls = create_array(AST_Node_Ptr)(0);
				Array(AST_Node_Ptr) free_calls = create_array(AST_Node_Ptr)(0);
				AST_Call *cuda_call = create_call_node();

				/* @todo Link ident to the kernel decl */
				cuda_call->ident = create_ident_node();
				append_expr_c_func_name(&cuda_call->ident->text, biop->rhs);
				append_str(&cuda_call->ident->text, "<<<dim_grid, dim_block>>>");

				/* Copy comments */
				copy_ast_node_base(AST_BASE(scope), AST_BASE(biop));

				/* Find vars used in expr */
				find_subnodes_of_type(&var_accesses, AST_access, AST_BASE(biop));

				for (k = 0; k < var_accesses.size; ++k) {
					CASTED_NODE(AST_Access, access, var_accesses.data[k]);
					ASSERT(!access->sub);
					ASSERT(access->base->type == AST_ident);
					{
						CASTED_NODE(AST_Ident, var, access->base);
						const char *cuda_var_name;
						const char *host_var_name = var->text.data;
						AST_Var_Decl *cuda_var_decl; 

						{ /* Cuda var declaration */
							cuda_var_decl = create_simple_var_decl(type.base_type_decl, host_var_name);
							append_str(&cuda_var_decl->ident->text, "_cuda");
							++cuda_var_decl->type->ptr_depth;
							push_array(AST_Node_Ptr)(&var_decls, AST_BASE(cuda_var_decl));
						}
						cuda_var_name = cuda_var_decl->ident->text.data;

						{ /* Device malloc */
							AST_Call *call = create_call_node();
							call->ident = create_ident_with_text("cudaMalloc");

							{ /* Args */
								/* @todo Proper AST */
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("(void**)&%s", cuda_var_name)));
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("sizeof(*%s)", cuda_var_name)));
							}
							push_array(AST_Node_Ptr)(&malloc_calls, AST_BASE(call));
						}

						{ /* Cuda kernel argument */
							push_array(AST_Node_Ptr)(&cuda_call->args,
									AST_BASE(create_ident_with_text(cuda_var_name)));
						}

						/* Memcpy from host to device */
						if (k > 0) {
							AST_Call *call = create_call_node();
							call->ident = create_ident_with_text("cudaMemcpy");

							{ /* Args */
								/* @todo Proper AST */
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("%s", cuda_var_name)));
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("%s", host_var_name)));
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("sizeof(*%s)", cuda_var_name)));
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("cudaMemcpyHostToDevice")));
							}
							push_array(AST_Node_Ptr)(&memcpy_to_device_calls, AST_BASE(call));
						}

						/* Memcpy from device to host */
						if (k == 0) {
							AST_Call *call = create_call_node();
							call->ident = create_ident_with_text("cudaMemcpy");

							{ /* Args */
								/* @todo Proper AST */
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("%s", host_var_name)));
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("%s", cuda_var_name)));
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("sizeof(*%s)", cuda_var_name)));
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("cudaMemcpyDeviceToHost")));
							}
							push_array(AST_Node_Ptr)(&memcpy_from_device_calls, AST_BASE(call));
						}

						{ /* Free device memory */
							AST_Call *call = create_call_node();
							call->ident = create_ident_with_text("cudaFree");

							{ /* Args */
								/* @todo Proper AST */
								push_array(AST_Node_Ptr)(&call->args,
										AST_BASE(create_ident_with_text("%s", cuda_var_name)));
							}
							push_array(AST_Node_Ptr)(&free_calls, AST_BASE(call));
						}
					}
				}

				{ /* Write generated nodes to scope in correct order */
					for (k = 0; k < var_decls.size; ++k)
						push_array(AST_Node_Ptr)(&scope->nodes, var_decls.data[k]);
					for (k = 0; k < malloc_calls.size; ++k)
						push_array(AST_Node_Ptr)(&scope->nodes, malloc_calls.data[k]);
					for (k = 0; k < memcpy_to_device_calls.size; ++k)
						push_array(AST_Node_Ptr)(&scope->nodes, memcpy_to_device_calls.data[k]);

					ASSERT(bt.matrix_rank == 2); /* @todo Other ranks */
					/* @todo Proper AST */
					push_array(AST_Node_Ptr)(
								&scope->nodes,
								AST_BASE(create_ident_with_text("dim3 dim_grid(1, 1, 1)")));
					push_array(AST_Node_Ptr)(
								&scope->nodes,
								AST_BASE(create_ident_with_text("dim3 dim_block(%i, %i, 1)",
																bt.matrix_dim[0], bt.matrix_dim[1])));

					push_array(AST_Node_Ptr)(&scope->nodes, AST_BASE(cuda_call));

					for (k = 0; k < memcpy_from_device_calls.size; ++k)
						push_array(AST_Node_Ptr)(&scope->nodes, memcpy_from_device_calls.data[k]);
					for (k = 0; k < free_calls.size; ++k)
						push_array(AST_Node_Ptr)(&scope->nodes, free_calls.data[k]);
				}

				destroy_array(AST_Node_Ptr)(&var_accesses);

				destroy_array(AST_Node_Ptr)(&var_decls);
				destroy_array(AST_Node_Ptr)(&malloc_calls);
				destroy_array(AST_Node_Ptr)(&memcpy_to_device_calls);
				destroy_array(AST_Node_Ptr)(&memcpy_from_device_calls);
				destroy_array(AST_Node_Ptr)(&free_calls);

				/* Mark biop to be replaced with the scope node */
				push_array(AST_Node_Ptr)(&replace_list_old, AST_BASE(biop));
				push_array(AST_Node_Ptr)(&replace_list_new, AST_BASE(scope));
			}
		}
	}

	{ /* Replace old nodes with new nodes */
		ASSERT(replace_list_new.size == replace_list_old.size);
		replace_nodes_in_ast(AST_BASE(root), replace_list_old.data, replace_list_new.data, replace_list_new.size);

		for (i = 0; i < replace_list_old.size; ++i)
			destroy_node(replace_list_old.data[i]);
	}

	destroy_array(AST_Node_Ptr)(&subnodes);
	destroy_array(AST_Node_Ptr)(&replace_list_old);
	destroy_array(AST_Node_Ptr)(&replace_list_new);
}

Array(char) gen_cuda_code(AST_Scope *root)
{
	Array(char) gen_src = create_array(char)(0);
	AST_Scope *modified_ast = (AST_Scope*)copy_ast(AST_BASE(root));

	lift_types_and_funcs_to_global_scope(modified_ast);
	add_builtin_c_decls_to_global_scope(modified_ast, false);
	apply_cuda_operator_overloading(modified_ast);
	ast_to_c_str(&gen_src, 0, AST_BASE(modified_ast));

	destroy_ast(modified_ast);
	return gen_src;
}
