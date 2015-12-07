#include "backend_cuda.h"
#include "backend_c.h"

void parallel_loops_to_cuda(AST_Scope *root)
{
	int i, k, m;
	Array(AST_Node_Ptr) replace_list_old = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) replace_list_new = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	AST_Parent_Map map = create_parent_map(AST_BASE(root));

	find_subnodes_of_type(&subnodes, AST_parallel, AST_BASE(root));

	for (i = 0; i < subnodes.size; ++i) {
		CASTED_NODE(AST_Parallel, parallel, subnodes.data[i]);
		Array(AST_Node_Ptr) var_accesses, cuda_var_decls;

		var_accesses = create_array(AST_Node_Ptr)(0);
		cuda_var_decls = create_array(AST_Node_Ptr)(0);

		{ /* Find vars used in loop */
			find_subnodes_of_type(&var_accesses, AST_access, parallel->output);
			find_subnodes_of_type(&var_accesses, AST_access, parallel->input);
			find_subnodes_of_type(&var_accesses, AST_access, AST_BASE(parallel->body));

			/* Erase locals */
			for (k = 0; k < var_accesses.size; ++k) {
				AST_Ident *ident = access_ident((AST_Access*)var_accesses.data[k]);
				if (is_subnode(&map, AST_BASE(parallel->body), ident->decl))
					erase_array(AST_Node_Ptr)(&var_accesses, k--, 1);
			}

			/* Erase builtin 'id' var */
			for (k = 0; k < var_accesses.size; ++k) {
				AST_Ident *ident = access_ident((AST_Access*)var_accesses.data[k]);
				if (!strcmp(ident->text.data, "id")) {
					erase_array(AST_Node_Ptr)(&var_accesses, k--, 1);
				}
			}

			/* Erase duplicates */
			/* @todo Remove O(n^2) */
			for (k = 0; k < var_accesses.size; ++k) {
				for (m = k + 1; m < var_accesses.size; ++m) {
					AST_Ident *a = access_ident((AST_Access*)var_accesses.data[k]);
					AST_Ident *b = access_ident((AST_Access*)var_accesses.data[m]);
					if (a->decl == b->decl)
						erase_array(AST_Node_Ptr)(&var_accesses, m--, 1);
				}
			}
		}

		{ /* Create cuda call site */
			AST_Scope *scope = create_scope_node();

			Array(AST_Node_Ptr) malloc_calls = create_array(AST_Node_Ptr)(0);
			Array(AST_Node_Ptr) memcpy_to_device_calls = create_array(AST_Node_Ptr)(0);
			Array(AST_Node_Ptr) memcpy_from_device_calls = create_array(AST_Node_Ptr)(0);
			Array(AST_Node_Ptr) free_calls = create_array(AST_Node_Ptr)(0);
			AST_Call *cuda_call = create_call_node();

			/* @todo Link ident to the kernel decl */
			cuda_call->ident = create_ident_with_text(NULL, "TODO_proper_kernel_name");
			/*append_expr_c_func_name(&cuda_call->ident->text, biop->rhs);*/
			append_str(&cuda_call->ident->text, "<<<dim_grid, dim_block>>>");

			/* Copy comments */
			copy_ast_node_base(AST_BASE(scope), AST_BASE(parallel));

			for (k = 0; k < var_accesses.size; ++k) {
				CASTED_NODE(AST_Access, access, var_accesses.data[k]);
				CASTED_NODE(AST_Ident, var, access_ident(access));
				/* const char *cuda_var_name; */
				const char *host_var_name = var->text.data;
				AST_Var_Decl *cuda_var_decl; 

				AST_Type type;
				if (!expr_type(&type, AST_BASE(access)))
					FAIL(("expr_type failed"));

				/*
				ASSERT(access->args.size == 0);
				ASSERT(access->base->type == AST_ident);
				*/

				{ /* Cuda var declaration */
					cuda_var_decl = create_simple_var_decl(type.base_type_decl, host_var_name);
					push_array(AST_Node_Ptr)(&cuda_var_decls, AST_BASE(cuda_var_decl));
				}
#if 0
					cuda_var_name = cuda_var_decl->ident->text.data;

					{ /* Device malloc */
						AST_Call *call = create_call_node();
						call->ident = create_ident_with_text(NULL, "cudaMalloc");

						{ /* Args */
							/* @todo Proper AST */
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "(void**)&%s", cuda_var_name)));
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "sizeof(*%s)", cuda_var_name)));
						}
						push_array(AST_Node_Ptr)(&malloc_calls, AST_BASE(call));
					}

					{ /* Cuda kernel argument */
						push_array(AST_Node_Ptr)(&cuda_call->args,
								AST_BASE(create_ident_with_text(NULL, cuda_var_name)));
					}

					/* Memcpy from host to device */
					if (k > 0) {
						AST_Call *call = create_call_node();
						call->ident = create_ident_with_text(NULL, "cudaMemcpy");

						{ /* Args */
							/* @todo Proper AST */
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "%s", cuda_var_name)));
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "%s", host_var_name)));
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "sizeof(*%s)", cuda_var_name)));
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "cudaMemcpyHostToDevice")));
						}
						push_array(AST_Node_Ptr)(&memcpy_to_device_calls, AST_BASE(call));
					}

					/* Memcpy from device to host */
					if (k == 0) {
						AST_Call *call = create_call_node();
						call->ident = create_ident_with_text(NULL, "cudaMemcpy");

						{ /* Args */
							/* @todo Proper AST */
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "%s", host_var_name)));
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "%s", cuda_var_name)));
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "sizeof(*%s)", cuda_var_name)));
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "cudaMemcpyDeviceToHost")));
						}
						push_array(AST_Node_Ptr)(&memcpy_from_device_calls, AST_BASE(call));
					}

					{ /* Free device memory */
						AST_Call *call = create_call_node();
						call->ident = create_ident_with_text(NULL, "cudaFree");

						{ /* Args */
							/* @todo Proper AST */
							push_array(AST_Node_Ptr)(&call->args,
									AST_BASE(create_ident_with_text(NULL, "%s", cuda_var_name)));
						}
						push_array(AST_Node_Ptr)(&free_calls, AST_BASE(call));
					}
#endif
			}

			{ /* Write generated nodes to scope in correct order */
				for (k = 0; k < cuda_var_decls.size; ++k)
					push_array(AST_Node_Ptr)(&scope->nodes, cuda_var_decls.data[k]);
				for (k = 0; k < malloc_calls.size; ++k)
					push_array(AST_Node_Ptr)(&scope->nodes, malloc_calls.data[k]);
				for (k = 0; k < memcpy_to_device_calls.size; ++k)
					push_array(AST_Node_Ptr)(&scope->nodes, memcpy_to_device_calls.data[k]);
#if 0
				ASSERT(bt.matrix_rank == 2); /* @todo Other ranks */
				/* @todo Proper AST */
				push_array(AST_Node_Ptr)(
							&scope->nodes,
							AST_BASE(create_ident_with_text("dim3 dim_grid(1, 1, 1)")));
				push_array(AST_Node_Ptr)(
							&scope->nodes,
							AST_BASE(create_ident_with_text("dim3 dim_block(%i, %i, 1)",
															bt.matrix_dim[0], bt.matrix_dim[1])));
#endif

				push_array(AST_Node_Ptr)(&scope->nodes, AST_BASE(cuda_call));

				for (k = 0; k < memcpy_from_device_calls.size; ++k)
					push_array(AST_Node_Ptr)(&scope->nodes, memcpy_from_device_calls.data[k]);
				for (k = 0; k < free_calls.size; ++k)
					push_array(AST_Node_Ptr)(&scope->nodes, free_calls.data[k]);
			}

			destroy_array(AST_Node_Ptr)(&malloc_calls);
			destroy_array(AST_Node_Ptr)(&memcpy_to_device_calls);
			destroy_array(AST_Node_Ptr)(&memcpy_from_device_calls);
			destroy_array(AST_Node_Ptr)(&free_calls);

			/* Mark parallel loop to be replaced with the scope node */
			push_array(AST_Node_Ptr)(&replace_list_old, AST_BASE(parallel));
			push_array(AST_Node_Ptr)(&replace_list_new, AST_BASE(scope));
		}

		{ /* Create cuda kernel */
			/* @todo __global__ attribute to function decl */
			AST_Func_Decl *kernel_decl = create_func_decl_node();
			kernel_decl->ident = create_ident_with_text(AST_BASE(kernel_decl), "TODO_proper_kernel_name");

			kernel_decl->return_type = create_type_node();
			kernel_decl->return_type->base_type_decl = find_builtin_type_decl(void_builtin_type(), root);
			ASSERT(kernel_decl->return_type->base_type_decl);

			/* Params */
			for (k = 0; k < cuda_var_decls.size; ++k) {
				push_array(AST_Var_Decl_Ptr)(&kernel_decl->params, (AST_Var_Decl*)copy_ast(cuda_var_decls.data[k]));
			}

			{ /* Body */
				/* Fields are passed by value */
				Array(AST_Node_Ptr) accesses = create_array(AST_Node_Ptr)(0);
				find_subnodes_of_type(&accesses, AST_access, AST_BASE(parallel->body));
				for (k = 0; k < accesses.size; ++k) {
					CASTED_NODE(AST_Access, access, accesses.data[k]);
					AST_Type type;
					if (!access->is_element_access)
						continue;
					if (!expr_type(&type, access->base))
						continue;
					if (!type.base_type_decl->is_builtin)
						continue;
					if (!type.base_type_decl->builtin_type.is_field)
						continue;
					access->implicit_deref = false;
				}
				destroy_array(AST_Node_Ptr)(&accesses);

				/* Add initialization for 'id' */
				for (k = 0; k < parallel->dim; ++k) {
					const char *comp_name[3] = { "x", "y", "z" };
					ASSERT(k < 3);
					add_parallel_id_init(root, parallel, k,
							try_create_access(
								AST_BASE(create_ident_with_text(NULL, "threadIdx.%s", comp_name[k]))));
				}

				/* Move the whole block from parallel loop -- re-resolve afterwards */
				unresolve_ast(AST_BASE(parallel->body));
				kernel_decl->body = parallel->body;
				parallel->body = NULL;
			}

			/* Insert kernel before current function call */
			{
				AST_Func_Decl *func = find_enclosing_func(&map, AST_BASE(parallel));
				AST_Node *parent_node = find_parent_node(&map, AST_BASE(func));
				AST_Scope *parent_scope;
				int ix;
				ASSERT(parent_node->type == AST_scope);
				parent_scope = (AST_Scope*)parent_node;
				ix = find_in_scope(parent_scope, AST_BASE(func));
				ASSERT(ix >= 0);

				insert_array(AST_Node_Ptr)(&parent_scope->nodes, ix, (AST_Node**)&kernel_decl, 1);
			}
		}

		destroy_array(AST_Node_Ptr)(&var_accesses);
		destroy_array(AST_Node_Ptr)(&cuda_var_decls);
	}

	{ /* Replace old nodes with new nodes */
		ASSERT(replace_list_new.size == replace_list_old.size);
		replace_nodes_in_ast(AST_BASE(root), replace_list_old.data, replace_list_new.data, replace_list_new.size);

		/* Subnodes should be deep-copied */
		for (i = 0; i < replace_list_old.size; ++i)
			destroy_node(replace_list_old.data[i]);
	}

	resolve_ast(root);

	destroy_parent_map(&map);
	destroy_array(AST_Node_Ptr)(&subnodes);
	destroy_array(AST_Node_Ptr)(&replace_list_old);
	destroy_array(AST_Node_Ptr)(&replace_list_new);
}

Array(char) gen_cuda_code(AST_Scope *root)
{
	Array(char) gen_src = create_array(char)(0);
	AST_Scope *modified_ast = (AST_Scope*)copy_ast(AST_BASE(root));
	lift_types_and_funcs_to_global_scope(modified_ast);
	/* @todo There's something wrong with lift_types_and_funcs_to_global_scope. This can't be before it without some resolving issues. */
	parallel_loops_to_cuda(modified_ast);
	add_builtin_c_decls_to_global_scope(modified_ast, true);
	apply_c_operator_overloading(modified_ast, true);

	append_c_stdlib_includes(&gen_src);
	ast_to_c_str(&gen_src, 0, AST_BASE(modified_ast));

	destroy_ast(modified_ast);
	return gen_src;
}
