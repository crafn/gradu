#include "backend_cuda.h"
#include "backend_c.h"

/* device field malloc/free and field copying */
void add_builtin_cuda_funcs(QC_AST_Scope *root)
{
	int i, k, m;

	for (i = 0; i < root->nodes.size; ++i) {
		QC_AST_Node *generated = NULL;

		if (root->nodes.data[i]->type == QC_AST_func_decl) {
			/* Create concrete field alloc and dealloc funcs */

			QC_CASTED_NODE(QC_AST_Func_Decl, func_decl, root->nodes.data[i]);
			if (!func_decl->is_builtin)
				continue;

			if (!strcmp(func_decl->ident->text.data, "alloc_device_field")) {
				QC_AST_Func_Decl *alloc_func = (QC_AST_Func_Decl*)qc_copy_ast(QC_AST_BASE(func_decl));
				QC_AST_Type_Decl *field_decl = alloc_func->return_type->base_type_decl;
				QC_Builtin_Type bt = field_decl->builtin_type;
				field_decl = field_decl->builtin_concrete_decl;
				QC_ASSERT(field_decl);

				QC_ASSERT(!alloc_func->body);

				alloc_func->is_builtin = QC_false;
				func_decl->builtin_concrete_decl = alloc_func;
				qc_append_str(&alloc_func->ident->text, "_");
				qc_append_builtin_type_c_str(&alloc_func->ident->text,
						alloc_func->return_type->base_type_decl->builtin_type);

				{ /* Function contents */
					QC_AST_Var_Decl *field_var_decl;
					QC_AST_Biop *sizeof_expr;
					QC_Array(QC_AST_Node_Ptr) size_accesses = qc_create_array(QC_AST_Node_Ptr)(alloc_func->params.size);
					QC_AST_Call *elements_assign;
					QC_AST_Biop *is_device_field_assign;
					QC_AST_Control *ret_stmt;

					alloc_func->body = qc_create_scope_node();

					field_var_decl = qc_create_simple_var_decl(field_decl, "field");
					qc_push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(field_var_decl));

					sizeof_expr =
						qc_create_sizeof(
							QC_AST_BASE(qc_create_deref(
								QC_AST_BASE(qc_create_simple_member_access(
									field_var_decl,
									c_mat_elements_decl(field_decl)
								))
							))
						);
					qc_push_array(QC_AST_Node_Ptr)(&size_accesses, QC_AST_BASE(sizeof_expr));
					for (k = 0; k < alloc_func->params.size; ++k) {
						qc_push_array(QC_AST_Node_Ptr)(&size_accesses,
							QC_AST_BASE(qc_create_access_for_var(alloc_func->params.data[k])));
					}

					elements_assign = qc_create_call_2(
						qc_create_ident_with_text(NULL, "cudaMalloc"),
						QC_AST_BASE(qc_create_cast( /* For c++ */
							qc_create_builtin_type(qc_void_builtin_type(), 2, root),
							QC_AST_BASE(qc_create_addrof(
								QC_AST_BASE(qc_create_simple_member_access(
									field_var_decl,
									c_mat_elements_decl(field_decl)
								))
							))
						)),
						qc_create_chained_expr(size_accesses.data, size_accesses.size, QC_Token_mul)
					);

					qc_push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(elements_assign));
					qc_destroy_array(QC_AST_Node_Ptr)(&size_accesses);

					for (k = 0; k < bt.field_dim; ++k) {
						QC_AST_Biop *assign =
							qc_create_assign(
								QC_AST_BASE(qc_create_member_array_access(
									qc_copy_ast(QC_AST_BASE(field_var_decl->ident)), /* @todo Access var */
									c_field_size_decl(field_decl),
									QC_AST_BASE(qc_create_integer_literal(k, NULL)),
									QC_false
								)),
								QC_AST_BASE(qc_create_access_for_var(
									alloc_func->params.data[k]
								))
							);
						qc_push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(assign));
					}

					is_device_field_assign =
						qc_create_assign(
							QC_AST_BASE(qc_create_simple_member_access(
								field_var_decl,
								qc_is_device_field_member_decl(field_decl)
							)),
							QC_AST_BASE(qc_create_integer_literal(1, root))
						);
					qc_push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(is_device_field_assign));

					ret_stmt =
						qc_create_return(QC_AST_BASE(
							qc_create_access_for_var(field_var_decl)
						));
					qc_push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(ret_stmt));
				}

				generated = QC_AST_BASE(alloc_func);
			} else if (!strcmp(func_decl->ident->text.data, "free_device_field")) {
				QC_AST_Func_Decl *free_func = (QC_AST_Func_Decl*)qc_copy_ast(QC_AST_BASE(func_decl));
				QC_AST_Type_Decl *field_decl = free_func->params.data[0]->type->base_type_decl;
				QC_Builtin_Type bt = field_decl->builtin_type;
				field_decl = field_decl->builtin_concrete_decl;

				QC_ASSERT(!free_func->body);
				QC_ASSERT(free_func->params.size == 1);

				free_func->is_builtin = QC_false;
				func_decl->builtin_concrete_decl = free_func;
				qc_append_str(&free_func->ident->text, "_");
				qc_append_builtin_type_c_str(&free_func->ident->text, bt);

				{ /* Function contents */
					QC_AST_Access *access_field_data = qc_create_member_access(
							qc_copy_ast(QC_AST_BASE(free_func->params.data[0]->ident)),
							c_mat_elements_decl(field_decl));
					QC_AST_Call *libc_free_call = qc_create_call_1(
							qc_create_ident_with_text(NULL, "cudaFree"),
							QC_AST_BASE(access_field_data));

					free_func->body = qc_create_scope_node();
					qc_push_array(QC_AST_Node_Ptr)(&free_func->body->nodes, QC_AST_BASE(libc_free_call));
				}

				generated = QC_AST_BASE(free_func);
			} else if (!strcmp(func_decl->ident->text.data, "memcpy_field")) {
				QC_AST_Func_Decl *memcpy_func = (QC_AST_Func_Decl*)qc_copy_ast(QC_AST_BASE(func_decl));
				QC_AST_Var_Decl *dst_field_var_decl = memcpy_func->params.data[0];
				QC_AST_Var_Decl *src_field_var_decl = memcpy_func->params.data[1];
				QC_AST_Type_Decl *field_decl = dst_field_var_decl->type->base_type_decl;
				QC_Builtin_Type bt = field_decl->builtin_type;
				field_decl = field_decl->builtin_concrete_decl;

				QC_ASSERT(!memcpy_func->body);
				QC_ASSERT(memcpy_func->params.size == 2);

				memcpy_func->is_builtin = QC_false;
				func_decl->builtin_concrete_decl = memcpy_func;
				qc_append_str(&memcpy_func->ident->text, "_");
				qc_append_builtin_type_c_str(&memcpy_func->ident->text, bt);
				
				memcpy_func->body = qc_create_scope_node();

				{ /* Function contents */
					/* @todo Generate matching size assert */
					for (k = 0; k < 4; ++k) {
						QC_AST_Access *is_dst_device, *is_src_device;
						QC_AST_Access *access_dst_field, *access_src_field;
						QC_AST_Cond *cond;
						int dst_host = k/2;
						int src_host = k%2;
						const char *cuda_memcpy_enums[4] = {
							"cudaMemcpyHostToHost",
							"cudaMemcpyHostToDevice",
							"cudaMemcpyDeviceToHost",
							"cudaMemcpyDeviceToDevice"
						};
						QC_Array(QC_AST_Node_Ptr) size_accesses = qc_create_array(QC_AST_Node_Ptr)(3);
						qc_push_array(QC_AST_Node_Ptr)(&size_accesses,
								QC_AST_BASE(c_create_mat_element_sizeof(dst_field_var_decl)));
						for (m = 0; m < bt.field_dim; ++m) {
							qc_push_array(QC_AST_Node_Ptr)(&size_accesses, 
								QC_AST_BASE(c_create_field_dim_size(qc_create_simple_access(dst_field_var_decl), m)));
						}

						access_dst_field =	qc_create_simple_member_access(
												memcpy_func->params.data[0],
												c_mat_elements_decl(field_decl));
						access_src_field =	qc_create_simple_member_access(
												memcpy_func->params.data[1],
												c_mat_elements_decl(field_decl));
						is_dst_device = qc_create_simple_member_access(
											dst_field_var_decl,
											qc_is_device_field_member_decl(field_decl)
										);
						is_src_device = qc_create_simple_member_access(
											src_field_var_decl,
											qc_is_device_field_member_decl(field_decl)
										);

						cond =	qc_create_if_1(
								QC_AST_BASE(qc_create_and(
									QC_AST_BASE(qc_create_equals(	QC_AST_BASE(is_dst_device),
															QC_AST_BASE(qc_create_integer_literal(src_host, root)))),
									QC_AST_BASE(qc_create_equals(	QC_AST_BASE(is_src_device),
															QC_AST_BASE(qc_create_integer_literal(dst_host, root))))
								)),
								QC_AST_BASE(qc_create_call_4(
									qc_create_ident_with_text(NULL, "cudaMemcpy"),
									QC_AST_BASE(access_dst_field),
									QC_AST_BASE(access_src_field),
									qc_create_chained_expr(size_accesses.data, size_accesses.size, QC_Token_mul),
									QC_AST_BASE(qc_create_ident_with_text(NULL, cuda_memcpy_enums[k]))
								))
								);

						qc_push_array(QC_AST_Node_Ptr)(&memcpy_func->body->nodes, QC_AST_BASE(cond));
						qc_destroy_array(QC_AST_Node_Ptr)(&size_accesses);
					}
				}

				generated = QC_AST_BASE(memcpy_func);
			}
		}

		if (generated) {
			/* Insert after current node */
			qc_insert_array(QC_AST_Node_Ptr)(&root->nodes, i + 1, &generated, 1);
			++i;
		}
	}
}

void parallel_loops_to_cuda(QC_AST_Scope *root)
{
	int kernel_count = 0;
	int i, k, m;
	QC_Array(QC_AST_Node_Ptr) replace_list_old = qc_create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) replace_list_new = qc_create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) subnodes = qc_create_array(QC_AST_Node_Ptr)(0);
	QC_AST_Parent_Map map = qc_create_parent_map(QC_AST_BASE(root));

	qc_find_subnodes_of_type(&subnodes, QC_AST_parallel, QC_AST_BASE(root));

	for (i = 0; i < subnodes.size; ++i) {
		QC_CASTED_NODE(QC_AST_Parallel, parallel, subnodes.data[i]);
		QC_Array(QC_AST_Node_Ptr) var_accesses;
		QC_Array(QC_AST_Var_Decl_Ptr) cuda_var_decls;

		var_accesses = qc_create_array(QC_AST_Node_Ptr)(0);
		cuda_var_decls = qc_create_array(QC_AST_Var_Decl_Ptr)(0);

		{ /* Find vars used in loop */
			/* @note We're relying on that input and output fields are first in the resulting array */
			qc_find_subnodes_of_type(&var_accesses, QC_AST_access, QC_AST_BASE(parallel));

			/* Erase locals */
			for (k = 0; k < var_accesses.size; ++k) {
				QC_AST_Ident *ident = qc_access_ident((QC_AST_Access*)var_accesses.data[k]);
				if (qc_is_subnode(&map, QC_AST_BASE(parallel->body), ident->decl))
					qc_erase_array(QC_AST_Node_Ptr)(&var_accesses, k--, 1);
			}

			/* Erase builtin 'id' var */
			for (k = 0; k < var_accesses.size; ++k) {
				QC_AST_Ident *ident = qc_access_ident((QC_AST_Access*)var_accesses.data[k]);
				if (!strcmp(ident->text.data, "id")) {
					qc_erase_array(QC_AST_Node_Ptr)(&var_accesses, k--, 1);
				}
			}

			/* Add __host__ __device__ to definitions of functions that are used in kernel */
			/* Erase function names */
			for (k = 0; k < var_accesses.size; ++k) {
				QC_AST_Ident *ident = qc_access_ident((QC_AST_Access*)var_accesses.data[k]);
				if (ident->decl->type == QC_AST_func_decl) {
					QC_AST_Func_Decl *decl = (QC_AST_Func_Decl*)ident->decl;
					if (!decl->is_extern)
						QC_AST_BASE(decl)->attribute = "__host__ __device__";

					qc_erase_array(QC_AST_Node_Ptr)(&var_accesses, k--, 1);
				}
			}

			/* Erase duplicates */
			/* @todo Remove O(n^2) */
			for (k = 0; k < var_accesses.size; ++k) {
				for (m = k + 1; m < var_accesses.size; ++m) {
					QC_AST_Ident *a = qc_access_ident((QC_AST_Access*)var_accesses.data[k]);
					QC_AST_Ident *b = qc_access_ident((QC_AST_Access*)var_accesses.data[m]);
					if (a->decl == b->decl)
						qc_erase_array(QC_AST_Node_Ptr)(&var_accesses, m--, 1);
				}
			}
		}

		{ /* Create cuda call site */
			QC_AST_Scope *scope = qc_create_scope_node();

			QC_Array(QC_AST_Node_Ptr) malloc_calls = qc_create_array(QC_AST_Node_Ptr)(0);
			QC_Array(QC_AST_Node_Ptr) memcpy_to_device_calls = qc_create_array(QC_AST_Node_Ptr)(0);
			QC_Array(QC_AST_Node_Ptr) memcpy_from_device_calls = qc_create_array(QC_AST_Node_Ptr)(0);
			QC_Array(QC_AST_Node_Ptr) free_calls = qc_create_array(QC_AST_Node_Ptr)(0);
			QC_AST_Call *cuda_call = qc_create_call_node();
			QC_AST_Ident *cuda_call_ident = qc_create_ident_with_text(NULL, "kernel_%i", kernel_count);

			/* @todo Link ident to the kernel decl */
			cuda_call->base = qc_try_create_access(QC_AST_BASE(cuda_call_ident));
			/*qc_append_expr_c_func_name(&cuda_call->ident->text, biop->rhs);*/
			qc_append_str(&cuda_call_ident->text, "<<<dim_grid, dim_block>>>");

			/* Copy comments */
			qc_copy_ast_node_base(QC_AST_BASE(scope), QC_AST_BASE(parallel));

			for (k = 0; k < var_accesses.size; ++k) {
				QC_CASTED_NODE(QC_AST_Access, access, var_accesses.data[k]);
				QC_CASTED_NODE(QC_AST_Ident, var, qc_access_ident(access));
				/*const char *cuda_var_name;*/
				const char *host_var_name = var->text.data;
				QC_AST_Var_Decl *cuda_var_decl; 

				QC_AST_Type type;
				if (!qc_expr_type(&type, QC_AST_BASE(access)))
					QC_FAIL(("qc_expr_type failed"));

				{ /* Cuda var declaration */
					cuda_var_decl = qc_create_simple_var_decl(type.base_type_decl, host_var_name);
					/*qc_append_str(&cuda_var_decl->ident->text, "_cuda");*/
					qc_push_array(QC_AST_Var_Decl_Ptr)(&cuda_var_decls, cuda_var_decl);
				}

				/*cuda_var_name = cuda_var_decl->ident->text.data;*/

				{ /* Cuda kernel argument */
					QC_AST_Node *deref_cuda_var = qc_create_full_deref(qc_copy_ast(QC_AST_BASE(access)));
					qc_push_array(QC_AST_Node_Ptr)(&cuda_call->args, deref_cuda_var);
				}
			}

			{ /* Write generated nodes to scope in correct order */
				QC_Array(QC_AST_Node_Ptr) size_accesses = qc_create_array(QC_AST_Node_Ptr)(0);

				for (k = 0; k < malloc_calls.size; ++k)
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, malloc_calls.data[k]);
				for (k = 0; k < memcpy_to_device_calls.size; ++k)
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, memcpy_to_device_calls.data[k]);

				/* @todo Proper AST */
				/* @todo Support non-two-dimensional fields! */
				qc_push_array(QC_AST_Node_Ptr)(
							&scope->nodes,
							QC_AST_BASE(qc_create_ident_with_text(NULL, "dim3 dim_grid(1, 1, 1)")));


				for (k = 0; k < parallel->dim; ++k) {
					qc_push_array(QC_AST_Node_Ptr)(&size_accesses, QC_AST_BASE(c_create_field_dim_size((QC_AST_Access*)qc_copy_ast(var_accesses.data[0]), k)));
				}

				qc_push_array(QC_AST_Node_Ptr)(
							&scope->nodes,
							QC_AST_BASE(qc_create_call_3(
								qc_create_ident_with_text(NULL, "dim3 dim_block"),
								qc_create_chained_expr(size_accesses.data, size_accesses.size, QC_Token_mul),
								QC_AST_BASE(qc_create_integer_literal(1, root)),
								QC_AST_BASE(qc_create_integer_literal(1, root))
							)));

				qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, QC_AST_BASE(cuda_call));

				for (k = 0; k < memcpy_from_device_calls.size; ++k)
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, memcpy_from_device_calls.data[k]);
				for (k = 0; k < free_calls.size; ++k)
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, free_calls.data[k]);

				qc_destroy_array(QC_AST_Node_Ptr)(&size_accesses);
			}

			qc_destroy_array(QC_AST_Node_Ptr)(&malloc_calls);
			qc_destroy_array(QC_AST_Node_Ptr)(&memcpy_to_device_calls);
			qc_destroy_array(QC_AST_Node_Ptr)(&memcpy_from_device_calls);
			qc_destroy_array(QC_AST_Node_Ptr)(&free_calls);

			/* Mark parallel loop to be replaced with the scope node */
			qc_push_array(QC_AST_Node_Ptr)(&replace_list_old, QC_AST_BASE(parallel));
			qc_push_array(QC_AST_Node_Ptr)(&replace_list_new, QC_AST_BASE(scope));
		}

		{ /* Create cuda kernel */
			QC_AST_Func_Decl *kernel_decl = qc_create_func_decl_node();
			kernel_decl->b.attribute = "__global__";
			kernel_decl->ident = qc_create_ident_with_text(QC_AST_BASE(kernel_decl), "kernel_%i", kernel_count++);

			kernel_decl->return_type = qc_create_type_node();
			kernel_decl->return_type->base_type_decl = qc_find_builtin_type_decl(qc_void_builtin_type(), root);
			QC_ASSERT(kernel_decl->return_type->base_type_decl);

			/* Params */
			for (k = 0; k < cuda_var_decls.size; ++k) {
				qc_push_array(QC_AST_Var_Decl_Ptr)(&kernel_decl->params, cuda_var_decls.data[k]);
			}

			{ /* Body */
				{ /* Fields are passed by value */
					QC_Array(QC_AST_Node_Ptr) accesses = qc_create_array(QC_AST_Node_Ptr)(0);
					qc_find_subnodes_of_type(&accesses, QC_AST_access, QC_AST_BASE(parallel->body));
					for (k = 0; k < accesses.size; ++k) {
						QC_CASTED_NODE(QC_AST_Access, access, accesses.data[k]);
						QC_AST_Type type;
						if (!access->is_element_access)
							continue;
						if (!qc_expr_type(&type, access->base))
							continue;
						if (!type.base_type_decl->is_builtin)
							continue;
						if (!type.base_type_decl->builtin_type.is_field)
							continue;
						access->implicit_deref = QC_false;
					}
					qc_destroy_array(QC_AST_Node_Ptr)(&accesses);
				}

				/* Add initialization for 'id' */
				for (k = parallel->dim - 1; k >= 0; --k) {
					QC_AST_Node *ix = QC_AST_BASE(qc_create_ident_with_text(NULL, "threadIdx.x"));
					QC_AST_Node *mul_expr;
					QC_AST_Node *mul_expr2;
					QC_AST_Node *expr;
					int n;

					{ /* .. % size1*size2 .. */
						QC_Array(QC_AST_Node_Ptr) sizes = qc_create_array(QC_AST_Node_Ptr)(0);
						for (n = 0; n <= k; n++)
							qc_push_array(QC_AST_Node_Ptr)(&sizes, QC_AST_BASE(c_create_field_dim_size(qc_create_simple_access(cuda_var_decls.data[0]), n)));
						mul_expr = qc_create_chained_expr(sizes.data, sizes.size, QC_Token_mul);
						qc_destroy_array(QC_AST_Node_Ptr)(&sizes);
					}

					{ /* .. / size1*size2 */
						QC_Array(QC_AST_Node_Ptr) sizes = qc_create_array(QC_AST_Node_Ptr)(0);
						for (n = 0; n < k; n++)
							qc_push_array(QC_AST_Node_Ptr)(&sizes, QC_AST_BASE(c_create_field_dim_size(qc_create_simple_access(cuda_var_decls.data[0]), n)));
						if (sizes.size > 0)
							mul_expr2 = qc_create_chained_expr(sizes.data, sizes.size, QC_Token_mul);
						else
							mul_expr2 = QC_AST_BASE(qc_create_integer_literal(1, root));
						qc_destroy_array(QC_AST_Node_Ptr)(&sizes);
					}

					expr = QC_AST_BASE(qc_create_biop(QC_Token_div, QC_AST_BASE(qc_create_biop(QC_Token_mod, ix, mul_expr)), mul_expr2));

					qc_add_parallel_id_init(root, parallel, k, expr);
				}

				/* Move the whole block from parallel loop -- re-resolve afterwards */
				qc_unresolve_ast(QC_AST_BASE(parallel->body));
				kernel_decl->body = parallel->body;
				parallel->body = NULL;
			}

			/* Insert kernel before current function call */
			{
				QC_AST_Func_Decl *func = qc_find_enclosing_func(&map, QC_AST_BASE(parallel));
				QC_AST_Node *parent_node = qc_find_parent_node(&map, QC_AST_BASE(func));
				QC_AST_Scope *parent_scope;
				int ix;
				QC_ASSERT(parent_node->type == QC_AST_scope);
				parent_scope = (QC_AST_Scope*)parent_node;
				ix = qc_find_in_scope(parent_scope, QC_AST_BASE(func));
				QC_ASSERT(ix >= 0);

				qc_insert_array(QC_AST_Node_Ptr)(&parent_scope->nodes, ix, (QC_AST_Node**)&kernel_decl, 1);
			}
		}

		qc_destroy_array(QC_AST_Node_Ptr)(&var_accesses);
		qc_destroy_array(QC_AST_Var_Decl_Ptr)(&cuda_var_decls);
	}

	{ /* Replace old nodes with new nodes */
		QC_ASSERT(replace_list_new.size == replace_list_old.size);
		qc_replace_nodes_in_ast(QC_AST_BASE(root), replace_list_old.data, replace_list_new.data, replace_list_new.size);

		/* Subnodes should be deep-copied */
		for (i = 0; i < replace_list_old.size; ++i)
			qc_destroy_node(replace_list_old.data[i]);
	}

	qc_resolve_ast(root);

	qc_destroy_parent_map(&map);
	qc_destroy_array(QC_AST_Node_Ptr)(&subnodes);
	qc_destroy_array(QC_AST_Node_Ptr)(&replace_list_old);
	qc_destroy_array(QC_AST_Node_Ptr)(&replace_list_new);
}

QC_Array(char) qc_gen_cuda_code(QC_AST_Scope *root)
{
	QC_Array(char) gen_src = qc_create_array(char)(0);
	QC_AST_Scope *modified_ast = (QC_AST_Scope*)qc_copy_ast(QC_AST_BASE(root));
	qc_lift_types_and_funcs_to_global_scope(modified_ast);
	qc_add_builtin_c_decls_to_global_scope(modified_ast, QC_false);
	add_builtin_cuda_funcs(modified_ast);
	/* @todo There's something wrong with qc_lift_types_and_funcs_to_global_scope. This can't be before it without some resolving issues. */
	parallel_loops_to_cuda(modified_ast);
	qc_apply_c_operator_overloading(modified_ast, QC_true);

	qc_append_c_stdlib_includes(&gen_src);
	qc_ast_to_c_str(&gen_src, 0, QC_AST_BASE(modified_ast));

	qc_destroy_ast(modified_ast);
	return gen_src;
}
