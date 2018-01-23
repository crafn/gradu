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
	int i, k, m, n;
	QC_Array(QC_AST_Node_Ptr) replace_list_old = qc_create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) replace_list_new = qc_create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) subnodes = qc_create_array(QC_AST_Node_Ptr)(0);
	QC_AST_Parent_Map map = qc_create_parent_map(QC_AST_BASE(root));

	qc_find_subnodes_of_type(&subnodes, QC_AST_parallel, QC_AST_BASE(root));

	for (i = 0; i < subnodes.size; ++i) {
		QC_CASTED_NODE(QC_AST_Parallel, parallel, subnodes.data[i]);
		QC_Array(QC_AST_Node_Ptr) unique_var_accesses;
		QC_Array(QC_AST_Node_Ptr) var_accesses;
		QC_Array(QC_AST_Var_Decl_Ptr) kernel_param_decls; /* Corresponds to unique_var_accesses */
		QC_AST_Var_Decl *oddeven_param_decl = NULL;
		QC_Array(int) kernel_param_flags; /* Corresponds to kernel_param_decls */

		unique_var_accesses = qc_create_array(QC_AST_Node_Ptr)(0);
		kernel_param_decls = qc_create_array(QC_AST_Var_Decl_Ptr)(0);
		kernel_param_flags = qc_create_array(int)(0);

		if (parallel->is_oddeven) { /* Extra param for indicating are we executing odd or even */
			oddeven_param_decl = qc_create_simple_var_decl(
				qc_find_builtin_type_decl(qc_int_builtin_type(), root),
				"is_odd"
			);
		}

		{ /* Find vars used in loop */
			/* @note We're relying on that input and output fields are first in the resulting array */
			qc_find_subnodes_of_type(&unique_var_accesses, QC_AST_access, QC_AST_BASE(parallel));

			/* Erase locals */
			for (k = 0; k < unique_var_accesses.size; ++k) {
				QC_AST_Ident *ident = qc_access_ident((QC_AST_Access*)unique_var_accesses.data[k]);
				if (qc_is_subnode(&map, QC_AST_BASE(parallel->body), ident->decl))
					qc_erase_array(QC_AST_Node_Ptr)(&unique_var_accesses, k--, 1);
			}

			/* Erase builtin 'id' var */
			for (k = 0; k < unique_var_accesses.size; ++k) {
				QC_AST_Ident *ident = qc_access_ident((QC_AST_Access*)unique_var_accesses.data[k]);
				if (!strcmp(ident->text.data, "id")) {
					qc_erase_array(QC_AST_Node_Ptr)(&unique_var_accesses, k--, 1);
				}
			}

			/* Add __host__ __device__ to definitions of functions that are used in kernel @todo Do recursively*/
			/* Erase function names */
			for (k = 0; k < unique_var_accesses.size; ++k) {
				QC_AST_Ident *ident = qc_access_ident((QC_AST_Access*)unique_var_accesses.data[k]);
				if (ident->decl->type == QC_AST_func_decl) {
					QC_CASTED_NODE(QC_AST_Func_Decl, decl, ident->decl);
					if (!decl->is_extern)
						QC_AST_BASE(decl)->attribute = "__host__ __device__";

					qc_erase_array(QC_AST_Node_Ptr)(&unique_var_accesses, k--, 1);
				}
			}

			/* Erase const globals (and add __constant__ to make them accessible in kernels)*/
			for (k = 0; k < unique_var_accesses.size; ++k) {
				QC_AST_Ident *ident = qc_access_ident((QC_AST_Access*)unique_var_accesses.data[k]);
				if (ident->decl->type == QC_AST_var_decl) {
					QC_CASTED_NODE(QC_AST_Var_Decl, decl, ident->decl);
					QC_ASSERT(decl->type);
					if (decl->type->is_const && decl->is_global) {
						QC_AST_BASE(decl)->attribute = "__constant__";
						qc_erase_array(QC_AST_Node_Ptr)(&unique_var_accesses, k--, 1);
					}
				}
			}

			var_accesses = qc_copy_array(QC_AST_Node_Ptr)(&unique_var_accesses);
			QC_ASSERT(var_accesses.size > 0);

			/* Erase duplicates */
			/* @todo Remove O(n^2) */
			for (k = 0; k < unique_var_accesses.size; ++k) {
				for (m = k + 1; m < unique_var_accesses.size; ++m) {
					QC_AST_Ident *a = qc_access_ident((QC_AST_Access*)unique_var_accesses.data[k]);
					QC_AST_Ident *b = qc_access_ident((QC_AST_Access*)unique_var_accesses.data[m]);
					if (a->decl == b->decl)
						qc_erase_array(QC_AST_Node_Ptr)(&unique_var_accesses, m--, 1);
				}
			}
		}

		{ /* Create cuda call site */
			QC_AST_Scope *scope = qc_create_scope_node();

			/* @todo malloc_calls, memcpy.. -> pre_kernel and post_kernel */
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

			for (k = 0; k < unique_var_accesses.size; ++k) {
				QC_CASTED_NODE(QC_AST_Access, access, unique_var_accesses.data[k]);
				QC_CASTED_NODE(QC_AST_Ident, var, qc_access_ident(access));
				/*const char *cuda_var_name;*/
				const char *host_var_name = var->text.data;
				QC_AST_Var_Decl *cuda_var_decl; 

				QC_AST_Type type;
				if (!qc_expr_type(&type, QC_AST_BASE(access)))
					QC_FAIL(("qc_expr_type failed"));

				{ /* Cuda kernel argument */
					QC_AST_Node *cuda_arg;

					QC_Bool is_modified_in_kernel = QC_false;
					for (m = 0; m < var_accesses.size; ++m) {
						QC_CASTED_NODE(QC_AST_Access, other_access, var_accesses.data[m]);
						QC_CASTED_NODE(QC_AST_Ident, other_ident, qc_access_ident(other_access));
						if (var->decl != other_ident->decl)
							continue;
						if (other_access->can_modify) {
							is_modified_in_kernel = QC_true;
							break;
						}
					}

					if (!is_modified_in_kernel) {
						/* Arg in the call */
						cuda_arg = qc_create_full_deref(qc_copy_ast(QC_AST_BASE(access))); /* Read-only variables are passed by value */
						qc_push_array(QC_AST_Node_Ptr)(&cuda_call->args, cuda_arg);

						/* Param in the kernel */
						cuda_var_decl = qc_create_simple_var_decl(type.base_type_decl, host_var_name);
						qc_push_array(QC_AST_Var_Decl_Ptr)(&kernel_param_decls, cuda_var_decl);
						qc_push_array(int)(&kernel_param_flags, 0);
					} else {
						/* Modified variables require uploading copy to device memory, passing by pointer, and downloading afterwards */
						/* @todo cuda var decl, upload and download calls */
						/*       push upload line to malloc calls, download line to free_calls!!! */
						QC_AST_Ident *mirror_ident = qc_create_ident_with_text(NULL, "cuda_%s", var->text.data);
						QC_AST_Var_Decl *mirror_decl;
						QC_AST_Call *mirror_free;

						/* Type *cuda_var = (Type*)cuda_upload_var(&host_var, sizeof(host_var)); */
						mirror_decl = qc_create_ptr_decl(((QC_AST_Var_Decl*)var->decl)->type->base_type_decl, mirror_ident, NULL);
						mirror_decl->value =
							QC_AST_BASE(qc_create_cast(
								(QC_AST_Type*)qc_copy_ast(QC_AST_BASE(mirror_decl->type)),
								QC_AST_BASE(qc_create_call_2(
									qc_create_ident_with_text(NULL, "cuda_upload_var"),
									QC_AST_BASE(qc_create_addrof(qc_copy_ast(QC_AST_BASE(access)))),
									QC_AST_BASE(qc_create_sizeof(qc_copy_ast(QC_AST_BASE(access))))
								))
							));
						qc_push_array(QC_AST_Node_Ptr)(&malloc_calls, QC_AST_BASE(mirror_decl));

						cuda_arg = qc_try_create_access(qc_copy_ast(QC_AST_BASE(mirror_decl->ident)));
						qc_push_array(QC_AST_Node_Ptr)(&cuda_call->args, cuda_arg);

						/* cuda_download_var(cuda_var, &host_var, sizeof(host_var)); */
						mirror_free = qc_create_call_3(
							qc_create_ident_with_text(NULL, "cuda_download_var"),
							qc_copy_ast(qc_try_create_access(qc_copy_ast(QC_AST_BASE(mirror_decl->ident)))),
							QC_AST_BASE(qc_create_addrof(qc_copy_ast(QC_AST_BASE(access)))),
							QC_AST_BASE(qc_create_sizeof(qc_copy_ast(QC_AST_BASE(access))))
						);
						qc_push_array(QC_AST_Node_Ptr)(&free_calls, QC_AST_BASE(mirror_free));

						/* Param in the kernel */
						cuda_var_decl = (QC_AST_Var_Decl*)qc_copy_ast(QC_AST_BASE(mirror_decl));
						qc_destroy_node(cuda_var_decl->value);
						cuda_var_decl->value = NULL;
						qc_push_array(QC_AST_Var_Decl_Ptr)(&kernel_param_decls, cuda_var_decl);
						qc_push_array(int)(&kernel_param_flags, 1);
					}
				}
			}

			{ /* Write generated nodes to scope in correct order */
				QC_Array(QC_AST_Node_Ptr) size_accesses = qc_create_array(QC_AST_Node_Ptr)(0);

				for (k = 0; k < malloc_calls.size; ++k)
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, malloc_calls.data[k]);
				for (k = 0; k < memcpy_to_device_calls.size; ++k)
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, memcpy_to_device_calls.data[k]);

				/* @todo Proper AST */
				qc_push_array(QC_AST_Node_Ptr)(
							&scope->nodes,
							QC_AST_BASE(qc_create_ident_with_text(NULL, "dim3 dim_grid(100, 1, 1)"))); /* @todo Dynamic value based on size of field */

				for (k = 0; k < parallel->dim; ++k) {
					qc_push_array(QC_AST_Node_Ptr)(&size_accesses, QC_AST_BASE(c_create_field_dim_size((QC_AST_Access*)qc_copy_ast(unique_var_accesses.data[0]), k)));
				}
				qc_push_array(QC_AST_Node_Ptr)(
							&scope->nodes,
							QC_AST_BASE(qc_create_call_3(
								qc_create_ident_with_text(NULL, "dim3 dim_block"),
								QC_AST_BASE(qc_create_biop(QC_Token_div,
									qc_create_chained_expr(size_accesses.data, size_accesses.size, QC_Token_mul),
									QC_AST_BASE(qc_create_integer_literal(100, root)) /* @todo Dynamic value based on size of field */
								)),
								QC_AST_BASE(qc_create_integer_literal(1, root)), 
								QC_AST_BASE(qc_create_integer_literal(1, root))
							)));

				if (!parallel->is_oddeven) {
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, QC_AST_BASE(cuda_call));
				} else {
					/* Two calls with extra oddeven argument */
					QC_AST_Literal *is_odd = qc_create_integer_literal(1, root);
					qc_push_array(QC_AST_Node_Ptr)(&cuda_call->args, QC_AST_BASE(is_odd));
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, QC_AST_BASE(cuda_call));
					qc_push_array(QC_AST_Node_Ptr)(&scope->nodes, qc_copy_ast(QC_AST_BASE(cuda_call)));
					is_odd->value.integer = 0; /* Now first call has param value 0 and latter 1 */
				}

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
			for (k = 0; k < kernel_param_decls.size; ++k) {
				qc_push_array(QC_AST_Var_Decl_Ptr)(&kernel_decl->params, kernel_param_decls.data[k]);
			}
			if (oddeven_param_decl)
				qc_push_array(QC_AST_Var_Decl_Ptr)(&kernel_decl->params, oddeven_param_decl);

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
					QC_AST_Node *ix = QC_AST_BASE(qc_create_ident_with_text(NULL, "(threadIdx.x + blockIdx.x*blockDim.x)"));
					QC_AST_Node *mul_expr;
					QC_AST_Node *mul_expr2;
					QC_AST_Node *expr;

					{ /* .. % size1*size2 .. */
						QC_Array(QC_AST_Node_Ptr) sizes = qc_create_array(QC_AST_Node_Ptr)(0);
						for (n = 0; n <= k; n++)
							qc_push_array(QC_AST_Node_Ptr)(&sizes, QC_AST_BASE(c_create_field_dim_size(qc_create_simple_access(kernel_param_decls.data[0]), n)));
						mul_expr = qc_create_chained_expr(sizes.data, sizes.size, QC_Token_mul);
						qc_destroy_array(QC_AST_Node_Ptr)(&sizes);
					}

					{ /* .. / size1*size2 */
						QC_Array(QC_AST_Node_Ptr) sizes = qc_create_array(QC_AST_Node_Ptr)(0);
						for (n = 0; n < k; n++)
							qc_push_array(QC_AST_Node_Ptr)(&sizes, QC_AST_BASE(c_create_field_dim_size(qc_create_simple_access(kernel_param_decls.data[0]), n)));
						if (sizes.size > 0)
							mul_expr2 = qc_create_chained_expr(sizes.data, sizes.size, QC_Token_mul);
						else
							mul_expr2 = QC_AST_BASE(qc_create_integer_literal(1, root));
						qc_destroy_array(QC_AST_Node_Ptr)(&sizes);
					}

					expr = QC_AST_BASE(qc_create_biop(QC_Token_div, QC_AST_BASE(qc_create_biop(QC_Token_mod, ix, mul_expr)), mul_expr2));

					qc_add_parallel_id_init(root, parallel, k, expr);
				}

				if (oddeven_param_decl) { /* Add an early return for those sites that should not be updated this turn */
					QC_AST_Cond *cond = qc_create_if_1(
						QC_AST_BASE(qc_create_biop(QC_Token_equals,
							QC_AST_BASE(qc_create_biop(QC_Token_mod,
								QC_AST_BASE(qc_create_ident_with_text(NULL, "(threadIdx.x + blockIdx.x*blockDim.x)")),
								QC_AST_BASE(qc_create_integer_literal(2, root))
							)),
							qc_try_create_access(qc_copy_ast(QC_AST_BASE(oddeven_param_decl->ident)))
						)),
						QC_AST_BASE(qc_create_return(NULL))
					);
					qc_insert_array(QC_AST_Node_Ptr)(&parallel->body->nodes, 0, (QC_AST_Node**)&cond, 1);
				}

				/* Substitute host var usage in parallel loop with new kernel params */
				QC_ASSERT(unique_var_accesses.size == kernel_param_decls.size);
				for (k = 0; k < var_accesses.size; ++k) {
					QC_CASTED_NODE(QC_AST_Access, var_access, var_accesses.data[k]);
					for (n = 0; n < unique_var_accesses.size; ++n) {
						QC_CASTED_NODE(QC_AST_Access, unique_access, unique_var_accesses.data[n]);
						QC_CASTED_NODE(QC_AST_Ident, var_ident, var_access->base);
						QC_CASTED_NODE(QC_AST_Ident, unique_ident, unique_access->base);
						if (!unique_access->is_var_access || !var_access->is_var_access)
							continue;

						if (var_ident->decl == unique_ident->decl) {
							QC_AST_Var_Decl *kernel_param_decl = kernel_param_decls.data[n];
							QC_ASSERT(var_ident->b.type == QC_AST_ident);
							/*var_ident->decl = QC_AST_BASE(kernel_param_decl);*/ /* Identifiers in parallel loop are re-resolved so this is unnecessary */
							var_ident->text.size = 0;
							qc_append_str(&var_ident->text, "%s", kernel_param_decl->ident->text.data);

							if (kernel_param_flags.data[n] == 1) { /* Variable is not read-only! */
								QC_AST_Node *parent = qc_find_parent_node(&map, QC_AST_BASE(var_access));

								QC_Bool handled = QC_false;
								if (parent->type == QC_AST_biop) {
									QC_CASTED_NODE(QC_AST_Biop, biop, parent);
									QC_AST_Call *call = NULL;

									/* Make modify atomic */
									if (biop->type == QC_Token_add_assign) {
										call = qc_create_call_2(qc_create_ident_with_text(NULL, "atomicAdd"), QC_AST_BASE(var_access), biop->rhs);
									} else if (biop->type == QC_Token_sub_assign) {
										call = qc_create_call_2(qc_create_ident_with_text(NULL, "atomicAdd"), QC_AST_BASE(var_access), QC_AST_BASE(qc_create_negation(biop->rhs)));
									}

									if (call) {
										QC_AST_Node *parent_of_biop = qc_find_parent_node(&map, parent);
										qc_replace_nodes_in_ast(parent_of_biop, (QC_AST_Node**)&biop, (QC_AST_Node**)&call, 1, 1);
										qc_shallow_destroy_node(QC_AST_BASE(biop));
										handled = QC_true;
									}
								}
								if (!handled) {
									/* Just add a dereference to make code compile */
									QC_AST_Biop *deref = qc_create_deref(QC_AST_BASE(var_access));
									qc_replace_nodes_in_ast(parent, (QC_AST_Node**)&var_access, (QC_AST_Node**)&deref, 1, 1);
								}
							}
						}
					}
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

		qc_destroy_array(QC_AST_Node_Ptr)(&unique_var_accesses);
		qc_destroy_array(QC_AST_Node_Ptr)(&var_accesses);
		qc_destroy_array(QC_AST_Var_Decl_Ptr)(&kernel_param_decls);
		qc_destroy_array(int)(&kernel_param_flags);
	}

	{ /* Replace old nodes with new nodes */
		QC_ASSERT(replace_list_new.size == replace_list_old.size);
		qc_replace_nodes_in_ast(QC_AST_BASE(root), replace_list_old.data, replace_list_new.data, replace_list_new.size, 0);

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
	qc_append_str(&gen_src, 
		"void *cuda_upload_var(void *host_var, int size)\n"
		"{\n"
		"	void *cuda_var;\n"
		"	cudaMalloc(&cuda_var, 4);\n"
		"	cudaMemcpy(cuda_var, host_var, size, cudaMemcpyHostToDevice);\n"
		"	return cuda_var;\n"
		"}\n"
		"void cuda_download_var(void *cuda_var, void *host_var, int size)\n"
		"{\n"
		"	cudaMemcpy(host_var, cuda_var, size, cudaMemcpyDeviceToHost);\n"
		"	cudaFree(cuda_var);\n"
		"}\n\n"); /* @todo Insert to AST */

	qc_ast_to_c_str(&gen_src, 0, QC_AST_BASE(modified_ast));

	qc_destroy_ast(modified_ast);
	return gen_src;
}
