#include "backend_c.h"

INTERNAL bool is_builtin_decl(AST_Node *node)
{
	if (node->type != AST_type_decl)
		return false;
	{
		CASTED_NODE(AST_Type_Decl, decl, node);
		return decl->is_builtin;
	}
}

void append_builtin_type_c_str(Array(char) *buf, Builtin_Type bt)
{
	int i;

	if (bt.is_void) {
		append_str(buf, "void", bt.bitness);
	} else if (bt.is_integer) {
		if (bt.is_unsigned)
			append_str(buf, "u");
		if (bt.bitness > 0)
			append_str(buf, "int%i_t", bt.bitness);
		else
			append_str(buf, "int");
	} else if (bt.is_char) {
		append_str(buf, "char");
	} else if (bt.is_float) {
		append_str(buf, "%s", bt.bitness == 64 ? "double" : "float");
	}

	if (bt.is_matrix) {
		append_str(buf, "mat");
		for (i = 0; i < bt.matrix_rank; ++i) {
			append_str(buf, "%i", bt.matrix_dim[i]);
			if (i + 1 < bt.matrix_rank)
				append_str(buf, "x");
		}
	}
}

void append_expr_c_func_name(Array(char) *buf, AST_Node *expr)
{
	Array(AST_Node_Ptr) nodes = create_array(AST_Node_Ptr)(0);
	int i;
	push_array(AST_Node_Ptr)(&nodes, expr);
	push_subnodes(&nodes, expr, true);

	for (i = 0; i < nodes.size; ++i) {
		AST_Node *node = nodes.data[i];

		switch (node->type) {
		case AST_ident: {
			CASTED_NODE(AST_Ident, ident, node);
			if (ident->decl->type == AST_var_decl) {
				CASTED_NODE(AST_Var_Decl, var_decl, ident->decl);
				AST_Type_Decl *type_decl = var_decl->type->base_type_decl;
				if (type_decl->is_builtin) {
					append_builtin_type_c_str(buf, type_decl->builtin_type);	
					append_str(buf, "_");
				} else {
					append_str(buf, "%s_", type_decl->ident->text.data);
				}
			}
		} break;

		case AST_biop: {
			CASTED_NODE(AST_Biop, biop, node);
			switch (biop->type) {
			case Token_add: append_str(buf, "add_"); break;
			case Token_sub: append_str(buf, "sub_"); break;
			case Token_mul: append_str(buf, "mul_"); break;
			case Token_div: append_str(buf, "div_"); break;
			default: append_str(buf, "_unhandled_op_");
			}
		} break;
	
		default:;
		}
	}

	destroy_array(AST_Node_Ptr)(&nodes);
}

INTERNAL void append_type_and_ident_str(Array(char) *buf, AST_Type *type, AST_Ident *ident)
{
	int i;
	if (type->is_const)
		append_str(buf, "const ");
	if (type->base_type_decl->is_builtin) {
		append_builtin_type_c_str(buf, type->base_type_decl->builtin_type);
		append_str(buf, " ");
	} else {
		append_str(buf, "%s ", type->base_type_decl->ident->text.data);
	}
	for (i = 0; i < type->ptr_depth; ++i)
		append_str(buf, "*");
	append_str(buf, "%s", ident->text.data);
	if (type->array_size > 0)
		append_str(buf, "[%i]", type->array_size);
}

INTERNAL AST_Ident *create_ident_for_builtin(Builtin_Type bt)
{
	AST_Ident *ident = create_ident_node();
	append_builtin_type_c_str(&ident->text, bt);
	return ident;
}

INTERNAL AST_Access *create_access_for_var(AST_Var_Decl *var_decl)
{
	AST_Access *access = create_access_node();
	AST_Ident *ident = create_ident_with_text(var_decl->ident->text.data);
	ident->decl = AST_BASE(var_decl);
	access->base = AST_BASE(ident);
	return access;
}

INTERNAL AST_Access *create_access_for_array(AST_Node *expr, int index)
{
	AST_Access *access = create_access_node();
	access->base = expr;
	push_array(AST_Node_Ptr)(&access->args, AST_BASE(create_integer_literal(index)));
	access->is_array_access = true;
	return access;
}

INTERNAL AST_Access *create_access_for_member(AST_Var_Decl *base_decl, AST_Var_Decl *member_decl)
{
	AST_Access *access = create_access_node();
	AST_Ident *base_ident = (AST_Ident*)shallow_copy_ast(AST_BASE(base_decl->ident));
	AST_Ident *member_ident = (AST_Ident*)shallow_copy_ast(AST_BASE(member_decl->ident));

	access->base = AST_BASE(base_ident);
	push_array(AST_Node_Ptr)(&access->args, AST_BASE(member_ident));
	access->is_member_access = true;

	return access;
}

INTERNAL AST_Var_Decl *c_mat_elements_decl(AST_Type_Decl *mat_decl)
{
	AST_Node *m = mat_decl->body->nodes.data[0];
	ASSERT(m->type == AST_var_decl);
	return (AST_Var_Decl*)m;
}

INTERNAL AST_Type_Decl *c_mat_decl(Builtin_Type bt, AST_Scope *root)
{
	return find_builtin_type_decl(bt, root)->builtin_concrete_decl;
}

/* @todo Generalize for n-rank matrices */
INTERNAL AST_Node *create_matrix_mul_expr(AST_Var_Decl *lhs, AST_Var_Decl *rhs, AST_Var_Decl *m_decl, Builtin_Type bt, int i, int j)
{
	AST_Node *expr = NULL;
	int k;
	int dimx = bt.matrix_dim[0];
	int dimy = bt.matrix_dim[1];
	ASSERT(lhs->type->base_type_decl == rhs->type->base_type_decl); /* @todo Handle different matrix types (m_decl for both, different dimensions) */

	/* lhs[0, j] * rhs[i, 0] + ... */
	for (k = 0; k < dimx; ++k) {
		int lhs_index = k + j*dimy;
		int rhs_index = i + k*dimy;
		AST_Access *lhs_m_access = create_access_for_member(lhs, m_decl);
		AST_Access *lhs_arr_access = create_access_for_array(AST_BASE(lhs_m_access), lhs_index);
		AST_Access *rhs_m_access = create_access_for_member(rhs, m_decl);
		AST_Access *rhs_arr_access = create_access_for_array(AST_BASE(rhs_m_access), rhs_index);
		AST_Biop *mul = create_biop_node();

		mul->type = Token_mul;
		mul->lhs = AST_BASE(lhs_arr_access);
		mul->rhs = AST_BASE(rhs_arr_access);

		if (!expr) {
			expr = AST_BASE(mul);
		} else {
			AST_Biop *sum = create_biop_node();
			sum->type = Token_add;
			sum->lhs = expr;
			sum->rhs = AST_BASE(mul);

			expr = AST_BASE(sum);
		}
	}
	ASSERT(expr);
	return expr;
}

typedef struct Trav_Ctx {
	int depth;
	/* Maps nodes from source AST tree to copied/modified AST tree */
	Hash_Table(AST_Node_Ptr, AST_Node_Ptr) src_to_dst;
} Trav_Ctx;

/* Establish mapping */
INTERNAL void map_nodes(Trav_Ctx *ctx, AST_Node *dst, AST_Node *src)
{ set_tbl(AST_Node_Ptr, AST_Node_Ptr)(&ctx->src_to_dst, src, dst); }

/* Retrieve mapping */
INTERNAL AST_Node *mapped_node(Trav_Ctx *ctx, AST_Node *src)
{ return get_tbl(AST_Node_Ptr, AST_Node_Ptr)(&ctx->src_to_dst, src); }


/* Creates copy of (partial) AST, dropping type and func decls */
/* @todo Remove in-place. This is almost identical to copy_ast */
INTERNAL AST_Node * copy_excluding_types_and_funcs(Trav_Ctx *ctx, AST_Node *node)
{
	AST_Node *copy = NULL;
	Array(AST_Node_Ptr) subnodes;
	Array(AST_Node_Ptr) refnodes;
	Array(AST_Node_Ptr) copied_subnodes;
	Array(AST_Node_Ptr) remapped_refnodes;
	int i;

	if (!node)
		return NULL;
	if (ctx->depth > 0 && (node->type == AST_type_decl || node->type == AST_func_decl))
		return NULL;

	++ctx->depth;

	{
		copy = create_ast_node(node->type);
		/* Map nodes before recursing -- dependencies are always to previous nodes */
		map_nodes(ctx, copy, node);

		/* @todo Do something for the massive number of allocations */
		subnodes = create_array(AST_Node_Ptr)(0);
		refnodes = create_array(AST_Node_Ptr)(0);

		push_immediate_subnodes(&subnodes, node);
		push_immediate_refnodes(&refnodes, node);

		copied_subnodes = create_array(AST_Node_Ptr)(subnodes.size);
		remapped_refnodes = create_array(AST_Node_Ptr)(refnodes.size);

		/* Copy subnodes */
		for (i = 0; i < subnodes.size; ++i) {
			AST_Node *copied_sub = copy_excluding_types_and_funcs(ctx, subnodes.data[i]);
			push_array(AST_Node_Ptr)(&copied_subnodes, copied_sub);
		}
		/* Remap referenced nodes */
		for (i = 0; i < refnodes.size; ++i) {
			AST_Node *remapped = mapped_node(ctx, refnodes.data[i]);
			push_array(AST_Node_Ptr)(&remapped_refnodes, remapped);
		}

		/* Fill created node with nodes of the destination tree and settings of the original node */
		copy_ast_node(	copy, node,
						copied_subnodes.data, copied_subnodes.size,
						remapped_refnodes.data, remapped_refnodes.size);

		destroy_array(AST_Node_Ptr)(&copied_subnodes);
		destroy_array(AST_Node_Ptr)(&remapped_refnodes);
		destroy_array(AST_Node_Ptr)(&subnodes);
		destroy_array(AST_Node_Ptr)(&refnodes);
	}

	--ctx->depth;

	return copy;
}

void lift_types_and_funcs_to_global_scope(AST_Scope *root)
{
	Trav_Ctx ctx = {0};
	AST_Scope *dst = create_ast();
	int i, k;

	/* @todo Size should be something like TOTAL_NODE_COUNT*2 */
	ctx.src_to_dst = create_tbl(AST_Node_Ptr, AST_Node_Ptr)(NULL, NULL, 1024);

	for (i = 0; i < root->nodes.size; ++i) {
		AST_Node *sub = root->nodes.data[i];
		Array(AST_Node_Ptr) decls = create_array(AST_Node_Ptr)(0);
		find_subnodes_of_type(&decls, AST_type_decl, sub);
		find_subnodes_of_type(&decls, AST_func_decl, sub);

		/* Lifted types and funcs */
		for (k = 0; k < decls.size; ++k) {
			/* @todo Rename the declarations to avoid name clashes */
			AST_Node *dst_decl = copy_excluding_types_and_funcs(&ctx, decls.data[k]);
			if (!dst_decl)
				continue;
			map_nodes(&ctx, dst_decl, decls.data[k]);
			push_array(AST_Node_Ptr)(&dst->nodes, dst_decl);
		}
		destroy_array(AST_Node_Ptr)(&decls);

		{ /* Copy bulk without inner types or funcs */
			AST_Node *copy = copy_excluding_types_and_funcs(&ctx, sub);
			if (copy) {
				map_nodes(&ctx, copy, sub);
				push_array(AST_Node_Ptr)(&dst->nodes, copy);
			}
		}
	}

	destroy_tbl(AST_Node_Ptr, AST_Node_Ptr)(&ctx.src_to_dst);

	move_ast(root, dst);
}

/* Modifies the AST */
void add_builtin_c_decls_to_global_scope(AST_Scope *root, bool func_decls)
{
	int i, k;
	Array(AST_Node_Ptr) generated_decls = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	push_subnodes(&subnodes, AST_BASE(root), false);

	/* Create decls */
	for (i = 0; i < subnodes.size; ++i) {
		/* Matrix type processing */
		if (subnodes.data[i]->type == AST_type_decl) {
			CASTED_NODE(AST_Type_Decl, decl, subnodes.data[i]);
			int elem_count = 1;
			AST_Type_Decl *mat_decl = NULL; /* Created C matrix type decl */
			AST_Var_Decl *member_decl = NULL; /* Member array decl of matrix */
			Builtin_Type bt;

			if (!decl->is_builtin)
				continue;

			bt = decl->builtin_type;
			if (!bt.is_matrix)
				continue;

			for (k = 0; k < bt.matrix_rank; ++k)
				elem_count *= bt.matrix_dim[k];

	 		{ /* Create matrix type decl */
				mat_decl = create_type_decl_node();
				mat_decl->ident = create_ident_for_builtin(bt);
				mat_decl->body = create_scope_node();
				decl->builtin_concrete_decl = mat_decl;

				{ /* struct member */
					AST_Type_Decl *m_type_decl = create_type_decl_node(); /* @todo Use existing type decl */

					/* Copy base type from matrix type for the struct member */
					m_type_decl->is_builtin = true;
					m_type_decl->builtin_type = bt;
					m_type_decl->builtin_type.is_matrix = false;

					member_decl = create_simple_var_decl(m_type_decl, "m");
					member_decl->type->array_size = elem_count;
					push_array(AST_Node_Ptr)(&mat_decl->body->nodes, AST_BASE(member_decl));

					ASSERT(m_type_decl);
					push_array(AST_Node_Ptr)(&generated_decls, AST_BASE(m_type_decl));
				}

				ASSERT(mat_decl);
				push_array(AST_Node_Ptr)(&generated_decls, AST_BASE(mat_decl));
			}

			if (func_decls) { /* Create matrix multiplication func */
				AST_Func_Decl *mul_decl = create_func_decl_node();
				AST_Var_Decl *lhs_decl = NULL;
				AST_Var_Decl *rhs_decl = NULL;
				mul_decl->ident = create_ident_for_builtin(bt);
				append_str(&mul_decl->ident->text, "_mul");

				mul_decl->return_type = create_type_node();
				mul_decl->return_type->base_type_decl = mat_decl;

				{ /* Params */
					lhs_decl = create_simple_var_decl(mat_decl, "lhs");
					rhs_decl = create_simple_var_decl(mat_decl, "rhs");

					push_array(AST_Var_Decl_Ptr)(&mul_decl->params, lhs_decl);
					push_array(AST_Var_Decl_Ptr)(&mul_decl->params, rhs_decl);
				}

				{ /* Body */
					int x, y;
					AST_Var_Decl *ret_decl = create_simple_var_decl(mat_decl, "ret");
					AST_Control *return_stmt = create_control_node();

					mul_decl->body = create_scope_node();
					push_array(AST_Node_Ptr)(&mul_decl->body->nodes, AST_BASE(ret_decl));

					ASSERT(bt.matrix_rank == 2); /* @todo General algo */
					/* Expression for matrix multiplication */
					for (x = 0; x < bt.matrix_dim[0]; ++x) {
						for (y = 0; y < bt.matrix_dim[1]; ++y) {
							int index = x + y*bt.matrix_dim[1];

							AST_Biop *assign = create_biop_node();
							AST_Access *member_access = create_access_for_member(ret_decl, member_decl);
							AST_Access *array_access = create_access_for_array(AST_BASE(member_access), index);

							assign->type = Token_assign;
							assign->lhs = AST_BASE(array_access);
							assign->rhs = create_matrix_mul_expr(lhs_decl, rhs_decl, member_decl, bt, x, y);
							push_array(AST_Node_Ptr)(&mul_decl->body->nodes, AST_BASE(assign));
						}
					}

					return_stmt->type = Token_kw_return;
					return_stmt->value = AST_BASE(create_access_for_var(ret_decl));
					push_array(AST_Node_Ptr)(&mul_decl->body->nodes, AST_BASE(return_stmt));
				}

				push_array(AST_Node_Ptr)(&generated_decls, AST_BASE(mul_decl));
			}
		}
	}

	{ /* Add C-compatible matrices and operations on top of the source */
		int place = 0;
		while (place < root->nodes.size && is_builtin_decl(root->nodes.data[place]))
			++place;
		insert_array(AST_Node_Ptr)(&root->nodes, place, generated_decls.data, generated_decls.size);
	}

	destroy_array(AST_Node_Ptr)(&subnodes);
	destroy_array(AST_Node_Ptr)(&generated_decls);
}

void apply_c_operator_overloading(AST_Scope *root, bool convert_mat_expr)
{
	int i;
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) replace_list_old = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) replace_list_new = create_array(AST_Node_Ptr)(0);
	push_subnodes(&subnodes, AST_BASE(root), false);

	for (i = 0; i < subnodes.size; ++i) {
		/* Handle matrix "operator overloading" */
		if (convert_mat_expr && subnodes.data[i]->type == AST_biop) {
			CASTED_NODE(AST_Biop, biop, subnodes.data[i]);
			AST_Type type;
			Builtin_Type bt;

			if (biop->type == Token_assign)
				continue;
			if (!expr_type(&type, AST_BASE(biop)))
				continue;
			if (!type.base_type_decl->is_builtin)
				continue;
			bt = type.base_type_decl->builtin_type;
			if (!bt.is_matrix)
				continue;

			{
				AST_Call *call = create_call_node();

				/* @todo Link ident to matrix type */
				call->ident = create_ident_node();
				append_builtin_type_c_str(&call->ident->text, bt);
				/* @todo Handle all operations */
				append_str(&call->ident->text, "_mul");

				{ /* Args */
					push_array(AST_Node_Ptr)(&call->args, biop->lhs);
					push_array(AST_Node_Ptr)(&call->args, biop->rhs);
				}

				/* Mark biop to be replaced with the function call */
				push_array(AST_Node_Ptr)(&replace_list_old, AST_BASE(biop));
				push_array(AST_Node_Ptr)(&replace_list_new, AST_BASE(call));
			}
		}

		/* Convert matrix element accesses to member array accesses */
		if (subnodes.data[i]->type == AST_access) {
			CASTED_NODE(AST_Access, access, subnodes.data[i]);
			AST_Type type;
			Builtin_Type bt;

			if (!access->is_element_access)
				continue;
			if (!expr_type(&type, access->base))
				continue;
			if (!type.base_type_decl->is_builtin)
				continue;
			bt = type.base_type_decl->builtin_type;
			if (!bt.is_matrix)
				continue;

			{ /* Transform from mat(1, 2) -> mat.m[1 + 2*sizex] */
				AST_Type_Decl *mat_decl = c_mat_decl(bt, root);
				AST_Access *member_access = create_access_node();
				AST_Access *array_access = create_access_node();
				AST_Biop *sum = create_biop_node();
				AST_Biop *prod = create_biop_node();

				member_access->base = access->base;
				push_array(AST_Node_Ptr)(&member_access->args, AST_BASE(create_access_for_var(c_mat_elements_decl(mat_decl))));
				member_access->is_member_access = true;

				array_access->base = AST_BASE(member_access);
				push_array(AST_Node_Ptr)(&array_access->args, AST_BASE(sum));
				array_access->is_array_access = true;

				ASSERT(access->args.size == 2); /* @todo Generic matrix rank */
				sum->type = Token_add;
				sum->lhs = access->args.data[0];
				sum->rhs = AST_BASE(prod);

				prod->type = Token_mul;
				prod->lhs = access->args.data[1];
				prod->rhs = AST_BASE(create_integer_literal(bt.matrix_dim[0]));

				push_array(AST_Node_Ptr)(&replace_list_old, AST_BASE(access));
				push_array(AST_Node_Ptr)(&replace_list_new, AST_BASE(array_access));
			}
		}
	}

	{ /* Replace old nodes with new nodes */
		ASSERT(replace_list_new.size == replace_list_old.size);
		replace_nodes_in_ast(AST_BASE(root), replace_list_old.data, replace_list_new.data, replace_list_new.size);

		/* No deep copies of branches */
		for (i = 0; i < replace_list_old.size; ++i)
			shallow_destroy_node(replace_list_old.data[i]);
	}

	destroy_array(AST_Node_Ptr)(&subnodes);
	destroy_array(AST_Node_Ptr)(&replace_list_old);
	destroy_array(AST_Node_Ptr)(&replace_list_new);
}

INTERNAL void append_c_comment(Array(char) *buf, Token *comment)
{
	if (comment->type == Token_line_comment)
		append_str(buf, "/*%.*s */", BUF_STR_ARGS(comment->text));
	else
		append_str(buf, "/*%.*s*/", BUF_STR_ARGS(comment->text));
}

bool ast_to_c_str(Array(char) *buf, int indent, AST_Node *node)
{
	int i, k;
	bool omitted = false;
	int indent_add = 4;

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		int new_indent = indent + indent_add;
		if (scope->is_root)
			new_indent = 0;

		if (!scope->is_root)
			append_str(buf, "{\n");
		for (i = 0; i < scope->nodes.size; ++i) {
			AST_Node *sub = scope->nodes.data[i];
			bool statement_omitted;

			/* Comments are enabled only for scope nodes for now */
			for (k = 0; k < sub->pre_comments.size; ++k) {
				Token *comment = sub->pre_comments.data[k];
				if (comment->empty_line_before)
					append_str(buf, "\n");
				append_str(buf, "%*s", new_indent, "");
				append_c_comment(buf, comment);
				append_str(buf, "\n");
			}

			if (sub->begin_tok && sub->begin_tok->empty_line_before)
				append_str(buf, "\n"); /* Retain some vertical spacing from original code */

			append_str(buf, "%*s", new_indent, "");
			statement_omitted = ast_to_c_str(buf, new_indent, sub);

			if (!statement_omitted &&	sub->type != AST_func_decl &&
										sub->type != AST_scope &&
										sub->type != AST_cond &&
										sub->type != AST_loop)
				append_str(buf, ";");

			if (!statement_omitted && !sub->begin_tok && scope->is_root)
				append_str(buf, "\n"); /* Line break after builtin type decls */

			for (k = 0; k < sub->post_comments.size; ++k) {
				append_str(buf, " ");
				append_c_comment(buf, sub->post_comments.data[k]);
			}

			if (!statement_omitted || sub->post_comments.size > 0)
				append_str(buf, "\n");
		}
		if (!scope->is_root)
			append_str(buf, "%*s}", indent, "");
	} break;

	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, node);
		append_str(buf, "%s", ident->text.data);
	} break;

	case AST_type: {
		FAIL(("Type should be handled in declaration, because type and identifier are mixed in C"));
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		if (decl->is_builtin) {
			omitted = true;
		} else {
			append_str(buf, "typedef struct ");
			append_str(buf, "%s\n", decl->ident->text.data);
			ast_to_c_str(buf, indent, AST_BASE(decl->body));
			append_str(buf, " %s", decl->ident->text.data);
		}
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		append_type_and_ident_str(buf, decl->type, decl->ident);
		if (decl->value) {
			append_str(buf, " = ");
			ast_to_c_str(buf, indent, decl->value);
		}
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		append_type_and_ident_str(buf, decl->return_type, decl->ident);
		append_str(buf, "(");
		for (i = 0; i < decl->params.size; ++i) {
			ast_to_c_str(buf, indent, AST_BASE(decl->params.data[i]));
			if (i + 1 < decl->params.size)
				append_str(buf, ", ");
		}
		if (decl->ellipsis) {
			if (decl->params.size > 0)
				append_str(buf, ", ");
			append_str(buf, "...");
		}
		append_str(buf, ")");
		if (decl->body) {
			append_str(buf, "\n");
			ast_to_c_str(buf, indent, AST_BASE(decl->body));
		} else {
			append_str(buf, ";");
		}
	} break;

	case AST_literal: {
		CASTED_NODE(AST_Literal, literal, node);
		switch (literal->type) {
		case Literal_int:
			append_str(buf, "%i", literal->value.integer);
		break;
		case Literal_string:
			append_str(buf, "\"%.*s\"", literal->value.string.len, literal->value.string.buf);
		break;
		default: FAIL(("Unknown literal type: %i", literal->type));
		}
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, node);
		if (biop->lhs && biop->rhs) {
			ast_to_c_str(buf, indent, biop->lhs);
			append_str(buf, " %s ", tokentype_codestr(biop->type));
			ast_to_c_str(buf, indent, biop->rhs);
		} else {
			/* Unary op */
			append_str(buf, "%s", tokentype_codestr(biop->type));
			ast_to_c_str(buf, indent, biop->rhs);
		}
	} break;

	case AST_control: {
		CASTED_NODE(AST_Control, control, node);
		append_str(buf, "%s", tokentype_codestr(control->type));
		if (control->value) {
			append_str(buf, " ");
			ast_to_c_str(buf, indent, control->value);
		}
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		ast_to_c_str(buf, indent, AST_BASE(call->ident));
		append_str(buf, "(");
		for (i = 0; i < call->args.size; ++i) {
			ast_to_c_str(buf, indent, call->args.data[i]);
			if (i + 1 < call->args.size)
				append_str(buf, ", ");
		}
		append_str(buf, ")");
	} break;

	case AST_access: {
		CASTED_NODE(AST_Access, access, node);
		ast_to_c_str(buf, indent, access->base);
		if (access->is_member_access) {
			append_str(buf, ".");
			ASSERT(access->args.size == 1);
			ast_to_c_str(buf, indent, access->args.data[0]);
		} else if (access->is_array_access) {
			append_str(buf, "[");
			ASSERT(access->args.size == 1);
			ast_to_c_str(buf, indent, access->args.data[0]);
			append_str(buf, "]");
		} else if (access->is_element_access) {
			FAIL(("All element accesses should be transformed to member + array accesses"));
		}
	} break;

	case AST_cond: {
		CASTED_NODE(AST_Cond, cond, node);
		append_str(buf, "if (");
		ast_to_c_str(buf, indent, cond->expr);
		append_str(buf, ") ");
		if (cond->body) {
			ast_to_c_str(buf, indent, AST_BASE(cond->body));
		} else {
			append_str(buf, "\n%*s;", indent + indent_add, "");
		}

		if (cond->after_else) {
			if (cond->body)
				append_str(buf, " ");
			else
				append_str(buf, "\n%*s", indent, "");
			append_str(buf, "else ");
			ast_to_c_str(buf, indent, cond->after_else);
		}
	} break;

	case AST_loop: {
		CASTED_NODE(AST_Loop, loop, node);
		if (loop->init) {
			append_str(buf, "for (");
			ast_to_c_str(buf, indent, loop->init);
			append_str(buf, "; ");
			ast_to_c_str(buf, indent, loop->cond);
			append_str(buf, "; ");
			ast_to_c_str(buf, indent, loop->incr);
			append_str(buf, ") ");
		} else {
			append_str(buf, "while (");
			ast_to_c_str(buf, indent, loop->cond);
			append_str(buf, ") ");
		}

		if (loop->body)
			ast_to_c_str(buf, indent, AST_BASE(loop->body));
		else
			append_str(buf, "\n%*s;", indent + indent_add, "");
	} break;

	default: FAIL(("ast_to_c_str: Unknown node type: %i", node->type));
	}

	return omitted;
}

Array(char) gen_c_code(AST_Scope *root)
{
	Array(char) gen_src = create_array(char)(1024);

	AST_Scope *modified_ast = (AST_Scope*)copy_ast(AST_BASE(root));
	lift_types_and_funcs_to_global_scope(modified_ast);
	add_builtin_c_decls_to_global_scope(modified_ast, true);
	apply_c_operator_overloading(modified_ast, true);

	ast_to_c_str(&gen_src, 0, AST_BASE(modified_ast));
	destroy_ast(modified_ast);
	return gen_src;
}

