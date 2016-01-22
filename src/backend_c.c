#include "backend_c.h"

INTERNAL bool nested_expr_needs_parens(QC_AST_Node *expr, QC_AST_Node *nested)
{
	if (expr->type != QC_AST_biop || nested->type != QC_AST_biop)
		return false;
	{
		QC_CASTED_NODE(QC_AST_Biop, op, expr);
		QC_CASTED_NODE(QC_AST_Biop, subop, nested);

		return biop_prec(op->type) > biop_prec(subop->type);
	}
}

#if 0
INTERNAL bool is_builtin_decl(QC_AST_Node *node)
{
	if (node->type == QC_AST_type_decl) {
		QC_CASTED_NODE(QC_AST_Type_Decl, decl, node);
		return decl->is_builtin;
	} else if (node->type == QC_AST_func_decl) {
		QC_CASTED_NODE(QC_AST_Func_Decl, decl, node);
		return decl->is_builtin;
	}

	return false;
}
#endif

void append_builtin_type_c_str(QC_Array(char) *buf, QC_Builtin_Type bt)
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

	if (bt.is_field) {
		append_str(buf, "field%i", bt.field_dim);
	}
}

void append_expr_c_func_name(QC_Array(char) *buf, QC_AST_Node *expr)
{
	QC_Array(QC_AST_Node_Ptr) nodes = create_array(QC_AST_Node_Ptr)(0);
	int i;
	push_array(QC_AST_Node_Ptr)(&nodes, expr);
	push_subnodes(&nodes, expr, true);

	for (i = 0; i < nodes.size; ++i) {
		QC_AST_Node *node = nodes.data[i];

		switch (node->type) {
		case QC_AST_ident: {
			QC_CASTED_NODE(QC_AST_Ident, ident, node);
			if (ident->decl->type == QC_AST_var_decl) {
				QC_CASTED_NODE(QC_AST_Var_Decl, var_decl, ident->decl);
				QC_AST_Type_Decl *type_decl = var_decl->type->base_type_decl;
				if (type_decl->is_builtin) {
					append_builtin_type_c_str(buf, type_decl->builtin_type);	
					append_str(buf, "_");
				} else {
					append_str(buf, "%s_", type_decl->ident->text.data);
				}
			}
		} break;

		case QC_AST_biop: {
			QC_CASTED_NODE(QC_AST_Biop, biop, node);
			switch (biop->type) {
			case QC_Token_add: append_str(buf, "add_"); break;
			case QC_Token_sub: append_str(buf, "sub_"); break;
			case QC_Token_mul: append_str(buf, "mul_"); break;
			case QC_Token_div: append_str(buf, "div_"); break;
			default: append_str(buf, "_unhandled_op_");
			}
		} break;
	
		default:;
		}
	}

	destroy_array(QC_AST_Node_Ptr)(&nodes);
}

INTERNAL void append_type_and_ident_str(QC_Array(char) *buf, QC_AST_Type *type, const char *ident)
{
	int i;
	if (type->is_const)
		append_str(buf, "const ");

	if (type->base_typedef) {
		append_str(buf, "%s ", type->base_typedef->ident->text.data);
	} else {
		if (type->base_type_decl->is_builtin) {
			append_builtin_type_c_str(buf, type->base_type_decl->builtin_type);
		} else {
			append_str(buf, "%s", type->base_type_decl->ident->text.data);
		}

		if (ident)
			append_str(buf, " ");
	}

	for (i = 0; i < type->ptr_depth; ++i)
		append_str(buf, "*");
	if (ident)
		append_str(buf, "%s", ident);
	if (type->array_size > 0)
		append_str(buf, "[%i]", type->array_size);
}

INTERNAL QC_AST_Ident *create_ident_for_builtin(QC_Builtin_Type bt)
{
	QC_AST_Ident *ident = create_ident_node();
	append_builtin_type_c_str(&ident->text, bt);
	return ident;
}

INTERNAL QC_AST_Access *create_access_for_var(QC_AST_Var_Decl *var_decl)
{
	QC_AST_Access *access = create_access_node();
	QC_AST_Ident *ident = create_ident_with_text(QC_AST_BASE(var_decl), var_decl->ident->text.data);
	access->base = QC_AST_BASE(ident);
	return access;
}

INTERNAL QC_AST_Access *create_array_access(QC_AST_Node *expr, QC_AST_Node *index)
{
	QC_AST_Access *access = create_access_node();
	access->base = expr;
	push_array(QC_AST_Node_Ptr)(&access->args, index);
	access->is_array_access = true;
	return access;
}

/* @todo Remove, use create_plain_member_access */
INTERNAL QC_AST_Access *create_member_access(QC_AST_Node *base, QC_AST_Var_Decl *member_decl)
{
	QC_AST_Access *access = create_access_node();
	QC_AST_Ident *member_ident = (QC_AST_Ident*)shallow_copy_ast(QC_AST_BASE(member_decl->ident));

	access->base = base;
	push_array(QC_AST_Node_Ptr)(&access->args, QC_AST_BASE(member_ident));
	access->is_member_access = true;

	return access;
}

/* @todo to ast.h */
INTERNAL QC_AST_Access *create_member_array_access(QC_AST_Node *base, QC_AST_Var_Decl *member_decl, QC_AST_Node *index, bool deref)
{
	QC_AST_Access *member = create_member_access(base, member_decl);
	member->implicit_deref = deref;
	return create_array_access(QC_AST_BASE(member), index);
}

/* Matrix or field */
INTERNAL QC_AST_Var_Decl *c_mat_elements_decl(QC_AST_Type_Decl *mat_decl)
{
	if (mat_decl->builtin_concrete_decl) /* Work with builtin matrix and concrete struct */
		mat_decl = mat_decl->builtin_concrete_decl;

	{
		QC_AST_Node *m = mat_decl->body->nodes.data[0];
		ASSERT(m->type == QC_AST_var_decl);
		return (QC_AST_Var_Decl*)m;
	}
}

INTERNAL QC_AST_Var_Decl *c_field_size_decl(QC_AST_Type_Decl *field_decl)
{
	if (field_decl->builtin_concrete_decl) /* Work with builtin field and concrete struct */
		field_decl = field_decl->builtin_concrete_decl;

	{
		QC_AST_Node *m = field_decl->body->nodes.data[1];
		ASSERT(m->type == QC_AST_var_decl);
		return (QC_AST_Var_Decl*)m;
	}
}

QC_AST_Var_Decl *is_device_field_member_decl(QC_AST_Type_Decl *field_decl)
{
	if (field_decl->builtin_concrete_decl) /* Work with builtin field and concrete struct */
		field_decl = field_decl->builtin_concrete_decl;

	{
		QC_AST_Node *m = field_decl->body->nodes.data[2];
		ASSERT(m->type == QC_AST_var_decl);
		return (QC_AST_Var_Decl*)m;
	}
}

/* field.size[dim_i] */
INTERNAL QC_AST_Access *c_create_field_dim_size(QC_AST_Access *field_access, int dim_i)
{
	QC_AST_Type type;
	if (!expr_type(&type, QC_AST_BASE(field_access)))
		FAIL(("c_create_field_dim_size: expr_type failed"));

	return create_member_array_access(
		create_full_deref(QC_AST_BASE(field_access)),
		c_field_size_decl(type.base_type_decl),
		QC_AST_BASE(create_integer_literal(dim_i, NULL)),
		false
	);
}

/* Matrix or field */
/* sizeof(*mat.m) */
INTERNAL QC_AST_Biop *c_create_mat_element_sizeof(QC_AST_Var_Decl *mat)
{
	return create_sizeof(
		QC_AST_BASE(create_deref(
			QC_AST_BASE(create_member_access(
				copy_ast(QC_AST_BASE(mat->ident)), /* @todo Access var */
				c_mat_elements_decl(mat->type->base_type_decl)
			))
		))
	);
}


INTERNAL QC_AST_Type_Decl *concrete_type_decl(QC_Builtin_Type bt, QC_AST_Scope *root)
{
	QC_AST_Type_Decl *decl = find_builtin_type_decl(bt, root)->builtin_concrete_decl;
	ASSERT(decl);
	return decl;
}

/* @todo Generalize for n-rank matrices */
INTERNAL QC_AST_Node *create_matrix_mul_expr(QC_AST_Var_Decl *lhs, QC_AST_Var_Decl *rhs, QC_AST_Var_Decl *m_decl, QC_Builtin_Type bt, int i, int j)
{
	QC_AST_Node *expr = NULL;
	int k;
	int dimx = bt.matrix_dim[0];
	int dimy = bt.matrix_dim[1];
	ASSERT(lhs->type->base_type_decl == rhs->type->base_type_decl); /* @todo Handle different matrix types (m_decl for both, different dimensions) */

	/* lhs[0, j] * rhs[i, 0] + ... */
	for (k = 0; k < dimx; ++k) {
		int lhs_index = k + j*dimy;
		int rhs_index = i + k*dimy;
		QC_AST_Access *lhs_m_access = create_member_access(shallow_copy_ast(QC_AST_BASE(lhs->ident)), m_decl);
		QC_AST_Access *lhs_arr_access =
			create_array_access(QC_AST_BASE(lhs_m_access), QC_AST_BASE(create_integer_literal(lhs_index, NULL)));
		QC_AST_Access *rhs_m_access = create_member_access(shallow_copy_ast(QC_AST_BASE(rhs->ident)), m_decl);
		QC_AST_Access *rhs_arr_access =
			create_array_access(QC_AST_BASE(rhs_m_access), QC_AST_BASE(create_integer_literal(rhs_index, NULL)));
		QC_AST_Biop *mul = create_biop_node();

		mul->type = QC_Token_mul;
		mul->lhs = QC_AST_BASE(lhs_arr_access);
		mul->rhs = QC_AST_BASE(rhs_arr_access);

		if (!expr) {
			expr = QC_AST_BASE(mul);
		} else {
			QC_AST_Biop *sum = create_biop_node();
			sum->type = QC_Token_add;
			sum->lhs = expr;
			sum->rhs = QC_AST_BASE(mul);

			expr = QC_AST_BASE(sum);
		}
	}
	ASSERT(expr);
	return expr;
}

void parallel_loops_to_ordinary(QC_AST_Scope *root)
{
	int i, k;
	QC_Array(QC_AST_Node_Ptr) replace_list_old = create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) replace_list_new = create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) subnodes = create_array(QC_AST_Node_Ptr)(0);
	find_subnodes_of_type(&subnodes, QC_AST_parallel, QC_AST_BASE(root));

	for (i = 0; i < subnodes.size; ++i) {
		QC_CASTED_NODE(QC_AST_Parallel, parallel, subnodes.data[i]);
		QC_AST_Scope *scope = create_scope_node();
		QC_AST_Loop *outer_loop = NULL;
		QC_AST_Loop *inner_loop = NULL;
		copy_ast_node_base(QC_AST_BASE(scope), QC_AST_BASE(parallel));

		/* Create nested loops */
		for (k = 0; k < parallel->dim; ++k) {
			QC_AST_Var_Decl *index_decl =
				create_var_decl(
						find_builtin_type_decl(int_builtin_type(), root),
						create_ident_with_text(NULL, "id_%i", k),
						QC_AST_BASE(create_integer_literal(0, root)));
			QC_AST_Loop *loop =
				create_for_loop(
					index_decl,
					QC_AST_BASE(create_call_2(
						create_ident_with_text(NULL, "size"),
						copy_ast(parallel->output),
						QC_AST_BASE(create_integer_literal(k, root))
					)),
					NULL
				);

			if (!outer_loop)
				outer_loop = loop;

			if (inner_loop) {
				inner_loop->body = create_scope_node();
				push_array(QC_AST_Node_Ptr)(&inner_loop->body->nodes, QC_AST_BASE(loop));
			}
			inner_loop = loop;
		}

		{ /* Add innermost loop content */
			for (k = 0; k < parallel->dim; ++k) {
				add_parallel_id_init(root, parallel, k,
						try_create_access(
							QC_AST_BASE(create_ident_with_text(NULL, "id_%i", k))));
			}

			inner_loop->body = (QC_AST_Scope*)copy_ast(QC_AST_BASE(parallel->body));
		}

		push_array(QC_AST_Node_Ptr)(&scope->nodes, QC_AST_BASE(outer_loop));

		push_array(QC_AST_Node_Ptr)(&replace_list_old, QC_AST_BASE(parallel));
		push_array(QC_AST_Node_Ptr)(&replace_list_new, QC_AST_BASE(scope));
	}

	{ /* Replace old nodes with new nodes */
		ASSERT(replace_list_new.size == replace_list_old.size);
		replace_nodes_in_ast(QC_AST_BASE(root), replace_list_old.data, replace_list_new.data, replace_list_new.size);

		/* Subnodes should be deep-copied */
		for (i = 0; i < replace_list_old.size; ++i)
			destroy_node(replace_list_old.data[i]);
	}

	/* Resolve calls of 'size' to corresponding functions */
	resolve_ast(root);

	destroy_array(QC_AST_Node_Ptr)(&subnodes);
	destroy_array(QC_AST_Node_Ptr)(&replace_list_old);
	destroy_array(QC_AST_Node_Ptr)(&replace_list_new);
}


void lift_var_decls(QC_AST_Scope *root)
{
	int i;
	QC_Array(QC_AST_Node_Ptr) subnodes = create_array(QC_AST_Node_Ptr)(0);
	QC_AST_Parent_Map map = create_parent_map(QC_AST_BASE(root));

	find_subnodes_of_type(&subnodes, QC_AST_var_decl, QC_AST_BASE(root));

	for (i = 0; i < subnodes.size; ++i) {
		QC_CASTED_NODE(QC_AST_Var_Decl, decl, subnodes.data[i]);
		QC_AST_Node *parent = find_parent_node(&map, QC_AST_BASE(decl));
		ASSERT(parent);

		switch (parent->type) {
		case QC_AST_loop: {
			QC_CASTED_NODE(QC_AST_Loop, loop, parent);
			QC_AST_Node *loop_parent = find_parent_node(&map, QC_AST_BASE(loop));
			ASSERT(loop_parent);
			ASSERT(loop_parent->type == QC_AST_scope);
			{
				QC_CASTED_NODE(QC_AST_Scope, scope, loop_parent);
				ASSERT(decl->value);

				/* Do variable initialization (not decl) in the for loop. */
				loop->init =
					QC_AST_BASE(create_assign(
						try_create_access(copy_ast(QC_AST_BASE(decl->ident))),
						decl->value /* Note: no copy */
					));
				decl->value = NULL; /* Moved to assign */

				/* Insert decl to the beginning of the scope containing the loop */
				insert_array(QC_AST_Node_Ptr)(&scope->nodes, 0, (QC_AST_Node**)&decl, 1);
			}
		} break;
		case QC_AST_scope: {
			QC_CASTED_NODE(QC_AST_Scope, scope, parent);
			int ix = find_in_scope(scope, QC_AST_BASE(decl));
			int target_ix = 0;
			ASSERT(ix >= 0);

			while (target_ix < ix) {
				QC_AST_Node_Type t = scope->nodes.data[target_ix]->type;
				if (t != QC_AST_var_decl && t != QC_AST_type_decl)
					break;
				++target_ix;
			}

			if (target_ix == ix)
				break;

			if (decl->value) {
				/* Change former decl to be the initialization. */
				scope->nodes.data[ix] =
					QC_AST_BASE(create_assign(
						try_create_access(copy_ast(QC_AST_BASE(decl->ident))),
						decl->value /* Note: no copy */
					));
				decl->value = NULL; /* Moved to assign node */
			} else {
				erase_array(QC_AST_Node_Ptr)(&scope->nodes, ix, 1);
			}

			/* Lift variable declaration to be the last decl of the scope. */
			ASSERT(ix > target_ix);
			insert_array(QC_AST_Node_Ptr)(&scope->nodes, target_ix, (QC_AST_Node**)&decl, 1);
			
		} break;
		default:;
		}
	}

	destroy_parent_map(&map);
	destroy_array(QC_AST_Node_Ptr)(&subnodes);
}

typedef struct Trav_Ctx {
	int depth;
	/* Maps nodes from source QC_AST tree to copied/modified QC_AST tree */
	QC_Hash_Table(QC_AST_Node_Ptr, QC_AST_Node_Ptr) src_to_dst;
} Trav_Ctx;

/* Establish mapping */
INTERNAL void map_nodes(Trav_Ctx *ctx, QC_AST_Node *dst, QC_AST_Node *src)
{ set_tbl(QC_AST_Node_Ptr, QC_AST_Node_Ptr)(&ctx->src_to_dst, src, dst); }

/* Retrieve mapping */
INTERNAL QC_AST_Node *mapped_node(Trav_Ctx *ctx, QC_AST_Node *src)
{ return get_tbl(QC_AST_Node_Ptr, QC_AST_Node_Ptr)(&ctx->src_to_dst, src); }


/* Creates copy of (partial) QC_AST, dropping type and func decls */
/* @todo Remove in-place. This is almost identical to copy_ast */
INTERNAL QC_AST_Node * copy_excluding_types_and_funcs(Trav_Ctx *ctx, QC_AST_Node *node)
{
	QC_AST_Node *copy = NULL;
	QC_Array(QC_AST_Node_Ptr) subnodes;
	QC_Array(QC_AST_Node_Ptr) refnodes;
	QC_Array(QC_AST_Node_Ptr) copied_subnodes;
	QC_Array(QC_AST_Node_Ptr) remapped_refnodes;
	int i;

	if (!node)
		return NULL;
	if (ctx->depth > 0 && (node->type == QC_AST_type_decl || node->type == QC_AST_func_decl))
		return NULL;

	++ctx->depth;

	{
		copy = create_ast_node(node->type);
		/* Map nodes before recursing -- dependencies are always to previous nodes */
		map_nodes(ctx, copy, node);

		/* @todo Do something for the massive number of allocations */
		subnodes = create_array(QC_AST_Node_Ptr)(0);
		refnodes = create_array(QC_AST_Node_Ptr)(0);

		push_immediate_subnodes(&subnodes, node);
		push_immediate_refnodes(&refnodes, node);

		copied_subnodes = create_array(QC_AST_Node_Ptr)(subnodes.size);
		remapped_refnodes = create_array(QC_AST_Node_Ptr)(refnodes.size);

		/* Copy subnodes */
		for (i = 0; i < subnodes.size; ++i) {
			QC_AST_Node *copied_sub = copy_excluding_types_and_funcs(ctx, subnodes.data[i]);
			push_array(QC_AST_Node_Ptr)(&copied_subnodes, copied_sub);
		}
		/* Remap referenced nodes */
		for (i = 0; i < refnodes.size; ++i) {
			QC_AST_Node *remapped = mapped_node(ctx, refnodes.data[i]);
			push_array(QC_AST_Node_Ptr)(&remapped_refnodes, remapped);
		}

		/* Fill created node with nodes of the destination tree and settings of the original node */
		copy_ast_node(	copy, node,
						copied_subnodes.data, copied_subnodes.size,
						remapped_refnodes.data, remapped_refnodes.size);

		destroy_array(QC_AST_Node_Ptr)(&copied_subnodes);
		destroy_array(QC_AST_Node_Ptr)(&remapped_refnodes);
		destroy_array(QC_AST_Node_Ptr)(&subnodes);
		destroy_array(QC_AST_Node_Ptr)(&refnodes);
	}

	--ctx->depth;

	return copy;
}

void lift_types_and_funcs_to_global_scope(QC_AST_Scope *root)
{
	Trav_Ctx ctx = {0};
	QC_AST_Scope *dst = create_ast();
	int i, k;

	/* @todo Size should be something like TOTAL_NODE_COUNT*2 */
	ctx.src_to_dst = create_tbl(QC_AST_Node_Ptr, QC_AST_Node_Ptr)(NULL, NULL, 1024);

	for (i = 0; i < root->nodes.size; ++i) {
		QC_AST_Node *sub = root->nodes.data[i];
		QC_Array(QC_AST_Node_Ptr) decls = create_array(QC_AST_Node_Ptr)(0);
		find_subnodes_of_type(&decls, QC_AST_type_decl, sub);
		find_subnodes_of_type(&decls, QC_AST_func_decl, sub);

		/* Lifted types and funcs */
		for (k = 0; k < decls.size; ++k) {
			/* @todo Rename the declarations to avoid name clashes */
			QC_AST_Node *dst_decl = copy_excluding_types_and_funcs(&ctx, decls.data[k]);
			if (!dst_decl)
				continue;
			map_nodes(&ctx, dst_decl, decls.data[k]);
			push_array(QC_AST_Node_Ptr)(&dst->nodes, dst_decl);
		}
		destroy_array(QC_AST_Node_Ptr)(&decls);

		{ /* Copy bulk without inner types or funcs */
			QC_AST_Node *copy = copy_excluding_types_and_funcs(&ctx, sub);
			if (copy) {
				map_nodes(&ctx, copy, sub);
				push_array(QC_AST_Node_Ptr)(&dst->nodes, copy);
			}
		}
	}

	destroy_tbl(QC_AST_Node_Ptr, QC_AST_Node_Ptr)(&ctx.src_to_dst);

	move_ast(root, dst);
}

/* Modifies the QC_AST */
void add_builtin_c_decls_to_global_scope(QC_AST_Scope *root, bool cpu_device_impl)
{
	int i, k;
	QC_AST_Func_Decl *last_alloc_field_func = NULL;
	QC_AST_Func_Decl *last_free_field_func = NULL;

	/* Create c decls for matrix and field builtin types */
	for (i = 0; i < root->nodes.size; ++i) {
		QC_AST_Node *node = root->nodes.data[i];
		QC_AST_Node *generated[4] = {0};

		/* Matrix and field type processing */
		if (node->type == QC_AST_type_decl) {
			QC_CASTED_NODE(QC_AST_Type_Decl, decl, node);
			int elem_count = 1;
			QC_AST_Type_Decl *mat_decl = NULL; /* Created C matrix type decl */
			QC_AST_Var_Decl *member_decl = NULL; /* Member array decl of matrix */
			QC_Builtin_Type bt;

			if (!decl->is_builtin)
				continue;

			bt = decl->builtin_type;
			if (!bt.is_matrix && !bt.is_field)
				continue;

			for (k = 0; k < bt.matrix_rank; ++k)
				elem_count *= bt.matrix_dim[k];

	 		{ /* Create matrix/field type decl */
				mat_decl = create_type_decl_node();
				mat_decl->ident = create_ident_for_builtin(bt);
				mat_decl->body = create_scope_node();
				decl->builtin_concrete_decl = mat_decl;

				{ /* struct members */
					QC_AST_Type_Decl *m_type_decl = create_type_decl_node(); /* @todo Use existing type decl */

					/* Copy base type from matrix type for the struct member */
					m_type_decl->is_builtin = true;
					m_type_decl->builtin_type = bt;
					if (m_type_decl->builtin_type.is_field)
						m_type_decl->builtin_type.is_field = false;
					else if (m_type_decl->builtin_type.is_matrix)
						m_type_decl->builtin_type.is_matrix = false;

					member_decl = create_simple_var_decl(m_type_decl, "m");
					if (bt.is_field)
						++member_decl->type->ptr_depth;
					else
						member_decl->type->array_size = elem_count;
					push_array(QC_AST_Node_Ptr)(&mat_decl->body->nodes, QC_AST_BASE(member_decl));

					if (bt.is_field) {
						{ /* Field has runtime size */
							QC_AST_Type_Decl *s_type_decl = create_type_decl_node(); /* @todo Use existing type decl */
							QC_AST_Var_Decl *s_decl;

							/* Copy base type from matrix type for the struct member */
							s_type_decl->is_builtin = true;
							s_type_decl->builtin_type.is_integer = true;

							s_decl = create_simple_var_decl(s_type_decl, "size");
							s_decl->type->array_size = bt.field_dim;
							push_array(QC_AST_Node_Ptr)(&mat_decl->body->nodes, QC_AST_BASE(s_decl));
							generated[0] = QC_AST_BASE(s_type_decl);
						}

						{ /* Field has a flag telling if it's a device or a host field */
							QC_AST_Var_Decl *flag_decl =
								create_simple_var_decl(	find_builtin_type_decl(int_builtin_type(), root),
														"is_device_field");

							push_array(QC_AST_Node_Ptr)(&mat_decl->body->nodes, QC_AST_BASE(flag_decl));
						}
					}

					ASSERT(m_type_decl);
					generated[1] = QC_AST_BASE(m_type_decl);
				}

				ASSERT(mat_decl);
				generated[2] = QC_AST_BASE(mat_decl);
			}

			/* Create matrix multiplication func */
			if (!bt.is_field &&
					bt.matrix_rank == 2) { /* @todo General multiplication algo */
				QC_AST_Func_Decl *mul_decl = create_func_decl_node();
				QC_AST_Var_Decl *lhs_decl = NULL;
				QC_AST_Var_Decl *rhs_decl = NULL;
				mul_decl->ident = create_ident_for_builtin(bt);
				append_str(&mul_decl->ident->text, "_mul");

				mul_decl->return_type = create_type_node();
				mul_decl->return_type->base_type_decl = mat_decl;

				{ /* Params */
					lhs_decl = create_simple_var_decl(mat_decl, "lhs");
					rhs_decl = create_simple_var_decl(mat_decl, "rhs");

					push_array(QC_AST_Var_Decl_Ptr)(&mul_decl->params, lhs_decl);
					push_array(QC_AST_Var_Decl_Ptr)(&mul_decl->params, rhs_decl);
				}

				{ /* Body */
					int x, y;
					QC_AST_Var_Decl *ret_decl = create_simple_var_decl(mat_decl, "ret");
					QC_AST_Control *return_stmt = create_control_node();

					mul_decl->body = create_scope_node();
					push_array(QC_AST_Node_Ptr)(&mul_decl->body->nodes, QC_AST_BASE(ret_decl));

					/* Expression for matrix multiplication */
					for (x = 0; x < bt.matrix_dim[0]; ++x) {
						for (y = 0; y < bt.matrix_dim[1]; ++y) {
							int index = x + y*bt.matrix_dim[1];

							QC_AST_Biop *assign = create_biop_node();
							QC_AST_Access *member_access =
								create_member_access(shallow_copy_ast(QC_AST_BASE(ret_decl->ident)), member_decl);
							QC_AST_Access *array_access =
								create_array_access(QC_AST_BASE(member_access),
													QC_AST_BASE(create_integer_literal(index, NULL)));

							assign->type = QC_Token_assign;
							assign->lhs = QC_AST_BASE(array_access);
							assign->rhs = create_matrix_mul_expr(lhs_decl, rhs_decl, member_decl, bt, x, y);
							push_array(QC_AST_Node_Ptr)(&mul_decl->body->nodes, QC_AST_BASE(assign));
						}
					}

					return_stmt->type = QC_Token_kw_return;
					return_stmt->value = QC_AST_BASE(create_access_for_var(ret_decl));
					push_array(QC_AST_Node_Ptr)(&mul_decl->body->nodes, QC_AST_BASE(return_stmt));
				}

				generated[3] = QC_AST_BASE(mul_decl);
			}
		} else if (node->type == QC_AST_func_decl) {
			/* Create concrete field alloc and dealloc funcs */

			QC_CASTED_NODE(QC_AST_Func_Decl, func_decl, node);
			if (!func_decl->is_builtin)
				continue;

			if (!strcmp(func_decl->ident->text.data, "alloc_field")) {
				QC_AST_Func_Decl *alloc_func = (QC_AST_Func_Decl*)copy_ast(QC_AST_BASE(func_decl));
				QC_AST_Type_Decl *field_decl = alloc_func->return_type->base_type_decl;
				QC_Builtin_Type bt = field_decl->builtin_type;
				field_decl = field_decl->builtin_concrete_decl;

				ASSERT(!alloc_func->body);

				alloc_func->is_builtin = false;
				func_decl->builtin_concrete_decl = alloc_func;
				append_str(&alloc_func->ident->text, "_");
				append_builtin_type_c_str(&alloc_func->ident->text,
						alloc_func->return_type->base_type_decl->builtin_type);

				{ /* Function contents */
					QC_AST_Var_Decl *field_var_decl;
					QC_AST_Biop *sizeof_expr;
					QC_Array(QC_AST_Node_Ptr) size_accesses = create_array(QC_AST_Node_Ptr)(alloc_func->params.size);
					QC_AST_Biop *elements_assign;
					QC_AST_Biop *is_device_field_assign;
					QC_AST_Control *ret_stmt;

					alloc_func->body = create_scope_node();

					field_var_decl = create_simple_var_decl(field_decl, "field");
					push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(field_var_decl));

					sizeof_expr = c_create_mat_element_sizeof(field_var_decl);
					push_array(QC_AST_Node_Ptr)(&size_accesses, QC_AST_BASE(sizeof_expr));
					for (k = 0; k < alloc_func->params.size; ++k) {
						push_array(QC_AST_Node_Ptr)(&size_accesses,
							QC_AST_BASE(create_access_for_var(alloc_func->params.data[k])));
					}

					elements_assign =
						create_assign(
							QC_AST_BASE(create_member_access(
								copy_ast(QC_AST_BASE(field_var_decl->ident)), /* @todo Access var */
								c_mat_elements_decl(field_decl)
							)),
							QC_AST_BASE(create_cast( /* For c++ */
								(QC_AST_Type*)copy_ast(QC_AST_BASE(c_mat_elements_decl(field_decl)->type)),
								QC_AST_BASE(create_call_1(
									create_ident_with_text(NULL, "malloc"),
									create_chained_expr(size_accesses.data, size_accesses.size, QC_Token_mul)
								))
							))
						);
					push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(elements_assign));
					destroy_array(QC_AST_Node_Ptr)(&size_accesses);

					for (k = 0; k < bt.field_dim; ++k) {
						QC_AST_Biop *assign =
							create_assign(
								QC_AST_BASE(c_create_field_dim_size(create_simple_access(field_var_decl), k)),
								QC_AST_BASE(create_access_for_var(
									alloc_func->params.data[k]
								))
							);
						push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(assign));
					}

					is_device_field_assign =
						create_assign(
							QC_AST_BASE(create_simple_member_access(
								field_var_decl,
								is_device_field_member_decl(field_decl)
							)),
							QC_AST_BASE(create_integer_literal(0, root))
						);
					push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(is_device_field_assign));

					ret_stmt =
						create_return(QC_AST_BASE(
							create_access_for_var(field_var_decl)
						));
					push_array(QC_AST_Node_Ptr)(&alloc_func->body->nodes, QC_AST_BASE(ret_stmt));
				}

				last_alloc_field_func = alloc_func;
				generated[0] = QC_AST_BASE(alloc_func);
			} else if (!strcmp(func_decl->ident->text.data, "free_field")) {
				QC_AST_Func_Decl *free_func = (QC_AST_Func_Decl*)copy_ast(QC_AST_BASE(func_decl));
				QC_AST_Type_Decl *field_decl = free_func->params.data[0]->type->base_type_decl;
				QC_Builtin_Type bt = field_decl->builtin_type;
				field_decl = field_decl->builtin_concrete_decl;

				ASSERT(!free_func->body);
				ASSERT(free_func->params.size == 1);

				free_func->is_builtin = false;
				func_decl->builtin_concrete_decl = free_func;
				append_str(&free_func->ident->text, "_");
				append_builtin_type_c_str(&free_func->ident->text, bt);

				{ /* Function contents */
					QC_AST_Access *access_field_data = create_member_access(
							copy_ast(QC_AST_BASE(free_func->params.data[0]->ident)),
							c_mat_elements_decl(field_decl));
					QC_AST_Call *libc_free_call = create_call_1(
							create_ident_with_text(NULL, "free"),
							QC_AST_BASE(access_field_data));

					free_func->body = create_scope_node();
					push_array(QC_AST_Node_Ptr)(&free_func->body->nodes, QC_AST_BASE(libc_free_call));
				}

				last_free_field_func = free_func;
				generated[0] = QC_AST_BASE(free_func);
			} else if (cpu_device_impl && !strcmp(func_decl->ident->text.data, "alloc_device_field")) {
				/* @todo Call to alloc_field */
				ASSERT(last_alloc_field_func);
				func_decl->builtin_concrete_decl = last_alloc_field_func;
			} else if (cpu_device_impl && !strcmp(func_decl->ident->text.data, "free_device_field")) {
				/* @todo Call to free_field */
				ASSERT(last_free_field_func);
				func_decl->builtin_concrete_decl = last_free_field_func;
			} else if (cpu_device_impl && !strcmp(func_decl->ident->text.data, "memcpy_field")) {
				QC_AST_Func_Decl *memcpy_func = (QC_AST_Func_Decl*)copy_ast(QC_AST_BASE(func_decl));
				QC_AST_Var_Decl *dst_field_var_decl = memcpy_func->params.data[0];
				QC_AST_Type_Decl *field_decl = dst_field_var_decl->type->base_type_decl;
				QC_Builtin_Type bt = field_decl->builtin_type;
				field_decl = field_decl->builtin_concrete_decl; /* From builtin to struct decl */

				memcpy_func->is_builtin = false;
				func_decl->builtin_concrete_decl = memcpy_func;

				append_str(&memcpy_func->ident->text, "_");
				append_builtin_type_c_str(&memcpy_func->ident->text, bt);

				{ /* Function contents */
					/* @todo Generate matching size assert */
					QC_AST_Access *access_dst_field = create_member_access(
							copy_ast(QC_AST_BASE(memcpy_func->params.data[0]->ident)),
							c_mat_elements_decl(field_decl));
					QC_AST_Access *access_src_field = create_member_access(
							copy_ast(QC_AST_BASE(memcpy_func->params.data[1]->ident)),
							c_mat_elements_decl(field_decl));
					QC_AST_Call *libc_free_call;

					QC_Array(QC_AST_Node_Ptr) size_accesses = create_array(QC_AST_Node_Ptr)(3);
					push_array(QC_AST_Node_Ptr)(&size_accesses,
							QC_AST_BASE(c_create_mat_element_sizeof(dst_field_var_decl)));
					for (k = 0; k < bt.field_dim; ++k) {
						push_array(QC_AST_Node_Ptr)(&size_accesses, 
							QC_AST_BASE(c_create_field_dim_size(create_simple_access(dst_field_var_decl), k)));
					}

					libc_free_call = create_call_3(
							create_ident_with_text(NULL, "memcpy"),
							QC_AST_BASE(access_dst_field),
							QC_AST_BASE(access_src_field),
							create_chained_expr(size_accesses.data, size_accesses.size, QC_Token_mul)
							);

					destroy_array(QC_AST_Node_Ptr)(&size_accesses);

					memcpy_func->body = create_scope_node();
					push_array(QC_AST_Node_Ptr)(&memcpy_func->body->nodes, QC_AST_BASE(libc_free_call));
				}

				generated[0] = QC_AST_BASE(memcpy_func);
			} else if (!strcmp(func_decl->ident->text.data, "size")) {
				QC_AST_Func_Decl *size_func = (QC_AST_Func_Decl*)copy_ast(QC_AST_BASE(func_decl));
				QC_AST_Type_Decl *field_decl = size_func->params.data[0]->type->base_type_decl;
				QC_Builtin_Type bt = field_decl->builtin_type;
				field_decl = field_decl->builtin_concrete_decl;

				ASSERT(!size_func->body);
				ASSERT(size_func->params.size == 2);

				size_func->is_builtin = false;
				func_decl->builtin_concrete_decl = size_func;
				append_str(&size_func->ident->text, "_");
				append_builtin_type_c_str(&size_func->ident->text, bt);

				size_func->body =
					create_scope_1(
						QC_AST_BASE(create_return(
							QC_AST_BASE(create_member_array_access(
								copy_ast(QC_AST_BASE(size_func->params.data[0]->ident)), /* @todo Access var */
								c_field_size_decl(field_decl),
								copy_ast(QC_AST_BASE(size_func->params.data[1]->ident)), /* @todo Access var */
								false
							))
						))
					);

				generated[0] = QC_AST_BASE(size_func);
			} else {

				/* @todo Make post-pass for ast which shows error for unresolved/implemented symbols */
				/*FAIL((	"add_builtin_c_decls_to_global_scope: Unknown field function: %s",
						func_decl->ident->text.data));*/
			}
		}

		for (k = 0; k < (int)(sizeof(generated)/sizeof(*generated)); ++k) {
			if (!generated[k])
				continue;

			/* Insert after current node */
			insert_array(QC_AST_Node_Ptr)(&root->nodes, i + 1, &generated[k], 1);
			++i;
		}
	}
}

void apply_c_operator_overloading(QC_AST_Scope *root, bool convert_mat_expr)
{
	int i, k;
	QC_Array(QC_AST_Node_Ptr) replace_list_old = create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) replace_list_new = create_array(QC_AST_Node_Ptr)(0);
	QC_Array(QC_AST_Node_Ptr) subnodes = create_array(QC_AST_Node_Ptr)(0);
	push_subnodes(&subnodes, QC_AST_BASE(root), false);

	for (i = 0; i < subnodes.size; ++i) {
		/* Handle matrix "operator overloading" */
		if (convert_mat_expr && subnodes.data[i]->type == QC_AST_biop) {
			QC_CASTED_NODE(QC_AST_Biop, biop, subnodes.data[i]);
			QC_AST_Type type;
			QC_Builtin_Type bt;

			if (biop->type == QC_Token_assign)
				continue;
			if (!expr_type(&type, QC_AST_BASE(biop)))
				continue;
			if (!type.base_type_decl->is_builtin)
				continue;
			bt = type.base_type_decl->builtin_type;
			if (!bt.is_matrix)
				continue;

			{
				QC_AST_Call *call = create_call_node();

				/* @todo Link ident to matrix type */
				call->ident = create_ident_node();
				append_builtin_type_c_str(&call->ident->text, bt);
				/* @todo Handle all operations */
				append_str(&call->ident->text, "_mul");

				{ /* Args */
					push_array(QC_AST_Node_Ptr)(&call->args, biop->lhs);
					push_array(QC_AST_Node_Ptr)(&call->args, biop->rhs);
				}

				/* Mark biop to be replaced with the function call */
				push_array(QC_AST_Node_Ptr)(&replace_list_old, QC_AST_BASE(biop));
				push_array(QC_AST_Node_Ptr)(&replace_list_new, QC_AST_BASE(call));
			}
		}

		/* Convert matrix/field element accesses to member array accesses */
		if (subnodes.data[i]->type == QC_AST_access) {
			QC_CASTED_NODE(QC_AST_Access, access, subnodes.data[i]);
			QC_AST_Type type;
			QC_Builtin_Type bt;

			if (!access->is_element_access)
				continue;
			if (!expr_type(&type, access->base))
				FAIL(("expr_type failed for access"));
			if (!type.base_type_decl->is_builtin)
				continue;
			bt = type.base_type_decl->builtin_type;
			if (!bt.is_matrix && !bt.is_field)
				continue;

			{ /* Transform from mat_or_field(1, 2) -> mat_or_field.m[1 + 2*sizex] */
				QC_AST_Type_Decl *mat_decl = concrete_type_decl(bt, root);
				QC_AST_Var_Decl *member_decl = c_mat_elements_decl(mat_decl);
				QC_AST_Access *member_access = create_access_node();
				QC_AST_Access *array_access = create_access_node();

				member_access->base = access->base;
				push_array(QC_AST_Node_Ptr)(&member_access->args, copy_ast(QC_AST_BASE(member_decl->ident)));
				member_access->implicit_deref = access->implicit_deref;
				member_access->is_member_access = true;

				array_access->base = QC_AST_BASE(member_access);
				array_access->is_array_access = true;

				if (bt.is_field) {
					/* Field access */
					QC_Array(QC_AST_Node_Ptr) multipliers = create_array(QC_AST_Node_Ptr)(0);
					QC_AST_Node *index_expr = NULL;
					QC_AST_Var_Decl *size_member_decl = c_field_size_decl(mat_decl);
					for (k = 0; k < bt.field_dim; ++k) {
						if (k == 0) {
							push_array(QC_AST_Node_Ptr)(&multipliers, QC_AST_BASE(create_integer_literal(1, NULL)));
						} else {
							QC_AST_Node *field_access = copy_ast(access->base);
							QC_AST_Access *size_access =
								create_member_array_access(
										field_access,
										size_member_decl,
										QC_AST_BASE(create_integer_literal(k - 1, NULL)),
										access->implicit_deref
								);

							if (k == 1) {
								push_array(QC_AST_Node_Ptr)(&multipliers, QC_AST_BASE(size_access));
							} else {
								QC_AST_Biop *mul = create_mul(copy_ast(multipliers.data[k - 1]), QC_AST_BASE(size_access));
								push_array(QC_AST_Node_Ptr)(&multipliers, QC_AST_BASE(mul));
							}
						}
					}
					ASSERT(multipliers.size == access->args.size);
					index_expr = create_chained_expr_2(multipliers.data, access->args.data,
							access->args.size, QC_Token_mul, QC_Token_add);
					push_array(QC_AST_Node_Ptr)(&array_access->args, index_expr);

					destroy_array(QC_AST_Node_Ptr)(&multipliers);

				} else {
					/* Matrix access */
					QC_Array(QC_AST_Node_Ptr) multipliers = create_array(QC_AST_Node_Ptr)(0);
					QC_AST_Node *index_expr = NULL;
					int mul_accum = 1;
					for (k = 0; k < bt.matrix_rank; ++k) {
						push_array(QC_AST_Node_Ptr)(&multipliers, QC_AST_BASE(create_integer_literal(mul_accum, NULL)));
						mul_accum *= bt.matrix_dim[k];
					}
					ASSERT(multipliers.size == access->args.size);
					index_expr = create_chained_expr_2(multipliers.data, access->args.data,
							access->args.size, QC_Token_mul, QC_Token_add);
					push_array(QC_AST_Node_Ptr)(&array_access->args, index_expr);

					destroy_array(QC_AST_Node_Ptr)(&multipliers);
				}

				push_array(QC_AST_Node_Ptr)(&replace_list_old, QC_AST_BASE(access));
				push_array(QC_AST_Node_Ptr)(&replace_list_new, QC_AST_BASE(array_access));
			}
		}
	}

	{ /* Replace old nodes with new nodes */
		ASSERT(replace_list_new.size == replace_list_old.size);
		replace_nodes_in_ast(QC_AST_BASE(root), replace_list_old.data, replace_list_new.data, replace_list_new.size);

		/* No deep copies of branches */
		for (i = 0; i < replace_list_old.size; ++i)
			shallow_destroy_node(replace_list_old.data[i]);
	}

	destroy_array(QC_AST_Node_Ptr)(&subnodes);
	destroy_array(QC_AST_Node_Ptr)(&replace_list_old);
	destroy_array(QC_AST_Node_Ptr)(&replace_list_new);
}

INTERNAL void append_c_comment(QC_Array(char) *buf, QC_Token *comment)
{
	if (comment->type == QC_Token_line_comment)
		append_str(buf, "/*%.*s */", BUF_STR_ARGS(comment->text));
	else
		append_str(buf, "/*%.*s*/", BUF_STR_ARGS(comment->text));
}

void append_c_stdlib_includes(QC_Array(char) *buf)
{
	append_str(buf, "#include <stdio.h>\n");
	append_str(buf, "#include <stdlib.h>\n");
	append_str(buf, "#include <string.h> /* memcpy */\n");
	append_str(buf, "\n");
}

bool ast_to_c_str(QC_Array(char) *buf, int indent, QC_AST_Node *node)
{
	int i, k;
	bool omitted = false;
	int indent_add = 4;

	if (node->attribute)
		append_str(buf, "%s ", node->attribute);

	switch (node->type) {
	case QC_AST_scope: {
		QC_CASTED_NODE(QC_AST_Scope, scope, node);
		int new_indent = indent + indent_add;
		if (scope->is_root)
			new_indent = 0;

		if (!scope->is_root)
			append_str(buf, "{\n");
		for (i = 0; i < scope->nodes.size; ++i) {
			QC_AST_Node *sub = scope->nodes.data[i];
			bool statement_omitted;

			/* Comments are enabled only for scope nodes for now */
			for (k = 0; k < sub->pre_comments.size; ++k) {
				QC_Token *comment = sub->pre_comments.data[k];
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

			if (!statement_omitted &&	sub->type != QC_AST_func_decl &&
										sub->type != QC_AST_scope &&
										sub->type != QC_AST_cond &&
										sub->type != QC_AST_loop)
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

	case QC_AST_ident: {
		QC_CASTED_NODE(QC_AST_Ident, ident, node);
		if (ident->designated)
			append_str(buf, ".");
		append_str(buf, "%s", ident->text.data);
	} break;

	case QC_AST_type: {
		/* Print type without identifier (like in casts)*/
		QC_CASTED_NODE(QC_AST_Type, type, node);
		append_type_and_ident_str(buf, type, NULL);
	} break;

	case QC_AST_type_decl: {
		QC_CASTED_NODE(QC_AST_Type_Decl, decl, node);
		if (decl->is_builtin) {
			omitted = true;
		} else {
			append_str(buf, "typedef struct ");
			append_str(buf, "%s\n", decl->ident->text.data);
			ast_to_c_str(buf, indent, QC_AST_BASE(decl->body));
			append_str(buf, " %s", decl->ident->text.data);
		}
	} break;

	case QC_AST_var_decl: {
		QC_CASTED_NODE(QC_AST_Var_Decl, decl, node);
		append_type_and_ident_str(buf, decl->type, decl->ident->text.data);
		if (decl->value) {
			append_str(buf, " = ");
			ast_to_c_str(buf, indent, decl->value);
		}
	} break;

	case QC_AST_func_decl: {
		QC_CASTED_NODE(QC_AST_Func_Decl, decl, node);
		if (decl->is_builtin) {
			omitted = true;
			break;
		}
		append_type_and_ident_str(buf, decl->return_type, decl->ident->text.data);
		append_str(buf, "(");
		for (i = 0; i < decl->params.size; ++i) {
			ast_to_c_str(buf, indent, QC_AST_BASE(decl->params.data[i]));
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
			ast_to_c_str(buf, indent, QC_AST_BASE(decl->body));
		} else {
			append_str(buf, ";");
		}
	} break;

	case QC_AST_literal: {
		QC_CASTED_NODE(QC_AST_Literal, literal, node);
		switch (literal->type) {
		case QC_Literal_int: append_str(buf, "%i", literal->value.integer); break;
		case QC_Literal_float: append_str(buf, "%f", literal->value.floating); break;
		case QC_Literal_string: append_str(buf, "\"%.*s\"", literal->value.string.len, literal->value.string.buf); break;
		case QC_Literal_null: append_str(buf, "NULL"); break;
		case QC_Literal_compound: {
			if (literal->value.compound.type) {
				/* Compound literal */
				append_str(buf, "(");
				ast_to_c_str(buf, indent, QC_AST_BASE(literal->value.compound.type));
				append_str(buf, ") ");
			}

			append_str(buf, "{ ");
			for (i = 0; i < literal->value.compound.subnodes.size; ++i) {
				ast_to_c_str(buf, indent, literal->value.compound.subnodes.data[i]);
				if (i + 1 < literal->value.compound.subnodes.size)
					append_str(buf, ", ");
			}
			append_str(buf, " }");
		} break;
		default: FAIL(("Unknown literal type: %i", literal->type));
		}
	} break;

	case QC_AST_biop: {
		QC_CASTED_NODE(QC_AST_Biop, biop, node);
		if (biop->lhs && biop->rhs) {
			bool lhs_parens = nested_expr_needs_parens(node, biop->lhs);
			bool rhs_parens = nested_expr_needs_parens(node, biop->rhs);
			if (lhs_parens)
				append_str(buf, "(");
			ast_to_c_str(buf, indent, biop->lhs);
			if (lhs_parens)
				append_str(buf, ")");

			append_str(buf, " %s ", qc_tokentype_codestr(biop->type));

			if (rhs_parens)
				append_str(buf, "(");
			ast_to_c_str(buf, indent, biop->rhs);
			if (rhs_parens)
				append_str(buf, ")");
		} else {
			bool parens_inside = (biop->type == QC_Token_kw_sizeof);

			/* Unary op */
			append_str(buf, "%s", qc_tokentype_codestr(biop->type));
			
			if (parens_inside)
				append_str(buf, "(");
			ast_to_c_str(buf, indent, biop->rhs);
			if (parens_inside)
				append_str(buf, ")");
		}
	} break;

	case QC_AST_control: {
		QC_CASTED_NODE(QC_AST_Control, control, node);
		append_str(buf, "%s", qc_tokentype_codestr(control->type));
		if (control->value) {
			append_str(buf, " ");
			ast_to_c_str(buf, indent, control->value);
		}
	} break;

	case QC_AST_call: {
		QC_CASTED_NODE(QC_AST_Call, call, node);
		QC_CASTED_NODE(QC_AST_Func_Decl, decl, call->ident->decl);
		if (decl && decl->is_builtin) {
			/* e.g. alloc_field -> alloc_field_floatfield2 */
			ast_to_c_str(buf, indent, QC_AST_BASE(decl->builtin_concrete_decl->ident));
		} else {
			ast_to_c_str(buf, indent, QC_AST_BASE(call->ident));
		}
		append_str(buf, "(");
		for (i = 0; i < call->args.size; ++i) {
			ast_to_c_str(buf, indent, call->args.data[i]);
			if (i + 1 < call->args.size)
				append_str(buf, ", ");
		}
		append_str(buf, ")");
	} break;

	case QC_AST_access: {
		QC_CASTED_NODE(QC_AST_Access, access, node);
		if (access->is_member_access) {
			bool parens = (access->base->type != QC_AST_ident && access->base->type != QC_AST_access);
			if (parens)
				append_str(buf, "(");
			ast_to_c_str(buf, indent, access->base);
			if (parens)
				append_str(buf, ")");
			if (access->implicit_deref)
				append_str(buf, "->");
			else
				append_str(buf, ".");
			ASSERT(access->args.size == 1);
			ast_to_c_str(buf, indent, access->args.data[0]);
		} else if (access->is_array_access) {
			ast_to_c_str(buf, indent, access->base);
			append_str(buf, "[");
			ASSERT(access->args.size == 1);
			ast_to_c_str(buf, indent, access->args.data[0]);
			append_str(buf, "]");
		} else if (access->is_element_access) {
			FAIL(("C does not support builtin element access (bug: these should be converted)"));
		} else {
			ast_to_c_str(buf, indent, access->base);
		}
	} break;

	case QC_AST_cond: {
		QC_CASTED_NODE(QC_AST_Cond, cond, node);
		append_str(buf, "if (");
		ast_to_c_str(buf, indent, cond->expr);
		append_str(buf, ") ");
		if (cond->body) {
			ast_to_c_str(buf, indent, QC_AST_BASE(cond->body));
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

	case QC_AST_loop: {
		QC_CASTED_NODE(QC_AST_Loop, loop, node);
		if (loop->init || loop->incr) {
			append_str(buf, "for (");
			if (loop->init)
				ast_to_c_str(buf, indent, loop->init);
			append_str(buf, "; ");
			ast_to_c_str(buf, indent, loop->cond);
			append_str(buf, "; ");
			if (loop->incr)
				ast_to_c_str(buf, indent, loop->incr);
			append_str(buf, ") ");
		} else {
			append_str(buf, "while (");
			ast_to_c_str(buf, indent, loop->cond);
			append_str(buf, ") ");
		}

		if (loop->body)
			ast_to_c_str(buf, indent, QC_AST_BASE(loop->body));
		else
			append_str(buf, "\n%*s;", indent + indent_add, "");
	} break;

	case QC_AST_cast: {
		QC_CASTED_NODE(QC_AST_Cast, cast, node);
		append_str(buf, "(");
		ast_to_c_str(buf, indent, QC_AST_BASE(cast->type));
		append_str(buf, ")");
		ast_to_c_str(buf, indent, cast->target);

	} break;

	case QC_AST_typedef: {
		QC_CASTED_NODE(QC_AST_Typedef, def, node);
		append_str(buf, "typedef ");
		append_type_and_ident_str(buf, def->type, def->ident->text.data);
	} break;

	case QC_AST_parallel: {
		FAIL(("Trying to output parallel node to C source"));
	} break;

	default: FAIL(("ast_to_c_str: Unknown node type: %i", node->type));
	}

	return omitted;
}

QC_Array(char) gen_c_code(QC_AST_Scope *root)
{
	QC_Array(char) gen_src = create_array(char)(1024);

	QC_AST_Scope *modified_ast = (QC_AST_Scope*)copy_ast(QC_AST_BASE(root));
	parallel_loops_to_ordinary(modified_ast);
	lift_var_decls(modified_ast);
	lift_types_and_funcs_to_global_scope(modified_ast);
	add_builtin_c_decls_to_global_scope(modified_ast, true);
	apply_c_operator_overloading(modified_ast, true);

	append_c_stdlib_includes(&gen_src);
	ast_to_c_str(&gen_src, 0, QC_AST_BASE(modified_ast));
	destroy_ast(modified_ast);
	return gen_src;
}

