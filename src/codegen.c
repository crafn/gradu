#include "codegen.h"

INTERNAL bool is_builtin_decl(AST_Node *node)
{
	if (node->type != AST_type_decl)
		return false;
	{
		CASTED_NODE(AST_Type_Decl, decl, node);
		return decl->is_builtin;
	}
}

INTERNAL void append_builtin_type_str(Array(char) *buf, Builtin_Type bt)
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
		append_str(buf, "_mat_");
		for (i = 0; i < bt.matrix_rank; ++i) {
			append_str(buf, "%i", bt.matrix_dim[i]);
			if (i + 1 < bt.matrix_rank)
				append_str(buf, "x");
		}
	}
}

INTERNAL void append_type_and_ident_str(Array(char) *buf, AST_Type *type, AST_Ident *ident)
{
	int i;
	if (type->base_type_decl->is_builtin) {
		append_builtin_type_str(buf, type->base_type_decl->builtin_type);
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

INTERNAL AST_Ident *create_ident_node_for_builtin(Builtin_Type bt)
{
	AST_Ident *ident = create_ident_node();
	append_builtin_type_str(&ident->text, bt);
	return ident;
}

INTERNAL AST_Ident *create_ident_node_with_text(const char *str)
{
	AST_Ident *ident = create_ident_node();
	append_str(&ident->text, "%s", str);
	return ident;
}

/* Innermost first */
INTERNAL void find_subnodes_of_type(Array(AST_Node_Ptr) *ret, AST_Node_Type type, AST_Node *node)
{
	int i;
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	push_subnodes(&subnodes, node, false);

	for (i = 0; i < subnodes.size; ++i) {
		if (subnodes.data[i]->type == type)
			push_array(AST_Node_Ptr)(ret, subnodes.data[i]);
	}

	destroy_array(AST_Node_Ptr)(&subnodes);
}

INTERNAL U32 hash(AST_Node_Ptr)(AST_Node_Ptr node) { return hash(Void_Ptr)(node); }
DECLARE_HASH_TABLE(AST_Node_Ptr, AST_Node_Ptr)
DEFINE_HASH_TABLE(AST_Node_Ptr, AST_Node_Ptr)

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
/* @todo Generalize */
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

/* Returns new AST */
INTERNAL AST_Scope *lift_types_and_funcs_to_global_scope(AST_Scope *root)
{
	Trav_Ctx ctx = {0};
	AST_Scope *dst = create_ast_tree();
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
	return dst;
}

/* Modifies the AST */
INTERNAL void expand_matrices(AST_Scope *root)
{
	int i, k;
	/* Create matrix type declarations and functions */
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) matrix_type_decls = create_array(AST_Node_Ptr)(0);
	push_subnodes(&subnodes, AST_BASE(root), false);

	for (i = 0; i < subnodes.size; ++i) {
		if (subnodes.data[i]->type != AST_type_decl)
			continue;
		{
			CASTED_NODE(AST_Type_Decl, decl, subnodes.data[i]);
			int elem_count = 1;

			if (!decl->is_builtin)
				continue;

			if (!decl->builtin_type.is_matrix)
				continue;

			for (k = 0; k < decl->builtin_type.matrix_rank; ++k)
				elem_count *= decl->builtin_type.matrix_dim[k];

	 		{ /* struct specific_matrix_type { ... }; */
				AST_Type_Decl * mat_decl = create_type_decl_node();
				mat_decl->ident = create_ident_node_for_builtin(decl->builtin_type);
				mat_decl->body = create_scope_node();

				{ /* struct members */
					AST_Type_Decl *m_type_decl = create_type_decl_node(); /* Could use existing type decl */
					AST_Var_Decl *m = create_var_decl_node();

					/* Copy base type from matrix type for the struct member */
					m_type_decl->is_builtin = true;
					m_type_decl->builtin_type = decl->builtin_type;
					m_type_decl->builtin_type.is_matrix = false;

					m->type = create_type_node();
					m->type->base_type_decl = m_type_decl;
					m->type->array_size = elem_count;
					m->ident = create_ident_node_with_text("m");
					ASSERT(m);
					push_array(AST_Node_Ptr)(&mat_decl->body->nodes, AST_BASE(m));

					ASSERT(m_type_decl);
					push_array(AST_Node_Ptr)(&matrix_type_decls, AST_BASE(m_type_decl));
				}

				ASSERT(mat_decl);
				push_array(AST_Node_Ptr)(&matrix_type_decls, AST_BASE(mat_decl));
			}
			/* @todo Matrix functions */
		}
	}

	{ /* Add/substitute matrix stuff with C-compatible matrices and operations */
		int place = 0;
		while (place < root->nodes.size && is_builtin_decl(root->nodes.data[place]))
			++place;
		insert_array(AST_Node_Ptr)(&root->nodes, place, matrix_type_decls.data, matrix_type_decls.size);
	}
	destroy_array(AST_Node_Ptr)(&matrix_type_decls);
	destroy_array(AST_Node_Ptr)(&subnodes);
}

INTERNAL void append_c_comment(Array(char) *buf, Token *comment)
{
	if (comment->type == Token_line_comment)
		append_str(buf, "/*%.*s */", BUF_STR_ARGS(comment->text));
	else
		append_str(buf, "/*%.*s*/", BUF_STR_ARGS(comment->text));
}

/* Almost 1-1 mapping between nodes and C constructs */
INTERNAL bool ast_to_c_str(Array(char) *buf, int indent, AST_Node *node)
{
	int i, k;
	bool omitted = false;

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		int new_indent = indent + 4;
		if (scope->is_root)
			new_indent = 0;

		if (!scope->is_root)
			append_str(buf, "%*s{\n", indent, "");
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

			if (!statement_omitted && sub->type != AST_func_decl)
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
			append_str(buf, "struct ");
			append_str(buf, "%s\n", decl->ident->text.data);
			ast_to_c_str(buf, indent, AST_BASE(decl->body));
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
		ast_to_c_str(buf, indent, biop->lhs);
		append_str(buf, " %s ", tokentype_codestr(biop->type));
		ast_to_c_str(buf, indent, biop->rhs);
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
		ast_to_c_str(buf, indent, AST_BASE(access->base));
		if (access->is_plain_access) {
			;
		} else if (access->is_member_access) {
			append_str(buf, ".");
			ast_to_c_str(buf, indent, access->sub);
		}
	} break;

	default: FAIL(("ast_to_c_str: Unknown node type: %i", node->type));
	}

	return omitted;
}

Array(char) gen_c_code(AST_Scope *root)
{
	Array(char) gen_src = create_array(char)(1024);

	AST_Scope *modified_ast = lift_types_and_funcs_to_global_scope(root);
	expand_matrices(modified_ast);

	print_ast(AST_BASE(modified_ast), 0);

	ast_to_c_str(&gen_src, 0, AST_BASE(modified_ast));

	destroy_ast_tree(modified_ast);
	return gen_src;
}

