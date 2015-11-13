#include "codegen.h"

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

			if (!statement_omitted && !sub->begin_tok)
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
		append_str(buf, "%.*s", BUF_STR_ARGS(ident->text));
	} break;

	case AST_type: {
		/* @todo This needs to be merged to decls, because type and identifier are mixed in C, like int (*foo)()*/
		CASTED_NODE(AST_Type, type, node);
		if (type->base_type_decl->is_builtin) {
			append_builtin_type_str(buf, type->base_type_decl->builtin_type);
			append_str(buf, " ");
		} else {
			append_str(buf, "%.*s ", BUF_STR_ARGS(type->base_type_decl->ident->text));
		}
		for (i = 0; i < type->ptr_depth; ++i)
			append_str(buf, "*");
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		if (decl->is_builtin) {
			Builtin_Type bt = decl->builtin_type;
			if (bt.is_matrix) {
				int elem_count = 1;

				append_str(buf, "struct ");
				append_builtin_type_str(buf, bt);
				append_str(buf, "\n{\n");

				{ /* Member array type */
					Builtin_Type member = bt;
					member.is_matrix = false;
					append_str(buf , "    ");
					append_builtin_type_str(buf, member);
				}

				for (i = 0; i < bt.matrix_rank; ++i)
					elem_count *= bt.matrix_dim[i];
				append_str(buf, " m[%i];\n", elem_count);

				append_str(buf, "}");
			} else {
				omitted = true;
			}
		} else {
			append_str(buf, "struct ");
			append_str(buf, "%.*s\n", BUF_STR_ARGS(decl->ident->text));
			ast_to_c_str(buf, indent, AST_BASE(decl->body));
		}
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		ast_to_c_str(buf, indent, AST_BASE(decl->type));
		append_str(buf, "%.*s", BUF_STR_ARGS(decl->ident->text));
		if (decl->value) {
			append_str(buf, " = ");
			ast_to_c_str(buf, indent, decl->value);
		}
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		ast_to_c_str(buf, indent, AST_BASE(decl->return_type));
		append_str(buf, "%.*s", BUF_STR_ARGS(decl->ident->text));
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

	default:;
	}

	return omitted;
}

Array(char) gen_c_code(AST_Scope *root)
{
	Array(char) buf = create_array(char)(1024);
	AST_Scope *c_ast = lift_types_and_funcs_to_global_scope(root);
	ast_to_c_str(&buf, 0, AST_BASE(c_ast));
	destroy_ast_tree(c_ast);
	return buf;
}

