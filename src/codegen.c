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

/* @todo Replace with generic linear traversal in dependency (innermost first) order */
INTERNAL void find_subnodes_of_type_impl(Array(AST_Node_Ptr) *result, AST_Node_Type type, AST_Node *node, int depth)
{
	/* @todo Create linear (inner, or outermost first) search for AST and use that */
	int i;
	if (!node)
		return;

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		for (i = 0; i < scope->nodes.size; ++i)
			find_subnodes_of_type_impl(result, type, scope->nodes.data[i], depth + 1);
	} break;

	case AST_ident: {
	} break;

	case AST_type: {
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->ident), depth + 1);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->body), depth + 1);
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->type), depth + 1);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->ident), depth + 1);
		find_subnodes_of_type_impl(result, type, decl->value, depth + 1);
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->return_type), depth + 1);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->ident), depth + 1);
		for (i = 0; i < decl->params.size; ++i)
			find_subnodes_of_type_impl(result, type, AST_BASE(decl->params.data[i]), depth +1);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->body), depth + 1);
	} break;

	case AST_literal: {
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, node);
		find_subnodes_of_type_impl(result, type, biop->lhs, depth + 1);
		find_subnodes_of_type_impl(result, type, biop->rhs, depth + 1);
	} break;

	case AST_control: {
		CASTED_NODE(AST_Control, control, node);
		find_subnodes_of_type_impl(result, type, control->value, depth + 1);
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		find_subnodes_of_type_impl(result, type, AST_BASE(call->ident), depth + 1);
		for (i = 0; i < call->args.size; ++i)
			find_subnodes_of_type_impl(result, type, call->args.data[i], depth +1);
	} break;

	default: FAIL(("find_subnodes_of_type: Unknown node type: %i", type));
	}

	if (depth > 0 && node->type == type)
		push_array(AST_Node_Ptr)(result, node);
}

/* Innermost first */
INTERNAL void find_subnodes_of_type(Array(AST_Node_Ptr) *decls, AST_Node_Type type, AST_Node *node)
{
	find_subnodes_of_type_impl(decls, type, node, 0);
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

/* @todo Replace with generic tree traversal and/or transform macro */
INTERNAL AST_Node * copy_excluding_types_and_funcs_impl(Trav_Ctx *ctx, AST_Node *node)
{
	int i;
	AST_Node *ret = NULL;
	if (!node)
		return NULL;

	++ctx->depth;

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		AST_Scope *copy = create_scope_node();
		map_nodes(ctx, AST_BASE(copy), node);
		{
			Array(AST_Node_Ptr) copied_subnodes = create_array(AST_Node_Ptr)(0);
			for (i = 0; i < scope->nodes.size; ++i) {
				AST_Node *subcopy = copy_excluding_types_and_funcs_impl(ctx, scope->nodes.data[i]);
				if (subcopy)
					push_array(AST_Node_Ptr)(&copied_subnodes, subcopy);
			}
			copy_scope_node(copy, scope, copied_subnodes.data, copied_subnodes.size);
			destroy_array(AST_Node_Ptr)(&copied_subnodes);
		}
		ret = AST_BASE(copy);
	} break;

	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, node);
		AST_Ident *copy = create_ident_node();
		map_nodes(ctx, AST_BASE(copy), node);
		copy_ident_node(copy, ident, mapped_node(ctx, ident->decl));
		ret = AST_BASE(copy);
	} break;

	case AST_type: {
		CASTED_NODE(AST_Type, type, node);
		AST_Type *copy = create_type_node();
		map_nodes(ctx, AST_BASE(copy), node);
		copy_type_node(copy, type, mapped_node(ctx, AST_BASE(type->base_type_decl)));
		ret = AST_BASE(copy);
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		if (ctx->depth > 1)
			break;
		{
			AST_Type_Decl *copy = create_type_decl_node();
			map_nodes(ctx, AST_BASE(copy), node);
			{
				AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->ident));
				AST_Node *copied_body = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->body));
				copy_type_decl_node(copy, decl, copied_ident, copied_body);
				ret = AST_BASE(copy);
			}
		}
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		AST_Var_Decl *copy = create_var_decl_node();
		map_nodes(ctx, AST_BASE(copy), node);
		{
			AST_Node *copied_type = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->type));
			AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->ident));
			AST_Node *copied_value = copy_excluding_types_and_funcs_impl(ctx, decl->value);
			copy_var_decl_node(copy, decl, copied_type, copied_ident, copied_value);
		}
		ret = AST_BASE(copy);
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		if (ctx->depth > 1)
			break;
		{
			AST_Func_Decl *copy = create_func_decl_node();
			map_nodes(ctx, AST_BASE(copy), node);
			{
				AST_Node *copied_ret_type = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->return_type));
				AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->ident));
				AST_Node *copied_body = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->body));
				Array(AST_Node_Ptr) copied_params = create_array(AST_Node_Ptr)(decl->params.size);
				for (i = 0; i < decl->params.size; ++i) {
					AST_Node *paramcopy =
						copy_excluding_types_and_funcs_impl(ctx, AST_BASE(decl->params.data[i]));
					push_array(AST_Node_Ptr)(&copied_params, paramcopy);
				}
				copy_func_decl_node(copy,	decl, copied_ret_type, copied_ident,
											copied_params.data, copied_params.size,
											copied_body);
				destroy_array(AST_Node_Ptr)(&copied_params);
			}
			ret = AST_BASE(copy);
		}
	} break;

	case AST_literal: {
		CASTED_NODE(AST_Literal, literal, node);
		AST_Literal *copy = create_literal_node();
		map_nodes(ctx, AST_BASE(copy), node);
		copy_literal_node(copy, literal);
		ret = AST_BASE(copy);
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, node);
		AST_Biop *copy = create_biop_node();
		map_nodes(ctx, AST_BASE(copy), node);
		copy_biop_node(copy,	biop,
								copy_excluding_types_and_funcs_impl(ctx, biop->lhs),
								copy_excluding_types_and_funcs_impl(ctx, biop->rhs));
		ret = AST_BASE(copy);
	} break;

	case AST_control: {
		CASTED_NODE(AST_Control, control, node);
		AST_Control *copy = create_control_node();
		map_nodes(ctx, AST_BASE(copy), node);
		copy_control_node(copy, control,
								copy_excluding_types_and_funcs_impl(ctx, control->value));
		ret = AST_BASE(copy);
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		AST_Call *copy = create_call_node();
		map_nodes(ctx, AST_BASE(copy), node);
		{
			AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(ctx, AST_BASE(call->ident));
			Array(AST_Node_Ptr) copied_args = create_array(AST_Node_Ptr)(call->args.size);
			for (i = 0; i < call->args.size; ++i) {
				AST_Node *argcopy =
					copy_excluding_types_and_funcs_impl(ctx, call->args.data[i]);
				push_array(AST_Node_Ptr)(&copied_args, argcopy);
			}
			copy_call_node(copy, call, copied_ident, copied_args.data, copied_args.size);
			destroy_array(AST_Node_Ptr)(&copied_args);
		}
		ret = AST_BASE(copy);
	} break;

	default: FAIL(("copy_excluding_types_and_funcs: Unknown node type: %i", node->type));
	}

	--ctx->depth;
	return ret;
}

/* Creates copy of (partial) AST, dropping type and func decls */
INTERNAL AST_Node * copy_excluding_types_and_funcs(Trav_Ctx *ctx, AST_Node *node)
{
	Trav_Ctx depth_reseted_ctx = *ctx;
	depth_reseted_ctx.depth = 0;
	return copy_excluding_types_and_funcs_impl(&depth_reseted_ctx, node);
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
				append_str(buf, ",");
		}
		append_str(buf, ")");
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

