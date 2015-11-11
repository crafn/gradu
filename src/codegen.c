#include "codegen.h"

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

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->ident), depth + 1);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->body), depth + 1);
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		find_subnodes_of_type_impl(result, type, decl->type, depth + 1);
		find_subnodes_of_type_impl(result, type, AST_BASE(decl->ident), depth + 1);
		find_subnodes_of_type_impl(result, type, decl->value, depth + 1);
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		find_subnodes_of_type_impl(result, type, decl->return_type, depth + 1);
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

/* @todo Replace with generic tree traversal macro */
INTERNAL AST_Node * copy_excluding_types_and_funcs_impl(AST_Node *node, int depth)
{
	int i;
	AST_Node *ret = NULL;
	if (!node)
		return NULL;

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		Array(AST_Node_Ptr) copied_subnodes = create_array(AST_Node_Ptr)(0);
		for (i = 0; i < scope->nodes.size; ++i) {
			AST_Node *subcopy = copy_excluding_types_and_funcs_impl(scope->nodes.data[i], depth + 1);
			if (subcopy)
				push_array(AST_Node_Ptr)(&copied_subnodes, subcopy);
		}
		ret = AST_BASE(copy_scope_node(scope, copied_subnodes.data, copied_subnodes.size));
		destroy_array(AST_Node_Ptr)(&copied_subnodes);
	} break;

	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, node);
		ret = AST_BASE(copy_ident_node(ident));
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		if (depth > 0)
			return NULL;
		{
			AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(AST_BASE(decl->ident), depth + 1);
			AST_Node *copied_body = copy_excluding_types_and_funcs_impl(AST_BASE(decl->body), depth + 1);
			ret = AST_BASE(copy_type_decl_node(decl, copied_ident, copied_body));
		}
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		{
			AST_Node *copied_type = copy_excluding_types_and_funcs_impl(decl->type, depth + 1);
			AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(AST_BASE(decl->ident), depth + 1);
			AST_Node *copied_value = copy_excluding_types_and_funcs_impl(decl->value, depth + 1);
			ret = AST_BASE(copy_var_decl_node(decl, copied_type, copied_ident, copied_value));
		}
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		if (depth > 0)
			return NULL;
		{
			AST_Node *copied_ret_type = copy_excluding_types_and_funcs_impl(decl->return_type, depth + 1);
			AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(AST_BASE(decl->ident), depth + 1);
			AST_Node *copied_body = copy_excluding_types_and_funcs_impl(AST_BASE(decl->body), depth + 1);
			Array(AST_Node_Ptr) copied_params = create_array(AST_Node_Ptr)(decl->params.size);
			for (i = 0; i < decl->params.size; ++i) {
				AST_Node *paramcopy =
					copy_excluding_types_and_funcs_impl(AST_BASE(decl->params.data[i]), depth + 1);
				push_array(AST_Node_Ptr)(&copied_params, paramcopy);
			}
			ret = AST_BASE(copy_func_decl_node(	decl, copied_ret_type, copied_ident,
												copied_params.data, copied_params.size,
												copied_body));
			destroy_array(AST_Node_Ptr)(&copied_params);
		}
	} break;

	case AST_literal: {
		CASTED_NODE(AST_Literal, literal, node);
		ret = AST_BASE(copy_literal_node(literal));
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, node);
		ret = AST_BASE(copy_biop_node(	biop,
										copy_excluding_types_and_funcs_impl(biop->lhs, depth + 1),
										copy_excluding_types_and_funcs_impl(biop->rhs, depth + 1)));
	} break;

	case AST_control: {
		CASTED_NODE(AST_Control, control, node);
		ret = AST_BASE(copy_control_node(	control,
											copy_excluding_types_and_funcs_impl(control->value, depth + 1)));
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		AST_Node *copied_ident = copy_excluding_types_and_funcs_impl(AST_BASE(call->ident), depth + 1);
		Array(AST_Node_Ptr) copied_args = create_array(AST_Node_Ptr)(call->args.size);
		for (i = 0; i < call->args.size; ++i) {
			AST_Node *argcopy =
				copy_excluding_types_and_funcs_impl(call->args.data[i], depth + 1);
			push_array(AST_Node_Ptr)(&copied_args, argcopy);
		}
		ret = AST_BASE(copy_call_node(call, copied_ident, copied_args.data, copied_args.size));
		destroy_array(AST_Node_Ptr)(&copied_args);
	} break;

	default: FAIL(("copy_excluding_types_and_funcs: Unknown node type: %i", node->type));
	}

	return ret;
}

/* Creates copy of (partial) AST, dropping type and func decls */
INTERNAL AST_Node * copy_excluding_types_and_funcs(AST_Node *node)
{ return copy_excluding_types_and_funcs_impl(node, 0); }

/* Returns new AST */
INTERNAL AST_Scope *lift_types_and_funcs_to_global_scope(AST_Scope *root)
{
	AST_Scope *dst = create_ast_tree();
	int i, k;
	for (i = 0; i < root->nodes.size; ++i) {
		AST_Node *sub = root->nodes.data[i];
		Array(AST_Node_Ptr) decls = create_array(AST_Node_Ptr)(0);
		find_subnodes_of_type(&decls, AST_type_decl, sub);
		find_subnodes_of_type(&decls, AST_func_decl, sub);

		/* Lift */
		for (k = 0; k < decls.size; ++k) {
			/* @todo Rename the declarations to avoid name clashes */
			AST_Node *dst_decl = copy_excluding_types_and_funcs(decls.data[k]);
			if (!dst_decl)
				continue;
			push_array(AST_Node_Ptr)(&dst->nodes, dst_decl);
		}
		destroy_array(AST_Node_Ptr)(&decls);

		{
			AST_Node *copy = copy_excluding_types_and_funcs(sub);
			if (copy)
				push_array(AST_Node_Ptr)(&dst->nodes, copy);
		}
	}
	return dst;
}

INTERNAL void append_c_comment(Array(char) *buf, Token *comment)
{
	if (comment->type == Token_line_comment)
		append_str(buf, "/*%.*s */", TOK_ARGS(comment));
	else
		append_str(buf, "/*%.*s*/", TOK_ARGS(comment));
}

/* Almost 1-1 mapping between nodes and C constructs */
INTERNAL void ast_to_c_str(Array(char) *buf, int indent, AST_Node *node)
{
	int i, k;

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
			ast_to_c_str(buf, new_indent, sub);

			if (sub->type != AST_func_decl)
				append_str(buf, ";");

			for (k = 0; k < sub->post_comments.size; ++k) {
				append_str(buf, " ");
				append_c_comment(buf, sub->post_comments.data[k]);
			}

			append_str(buf, "\n");
		}
		if (!scope->is_root)
			append_str(buf, "%*s}", indent, "");
	} break;

	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, node);
		append_str(buf, "%.*s", TOK_ARGS(ident));
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		append_str(buf, "struct ");
		append_str(buf, "%.*s\n", TOK_ARGS(decl->ident));
		ast_to_c_str(buf, indent, AST_BASE(decl->body));
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		ast_to_c_str(buf, indent, decl->type);
		append_str(buf, " ");
		for (i = 0; i < decl->ptr_depth; ++i)
			append_str(buf, "*");
		append_str(buf, "%.*s", TOK_ARGS(decl->ident));
		if (decl->value) {
			append_str(buf, " = ");
			ast_to_c_str(buf, indent, decl->value);
		}
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		ast_to_c_str(buf, indent, decl->return_type);
		append_str(buf, " ");
		append_str(buf, "%.*s", TOK_ARGS(decl->ident));
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
}

Array(char) gen_c_code(AST_Scope *root)
{
	Array(char) buf = create_array(char)(1024);
	AST_Scope *c_ast = lift_types_and_funcs_to_global_scope(root);
	ast_to_c_str(&buf, 0, AST_BASE(c_ast));
	destroy_ast_tree(c_ast);
	return buf;
}

