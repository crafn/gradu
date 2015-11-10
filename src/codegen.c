#include "codegen.h"

INTERNAL void find_subnodes_of_type_impl(Array(AstNodePtr) *result, AstNodeType type, AstNode *node, int depth)
{
	/* @todo Create linear (inner, or outermost first) search for AST and use that */
	int i;
	switch (node->type) {
	case AstNodeType_scope: {
		CASTED_NODE(ScopeAstNode, scope, node);
		for (i = 0; i < scope->nodes.size; ++i)
			find_subnodes_of_type_impl(result, type, scope->nodes.data[i], depth + 1);
	} break;
	case AstNodeType_ident: {
	} break;
	case AstNodeType_decl: {
		CASTED_NODE(DeclAstNode, decl, node);
		if (decl->type)
			find_subnodes_of_type_impl(result, type, decl->type, depth + 1);
		if (decl->ident)
			find_subnodes_of_type_impl(result, type, AST_BASE(decl->ident), depth + 1);
		if (decl->value)
			find_subnodes_of_type_impl(result, type, decl->value, depth + 1);
	} break;
	case AstNodeType_literal: {
	} break;
	case AstNodeType_biop: {
		CASTED_NODE(BiopAstNode, biop, node);
		find_subnodes_of_type_impl(result, type, biop->lhs, depth + 1);
		find_subnodes_of_type_impl(result, type, biop->rhs, depth + 1);
	} break;
	default: FAIL(("find_subnodes_of_type: Unknown node type: %i", type));
	}

	if (depth > 0 && node->type == type)
		push_array(AstNodePtr)(result, node);
}

/* Innermost first */
INTERNAL Array(AstNodePtr) find_subnodes_of_type(AstNodeType type, AstNode *node)
{
	Array(AstNodePtr) decls = create_array(AstNodePtr)(0);
	find_subnodes_of_type_impl(&decls, type, node, 0);
	return decls;
}

INTERNAL AstNode * copy_excluding_types_and_funcs_impl(AstNode *node, int depth)
{
	int i;
	AstNode *ret = NULL;
	if (!node)
		return NULL;

	switch (node->type) {
	case AstNodeType_scope: {
		CASTED_NODE(ScopeAstNode, scope, node);
		Array(AstNodePtr) copied_subnodes = create_array(AstNodePtr)(0);
		for (i = 0; i < scope->nodes.size; ++i) {
			AstNode *subcopy = copy_excluding_types_and_funcs_impl(scope->nodes.data[i], depth + 1);
			if (subcopy)
				push_array(AstNodePtr)(&copied_subnodes, subcopy);
		}
		ret = AST_BASE(copy_scope_node(scope, copied_subnodes.data, copied_subnodes.size));
		destroy_array(AstNodePtr)(&copied_subnodes);
	} break;
	case AstNodeType_ident: {
		CASTED_NODE(IdentAstNode, ident, node);
		ret = AST_BASE(copy_ident_node(ident));
	} break;
	case AstNodeType_decl: {
		CASTED_NODE(DeclAstNode, decl, node);
		if (depth > 0 && (decl->is_type_decl || decl->is_func_decl))
			return NULL;
		{
			AstNode *copied_type = copy_excluding_types_and_funcs_impl(decl->type, depth + 1);
			AstNode *copied_ident = copy_excluding_types_and_funcs_impl(AST_BASE(decl->ident), depth + 1);
			AstNode *copied_value = copy_excluding_types_and_funcs_impl(decl->value, depth + 1);
			if (	!copied_type == !decl->type &&
					!copied_ident == !decl->ident &&
					!copied_value == !decl->value)
				ret = AST_BASE(copy_decl_node(decl, copied_type, copied_ident, copied_value));
		}
	} break;
	case AstNodeType_literal: {
		CASTED_NODE(LiteralAstNode, literal, node);
		ret = AST_BASE(copy_literal_node(literal));
	} break;
	case AstNodeType_biop: {
		CASTED_NODE(BiopAstNode, biop, node);
		ret = AST_BASE(copy_biop_node(	biop,
										copy_excluding_types_and_funcs_impl(biop->lhs, depth + 1),
										copy_excluding_types_and_funcs_impl(biop->rhs, depth + 1)));
	} break;
	default: FAIL(("copy_excluding_types_and_funcs: Unknown node type: %i", node->type));
	}

	return ret;
}

/* Creates copy of (partial) AST, dropping type and func decls */
INTERNAL AstNode * copy_excluding_types_and_funcs(AstNode *node)
{ return copy_excluding_types_and_funcs_impl(node, 0); }

/* Returns new AST */
INTERNAL ScopeAstNode *lift_types_and_funcs_to_global_scope(ScopeAstNode *root)
{
	ScopeAstNode *dst = create_ast_tree();
	int i, k;
	for (i = 0; i < root->nodes.size; ++i) {
		AstNode *sub = root->nodes.data[i];
		Array(AstNodePtr) decls = find_subnodes_of_type(AstNodeType_decl, sub);

		/* Lift */
		for (k = 0; k < decls.size; ++k) {
			CASTED_NODE(DeclAstNode, decl, decls.data[k]);
			AstNode *dst_decl;
			if (!decl->is_type_decl && !decl->is_func_decl)
				continue;
			/* @todo Rename the declarations to avoid name clashes */
			dst_decl = copy_excluding_types_and_funcs(AST_BASE(decl));
			if (!dst_decl)
				continue;
			push_array(AstNodePtr)(&dst->nodes, dst_decl);
		}
		destroy_array(AstNodePtr)(&decls);

		{
			AstNode *copy = copy_excluding_types_and_funcs(sub);
			if (copy)
				push_array(AstNodePtr)(&dst->nodes, copy);
		}
	}
	return dst;
}

INTERNAL bool is_func_decl(AstNode *node)
{
	if (node->type != AstNodeType_decl)
		return false;
	{
		CASTED_NODE(DeclAstNode, decl, node);
		return decl->is_func_decl;
	}
}

/* Almost 1-1 mapping between nodes and C constructs */
INTERNAL void ast_to_c_str(Array(char) *buf, int indent, AstNode *node)
{
	int i;

	switch (node->type) {
	case AstNodeType_scope: {
		CASTED_NODE(ScopeAstNode, scope, node);
		int new_indent = indent + 4;
		if (scope->is_root)
			new_indent = 0;

		if (!scope->is_root)
			append_str(buf, "%*s{\n", indent, "");
		for (i = 0; i < scope->nodes.size; ++i) {
			AstNode *sub = scope->nodes.data[i];
			if (sub->begin_tok && sub->begin_tok->empty_line_before)
				append_str(buf, "\n"); /* Retain some vertical spacing from original code */
			append_str(buf, "%*s", new_indent, "");
			ast_to_c_str(buf, new_indent, sub);

			if (!is_func_decl(sub))
				append_str(buf, ";");
			append_str(buf, "\n");
		}
		if (!scope->is_root)
			append_str(buf, "%*s}", indent, "");
	} break;
	case AstNodeType_ident: {
		CASTED_NODE(IdentAstNode, ident, node);
		append_str(buf, "%.*s", TOK_ARGS(ident));
	} break;
	case AstNodeType_decl: {
		CASTED_NODE(DeclAstNode, decl, node);
		if (decl->is_type_decl) {
			append_str(buf, "struct ");
		} else {
			ast_to_c_str(buf, indent, decl->type);
			append_str(buf, " ");
		}

		append_str(buf, "%.*s", TOK_ARGS(decl->ident));

		if (decl->is_type_decl) {
			append_str(buf, " { /* @todo */ }");
		} else if (decl->is_func_decl) {
			append_str(buf, "( /* @todo */ )\n");
			ast_to_c_str(buf, indent, decl->value);
		}
	} break;
	case AstNodeType_literal: {
		CASTED_NODE(LiteralAstNode, literal, node);
		/* @todo Other types */
		ASSERT(literal->type == LiteralType_int);
		append_str(buf, "%i", literal->value.integer);
	} break;
	case AstNodeType_biop: {
		CASTED_NODE(BiopAstNode, biop, node);
		ast_to_c_str(buf, indent, biop->lhs);
		append_str(buf, " %s ", tokentype_codestr(biop->type));
		ast_to_c_str(buf, indent, biop->rhs);
	} break;
	default:;
	}
}

Array(char) gen_c_code(ScopeAstNode *root)
{
	Array(char) buf = create_array(char)(1024);
	ScopeAstNode *c_ast = lift_types_and_funcs_to_global_scope(root);
	ast_to_c_str(&buf, 0, AST_BASE(c_ast));
	destroy_ast_tree(c_ast);
	return buf;
}

