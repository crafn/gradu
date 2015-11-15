#include "ast.h"

DEFINE_ARRAY(AST_Node_Ptr)
DEFINE_ARRAY(AST_Var_Decl_Ptr)
DEFINE_ARRAY(Token_Ptr)

INTERNAL AST_Node *create_node_impl(AST_Node_Type type, int size)
{
	AST_Node *n = calloc(1, size);
	n->type = type;
	n->pre_comments = create_array(Token_Ptr)(0);
	n->post_comments = create_array(Token_Ptr)(0);
	return n;
}
#define CREATE_NODE(type, type_enum) ((type*)create_node_impl(type_enum, sizeof(type)))

AST_Node *create_ast_node(AST_Node_Type type)
{
	switch (type) {
		case AST_scope: return AST_BASE(create_scope_node());
		case AST_ident: return AST_BASE(create_ident_node());
		case AST_type: return AST_BASE(create_type_node());
		case AST_type_decl: return AST_BASE(create_type_decl_node());
		case AST_var_decl: return AST_BASE(create_var_decl_node());
		case AST_func_decl: return AST_BASE(create_func_decl_node());
		case AST_literal: return AST_BASE(create_literal_node());
		case AST_biop: return AST_BASE(create_biop_node());
		case AST_control: return AST_BASE(create_control_node());
		case AST_call: return AST_BASE(create_call_node());
		case AST_access: return AST_BASE(create_access_node());
		default: FAIL(("create_ast_node: Unknown node type %i", type));
	}
}

AST_Scope *create_scope_node()
{
	AST_Scope *scope = CREATE_NODE(AST_Scope, AST_scope);
	scope->nodes = create_array(AST_Node_Ptr)(8);
	return scope;
}

AST_Ident *create_ident_node()
{
	AST_Ident * ident = CREATE_NODE(AST_Ident, AST_ident);
	ident->text = create_array(char)(1);
	append_str(&ident->text, "");
	return ident;
}

AST_Type *create_type_node()
{ return CREATE_NODE(AST_Type, AST_type); }

AST_Type_Decl *create_type_decl_node()
{ return CREATE_NODE(AST_Type_Decl, AST_type_decl); }

AST_Var_Decl *create_var_decl_node()
{ return CREATE_NODE(AST_Var_Decl, AST_var_decl); }

AST_Func_Decl *create_func_decl_node()
{ return CREATE_NODE(AST_Func_Decl, AST_func_decl); }

AST_Literal *create_literal_node()
{ return CREATE_NODE(AST_Literal, AST_literal); }

AST_Biop *create_biop_node()
{ return CREATE_NODE(AST_Biop, AST_biop); }

AST_Control *create_control_node()
{ return CREATE_NODE(AST_Control, AST_control); }

AST_Call *create_call_node()
{
	AST_Call *call = CREATE_NODE(AST_Call, AST_call);
	call->args = create_array(AST_Node_Ptr)(0);
	return call;
}

AST_Access *create_access_node()
{ return CREATE_NODE(AST_Access, AST_access); }


/* Node copying */

void copy_ast_node_base(AST_Node *dst, AST_Node *src)
{
	int i;
	dst->type = src->type;
	dst->begin_tok = src->begin_tok;
	for (i = 0; i < src->pre_comments.size; ++i)
		push_array(Token_Ptr)(&dst->pre_comments, src->pre_comments.data[i]);
	for (i = 0; i < src->post_comments.size; ++i)
		push_array(Token_Ptr)(&dst->post_comments, src->post_comments.data[i]);
}

void copy_ast_node(AST_Node *copy, AST_Node *node, AST_Node **subnodes, int subnode_count, AST_Node **refnodes, int refnode_count)
{
	ASSERT(copy->type == node->type);
	switch (node->type) {
		case AST_scope:
			ASSERT(refnode_count == 0);
			copy_scope_node((AST_Scope*)copy, (AST_Scope*)node, subnodes, subnode_count);
		break;
		case AST_ident: {
			ASSERT(subnode_count == 0 && refnode_count == 1);
			copy_ident_node((AST_Ident*)copy, (AST_Ident*)node, refnodes[0]);
		} break;
		case AST_type: {
			ASSERT(subnode_count == 0 && refnode_count == 1);
			copy_type_node((AST_Type*)copy, (AST_Type*)node, refnodes[0]);
		} break;
		case AST_type_decl: {
			ASSERT(subnode_count == 2 && refnode_count == 0);
			copy_type_decl_node((AST_Type_Decl*)copy, (AST_Type_Decl*)node, subnodes[0], subnodes[1]);
		} break;
		case AST_var_decl: {
			ASSERT(subnode_count == 3 && refnode_count == 0);
			copy_var_decl_node((AST_Var_Decl*)copy, (AST_Var_Decl*)node, subnodes[0], subnodes[1], subnodes[2]);
		} break;
		case AST_func_decl: {
			ASSERT(subnode_count >= 3 && refnode_count == 0);
			copy_func_decl_node((AST_Func_Decl*)copy, (AST_Func_Decl*)node, subnodes[0], subnodes[1], subnodes[2], &subnodes[3], subnode_count - 3);
		} break;
		case AST_literal: {
			ASSERT(subnode_count == 0 && refnode_count == 0);
			copy_literal_node((AST_Literal*)copy, (AST_Literal*)node);
		} break;
		case AST_biop: {
			ASSERT(subnode_count == 2 && refnode_count == 0);
			copy_biop_node((AST_Biop*)copy, (AST_Biop*)node, subnodes[0], subnodes[1]);
		} break;
		case AST_control: {
			ASSERT(subnode_count == 1 && refnode_count == 0);
			copy_control_node((AST_Control*)copy, (AST_Control*)node, subnodes[0]);
		} break;
		case AST_call: {
			ASSERT(subnode_count >= 1 && refnode_count == 0);
			copy_call_node((AST_Call*)copy, (AST_Call*)node, subnodes[0], &subnodes[1], subnode_count - 1);
		} break;
		case AST_access: {
			ASSERT(subnode_count >= 2 && refnode_count == 0);
			copy_access_node((AST_Access*)copy, (AST_Access*)node, subnodes[0], subnodes[1]);
		} break;
		default: FAIL(("copy_ast_node: Unknown node type %i", node->type));
	}
}

void shallow_copy_ast_node(AST_Node *copy, AST_Node* node)
{
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	Array(AST_Node_Ptr) refnodes = create_array(AST_Node_Ptr)(0);

	push_immediate_subnodes(&subnodes, node);
	push_immediate_refnodes(&refnodes, node);
	copy_ast_node(copy, node, subnodes.data, subnodes.size, refnodes.data, refnodes.size);

	destroy_array(AST_Node_Ptr)(&subnodes);
	destroy_array(AST_Node_Ptr)(&refnodes);
}

void copy_scope_node(AST_Scope *copy, AST_Scope *scope, AST_Node **subnodes, int subnode_count)
{
	int i;
	copy_ast_node_base(AST_BASE(copy), AST_BASE(scope));

	clear_array(AST_Node_Ptr)(&copy->nodes);
	for (i = 0; i < subnode_count; ++i) {
		if (!subnodes[i])
			continue;
		push_array(AST_Node_Ptr)(&copy->nodes, subnodes[i]);
	}
	copy->is_root = scope->is_root;
}

void copy_ident_node(AST_Ident *copy, AST_Ident *ident, AST_Node *ref_to_decl)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(ident));
	if (copy != ident) {
		destroy_array(char)(&copy->text);
		copy->text = copy_array(char)(&ident->text);
	}
	copy->decl = ref_to_decl;
}

void copy_type_node(AST_Type *copy, AST_Type *type, AST_Node *ref_to_base_type_decl)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(type));
	ASSERT(ref_to_base_type_decl->type == AST_type_decl);
	copy->base_type_decl = (AST_Type_Decl*)ref_to_base_type_decl;
	copy->ptr_depth = type->ptr_depth;
}

void copy_type_decl_node(AST_Type_Decl *copy, AST_Type_Decl *decl, AST_Node *ident, AST_Node *body)
{
	ASSERT(!ident || ident->type == AST_ident);
	ASSERT(!body || body->type == AST_scope);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->ident = (AST_Ident*)ident;
	copy->body = (AST_Scope*)body;
	copy->is_builtin = decl->is_builtin;
	copy->builtin_type = decl->builtin_type;
}

void copy_var_decl_node(AST_Var_Decl *copy, AST_Var_Decl *decl, AST_Node *type, AST_Node *ident, AST_Node *value)
{
	ASSERT(type->type == AST_type);
	ASSERT(ident->type == AST_ident);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->type = (AST_Type*)type;
	copy->ident = (AST_Ident*)ident;
	copy->value = value;
}

void copy_func_decl_node(AST_Func_Decl *copy, AST_Func_Decl *decl, AST_Node *return_type, AST_Node *ident, AST_Node *body, AST_Node **params, int param_count)
{
	int i;
	ASSERT(ident->type == AST_ident);
	ASSERT(return_type->type == AST_type);
	ASSERT(!body || body->type == AST_scope);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->return_type = (AST_Type*)return_type;
	copy->ident = (AST_Ident*)ident;
	copy->body = (AST_Scope*)body;

	clear_array(AST_Var_Decl_Ptr)(&copy->params);
	for (i = 0; i < param_count; ++i) {
		ASSERT(params[i]->type == AST_var_decl);
		push_array(AST_Var_Decl_Ptr)(&copy->params, (AST_Var_Decl_Ptr)params[i]);
	}
}

void copy_literal_node(AST_Literal *copy, AST_Literal *literal)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(literal));
	*copy = *literal;
}

void copy_biop_node(AST_Biop *copy, AST_Biop *biop, AST_Node *lhs, AST_Node *rhs)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(biop));
	copy->type = biop->type;
	copy->lhs = lhs;
	copy->rhs = rhs;
}

void copy_control_node(AST_Control *copy, AST_Control *control, AST_Node *value)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(control));
	copy->type = control->type;
	copy->value = value;
}

void copy_call_node(AST_Call *copy, AST_Call *call, AST_Node *ident, AST_Node **args, int arg_count)
{
	int i;
	ASSERT(ident->type == AST_ident);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(call));
	copy->ident = (AST_Ident*)ident;
	clear_array(AST_Node_Ptr)(&copy->args);
	for (i = 0; i < arg_count; ++i) {
		push_array(AST_Node_Ptr)(&copy->args, args[i]);
	}
}

void copy_access_node(AST_Access *copy, AST_Access *access, AST_Node *base, AST_Node *sub)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(access));
	copy->base = base;
	copy->sub = sub;
	copy->is_member_access = access->is_member_access;
	copy->is_array_access = access->is_array_access;
}

void destroy_node(AST_Node *node)
{
	int i;
	if (!node)
		return;

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		for (i = 0; i < scope->nodes.size; ++i)
			destroy_node(scope->nodes.data[i]);
	} break;

	case AST_ident: {
	} break;

	case AST_type: {
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		destroy_node(AST_BASE(decl->ident));
		destroy_node(AST_BASE(decl->body));
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		destroy_node(AST_BASE(decl->type));
		destroy_node(AST_BASE(decl->ident));
		destroy_node(decl->value);
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		destroy_node(AST_BASE(decl->return_type));
		destroy_node(AST_BASE(decl->ident));
		for (i = 0; i < decl->params.size; ++i)
			destroy_node(AST_BASE(decl->params.data[i]));
		destroy_node(AST_BASE(decl->body));
	} break;

	case AST_literal: {
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, op, node);
		destroy_node(op->lhs);
		destroy_node(op->rhs);
	} break;

	case AST_control: {
		CASTED_NODE(AST_Control, control, node);
		destroy_node(control->value);
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		destroy_node(AST_BASE(call->ident));
		for (i = 0; i < call->args.size; ++i)
			destroy_node(call->args.data[i]);
	} break;

	case AST_access: {
		CASTED_NODE(AST_Access, access, node);
		destroy_node(access->base);
		destroy_node(access->sub);
	} break;
	
	default: FAIL(("destroy_node: Unknown node type %i", node->type));
	}
	shallow_destroy_node(node);
}

void shallow_destroy_node(AST_Node *node)
{
	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		destroy_array(AST_Node_Ptr)(&scope->nodes);
	} break;

	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, node);
		destroy_array(char)(&ident->text);
	} break;

	case AST_type: {
	} break;

	case AST_type_decl: {
	} break;

	case AST_var_decl: {
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		destroy_array(AST_Var_Decl_Ptr)(&decl->params);
	} break;

	case AST_literal: {
	} break;

	case AST_biop: {
	} break;

	case AST_control: {
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		destroy_array(AST_Node_Ptr)(&call->args);
	} break;

	case AST_access: {
	} break;

	default: FAIL(("shallow_destroy_node: Unknown node type %i", node->type));
	};

	destroy_array(Token_Ptr)(&node->pre_comments);
	destroy_array(Token_Ptr)(&node->post_comments);
	free(node);
}

bool expr_type(AST_Type *ret, AST_Node *expr)
{
	bool success = false;
	memset(ret, 0, sizeof(*ret));

	switch (expr->type) {
	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, expr);
		ASSERT(ident->decl->type == AST_var_decl);
		{
			CASTED_NODE(AST_Var_Decl, decl, ident->decl);
			*ret = *decl->type;
			success = true;
		}
	} break;

	case AST_literal: {
		/* @todo */
	} break;

	case AST_access: {
		CASTED_NODE(AST_Access, access, expr);
		if (access->is_member_access) {
			success = expr_type(ret, access->sub);
		} else if (access->is_array_access) {
			success = expr_type(ret, access->base);
			--ret->ptr_depth;
		} else {
			success = expr_type(ret, access->base);
		}
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, expr);
		/* @todo Operation can yield different types than either of operands (2x1 * 1x2 matrices for example) */
		success = expr_type(ret, biop->lhs);
	} break;

	default: FAIL(("expr_type: Unknown node type %i", expr->type));
	}

	return success;
}

AST_Scope *create_ast_tree()
{
	AST_Scope *root = create_scope_node();
	root->is_root = true;
	return root;
}

void destroy_ast_tree(AST_Scope *node)
{ destroy_node((AST_Node*)node); }

INTERNAL void print_indent(int indent)
{ printf("%*s", indent, ""); }

void push_immediate_subnodes(Array(AST_Node_Ptr) *ret, AST_Node *node)
{
	int i;
	if (!node)
		return;

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		for (i = 0; i < scope->nodes.size; ++i)
			push_array(AST_Node_Ptr)(ret, scope->nodes.data[i]);
	} break;

	case AST_ident: {
	} break;

	case AST_type: {
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->ident));
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->body));
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->type));
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->ident));
		push_array(AST_Node_Ptr)(ret, decl->value);
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->return_type));
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->ident));
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->body));
		for (i = 0; i < decl->params.size; ++i)
			push_array(AST_Node_Ptr)(ret, AST_BASE(decl->params.data[i]));
	} break;

	case AST_literal: {
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, node);
		push_array(AST_Node_Ptr)(ret, biop->lhs);
		push_array(AST_Node_Ptr)(ret, biop->rhs);
	} break;

	case AST_control: {
		CASTED_NODE(AST_Control, control, node);
		push_array(AST_Node_Ptr)(ret, control->value);
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(call->ident));
		for (i = 0; i < call->args.size; ++i)
			push_array(AST_Node_Ptr)(ret, call->args.data[i]);
	} break;

	case AST_access: {
		CASTED_NODE(AST_Access, access, node);
		push_array(AST_Node_Ptr)(ret, access->base);
		push_array(AST_Node_Ptr)(ret, access->sub);
	} break;

	default: FAIL(("push_immediate_subnodes: Unknown node type: %i", node->type));
	}
}

void push_immediate_refnodes(Array(AST_Node_Ptr) *ret, AST_Node *node)
{
	if (!node)
		return;

	switch (node->type) {
	case AST_scope: break;

	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, node);
		push_array(AST_Node_Ptr)(ret, ident->decl);
	} break;

	case AST_type: {
		CASTED_NODE(AST_Type, type, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(type->base_type_decl));
	} break;

	case AST_type_decl: break;
	case AST_var_decl: break;
	case AST_func_decl: break;
	case AST_literal: break;
	case AST_biop: break;
	case AST_control: break;
	case AST_call: break;
	case AST_access: break;

	default: FAIL(("push_immediate_refnodes: Unknown node type: %i", node->type));
	}
}

void push_subnodes(Array(AST_Node_Ptr) *ret, AST_Node *node, bool push_before_recursing)
{
	int i;
	Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
	push_immediate_subnodes(&subnodes, node);

	for (i = 0; i < subnodes.size; ++i) {
		if (!subnodes.data[i])
			continue;

		if (push_before_recursing)
			push_array(AST_Node_Ptr)(ret, subnodes.data[i]);

		push_subnodes(ret, subnodes.data[i], push_before_recursing);

		if (!push_before_recursing)
			push_array(AST_Node_Ptr)(ret, subnodes.data[i]);
	}

	destroy_array(AST_Node_Ptr)(&subnodes);
}

AST_Node *replace_nodes_in_ast(AST_Node *node, AST_Node **old_nodes, AST_Node **new_nodes, int node_count)
{
	int i, k;

	if (!node || node_count == 0)
		return node;

	/* @todo Use Hash_Table to eliminate O(n^2) */
	/* Replacing happens before recursing, so that old_nodes in contained new_nodes are also replaced */
	for (i = 0; i < node_count; ++i) {
		if (node == old_nodes[i]) {
			node = new_nodes[i];
			break;
		}
	}

	{
		/* @todo Do something for the massive number of allocations */
		Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
		Array(AST_Node_Ptr) refnodes = create_array(AST_Node_Ptr)(0);

		push_immediate_subnodes(&subnodes, node);
		push_immediate_refnodes(&refnodes, node);

		/* Replace subnodes */
		for (i = 0; i < subnodes.size; ++i) {
			subnodes.data[i] = replace_nodes_in_ast(subnodes.data[i], old_nodes, new_nodes, node_count);
		}
		/* Replace referenced nodes */
		for (i = 0; i < refnodes.size; ++i) {
			for (k = 0; k < node_count; ++k) {
				if (refnodes.data[i] == old_nodes[k]) {
					refnodes.data[i] = new_nodes[k];
					break;
				}
			}
		}

		/* Update replaced pointers to node */
		copy_ast_node(	node, node,
						subnodes.data, subnodes.size,
						refnodes.data, refnodes.size);

		destroy_array(AST_Node_Ptr)(&subnodes);
		destroy_array(AST_Node_Ptr)(&refnodes);
	}

	return node;
}

void print_ast(AST_Node *node, int indent)
{
	int i;
	if (!node)
		return;

	print_indent(indent);

	switch (node->type) {
	case AST_scope: {
		CASTED_NODE(AST_Scope, scope, node);
		printf("scope %i\n", scope->nodes.size);
		for (i = 0; i < scope->nodes.size; ++i)
			print_ast(scope->nodes.data[i], indent + 2);
	} break;

	case AST_ident: {
		CASTED_NODE(AST_Ident, ident, node);
		printf("ident: %s\n", ident->text.data);
	} break;

	case AST_type: {
		CASTED_NODE(AST_Type, type, node);
		if (type->base_type_decl->is_builtin)
			printf("builtin_type\n");
		else
			printf("type %s %i\n", type->base_type_decl->ident->text.data, type->ptr_depth);
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		if (decl->is_builtin)
			printf("builtin_type_decl\n");
		else
			printf("type_decl\n");
		print_ast(AST_BASE(decl->ident), indent + 2);
		print_ast(AST_BASE(decl->body), indent + 2);
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		printf("var_decl\n");
		print_ast(AST_BASE(decl->type), indent + 2);
		print_ast(AST_BASE(decl->ident), indent + 2);
		print_ast(decl->value, indent + 2);
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		printf("func_decl\n");
		print_ast(AST_BASE(decl->return_type), indent + 2);
		print_ast(AST_BASE(decl->ident), indent + 2);
		print_ast(AST_BASE(decl->body), indent + 2);
	} break;

	case AST_literal: {
		CASTED_NODE(AST_Literal, literal, node);
		printf("literal: ");
		switch (literal->type) {
			case Literal_int: printf("%i\n", literal->value.integer); break;
			case Literal_string: printf("%.*s\n", literal->value.string.len, literal->value.string.buf); break;
			default: FAIL(("Unknown literal type: %i", literal->type));
		}
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, op, node);
		printf("biop %s\n", tokentype_str(op->type));
		print_ast(op->lhs, indent + 2);
		print_ast(op->rhs, indent + 2);
	} break;

	case AST_control: {
		CASTED_NODE(AST_Control, control, node);
		printf("control %s\n", tokentype_str(control->type));
		print_ast(control->value, indent + 2);
	} break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		printf("call\n");
		print_ast(AST_BASE(call->ident), indent + 2);
		for (i = 0; i < call->args.size; ++i)
			print_ast(call->args.data[i], indent + 2);
	} break;

	case AST_access: {
		CASTED_NODE(AST_Access, access, node);
		printf("access\n");
		print_ast(access->base, indent + 2);
		print_ast(access->sub, indent + 2);
	} break;

	default: FAIL(("print_ast: Unknown node type %i", node->type));
	};
}

