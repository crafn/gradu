#include "ast.h"

DEFINE_ARRAY(AST_Node_Ptr)
DEFINE_ARRAY(AST_Var_Decl_Ptr)
DEFINE_ARRAY(Token_Ptr)
DEFINE_HASH_TABLE(AST_Node_Ptr, AST_Node_Ptr)

bool type_node_equals(AST_Type a, AST_Type b)
{
	if (	a.ptr_depth != b.ptr_depth ||
			a.array_size != b.array_size ||
			a.is_const != b.is_const ||
			a.base_type_decl != b.base_type_decl)
		return false;
	return true;
}

bool builtin_type_equals(Builtin_Type a, Builtin_Type b)
{
	bool is_same_matrix = (a.is_matrix == b.is_matrix);
	bool is_same_field = (a.is_field == b.is_field);
	if (is_same_matrix && a.is_matrix)
		is_same_matrix = !memcmp(a.matrix_dim, b.matrix_dim, sizeof(a.matrix_dim));
	if (is_same_field && a.is_field)
		is_same_field = (a.field_dim == b.field_dim);

	return	a.is_void == b.is_void &&
			a.is_integer == b.is_integer &&
			a.is_float == b.is_float &&
			a.bitness == b.bitness &&
			a.is_unsigned == b.is_unsigned &&
			is_same_matrix &&
			is_same_field;
}

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
		case AST_cond: return AST_BASE(create_cond_node());
		case AST_loop: return AST_BASE(create_loop_node());
		case AST_cast: return AST_BASE(create_cast_node());
		case AST_typedef: return AST_BASE(create_typedef_node());
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
{
	AST_Access *access = CREATE_NODE(AST_Access, AST_access);
	access->args = create_array(AST_Node_Ptr)(1);
	return access;
}

AST_Cond *create_cond_node()
{ return CREATE_NODE(AST_Cond, AST_cond); }

AST_Loop *create_loop_node()
{ return CREATE_NODE(AST_Loop, AST_loop); }

AST_Cast *create_cast_node()
{ return CREATE_NODE(AST_Cast, AST_cast); }

AST_Typedef *create_typedef_node()
{ return CREATE_NODE(AST_Typedef, AST_typedef); }


/* Node copying */

void copy_ast_node_base(AST_Node *dst, AST_Node *src)
{
	int i;
	if (dst == src)
		return;
	/* Type of a node can not be changed */
	/*dst->type = src->type;*/
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
			ASSERT(subnode_count == 0 && refnode_count == 2);
			copy_type_node((AST_Type*)copy, (AST_Type*)node, refnodes[0], refnodes[1]);
		} break;

		case AST_type_decl: {
			ASSERT(subnode_count == 2 && refnode_count == 2);
			copy_type_decl_node((AST_Type_Decl*)copy, (AST_Type_Decl*)node, subnodes[0], subnodes[1], refnodes[0], refnodes[1]);
		} break;

		case AST_var_decl: {
			ASSERT(subnode_count == 3 && refnode_count == 0);
			copy_var_decl_node((AST_Var_Decl*)copy, (AST_Var_Decl*)node, subnodes[0], subnodes[1], subnodes[2]);
		} break;

		case AST_func_decl: {
			ASSERT(subnode_count >= 3 && refnode_count == 1);
			copy_func_decl_node((AST_Func_Decl*)copy, (AST_Func_Decl*)node, subnodes[0], subnodes[1], subnodes[2], &subnodes[3], subnode_count - 3, refnodes[0]);
		} break;

		case AST_literal: {
			ASSERT(subnode_count == 0 && refnode_count == 1);
			copy_literal_node((AST_Literal*)copy, (AST_Literal*)node, refnodes[0]);
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
			ASSERT(subnode_count >= 1 && refnode_count == 0);
			copy_access_node((AST_Access*)copy, (AST_Access*)node, subnodes[0], &subnodes[1], subnode_count - 1);
		} break;

		case AST_cond: {
			ASSERT(subnode_count == 3 && refnode_count == 0);
			copy_cond_node((AST_Cond*)copy, (AST_Cond*)node, subnodes[0], subnodes[1], subnodes[2]);
		} break;

		case AST_loop: {
			ASSERT(subnode_count == 4 && refnode_count == 0);
			copy_loop_node((AST_Loop*)copy, (AST_Loop*)node, subnodes[0], subnodes[1], subnodes[2], subnodes[3]);
		} break;

		case AST_cast: {
			ASSERT(subnode_count == 2 && refnode_count == 0);
			copy_cast_node((AST_Cast*)copy, (AST_Cast*)node, subnodes[0], subnodes[1]);
		} break;

		case AST_typedef: {
			ASSERT(subnode_count == 2 && refnode_count == 0);
			copy_typedef_node((AST_Typedef*)copy, (AST_Typedef*)node, subnodes[0], subnodes[1]);
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

void copy_type_node(AST_Type *copy, AST_Type *type, AST_Node *ref_to_base_type_decl, AST_Node *ref_to_base_typedef)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(type));
	ASSERT(!ref_to_base_type_decl || ref_to_base_type_decl->type == AST_type_decl);
	ASSERT(!ref_to_base_typedef || ref_to_base_typedef->type == AST_typedef);
	copy->base_type_decl = (AST_Type_Decl*)ref_to_base_type_decl;
	copy->base_typedef = (AST_Typedef*)ref_to_base_typedef;
	copy->ptr_depth = type->ptr_depth;
	copy->array_size = type->array_size;
	copy->is_const = type->is_const;
}

void copy_type_decl_node(AST_Type_Decl *copy, AST_Type_Decl *decl, AST_Node *ident, AST_Node *body, AST_Node *builtin_sub_decl_ref, AST_Node *builtin_decl_ref)
{
	ASSERT(!ident || ident->type == AST_ident);
	ASSERT(!body || body->type == AST_scope);
	ASSERT(!builtin_decl_ref || builtin_decl_ref->type == AST_type_decl);
	ASSERT(!builtin_sub_decl_ref || builtin_sub_decl_ref->type == AST_type_decl);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->ident = (AST_Ident*)ident;
	copy->body = (AST_Scope*)body;
	copy->is_builtin = decl->is_builtin;
	copy->builtin_type = decl->builtin_type;
	copy->builtin_sub_type_decl = (AST_Type_Decl*)builtin_sub_decl_ref;
	copy->builtin_concrete_decl = (AST_Type_Decl*)builtin_decl_ref;
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

void copy_func_decl_node(AST_Func_Decl *copy, AST_Func_Decl *decl, AST_Node *return_type, AST_Node *ident, AST_Node *body, AST_Node **params, int param_count, AST_Node *backend_decl_ref)
{
	int i;
	ASSERT(ident->type == AST_ident);
	ASSERT(return_type->type == AST_type);
	ASSERT(!body || body->type == AST_scope);
	ASSERT(!backend_decl_ref || backend_decl_ref->type == AST_func_decl);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->return_type = (AST_Type*)return_type;
	copy->ident = (AST_Ident*)ident;
	copy->body = (AST_Scope*)body;
	copy->ellipsis = decl->ellipsis;
	copy->is_builtin = decl->is_builtin;
	copy->builtin_concrete_decl = (AST_Func_Decl*)backend_decl_ref;

	clear_array(AST_Var_Decl_Ptr)(&copy->params);
	for (i = 0; i < param_count; ++i) {
		ASSERT(params[i]->type == AST_var_decl);
		push_array(AST_Var_Decl_Ptr)(&copy->params, (AST_Var_Decl_Ptr)params[i]);
	}
}

void copy_literal_node(AST_Literal *copy, AST_Literal *literal, AST_Node *type_decl_ref)
{
	ASSERT(!type_decl_ref || type_decl_ref->type == AST_type_decl);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(literal));
	copy->type = literal->type;
	copy->value = literal->value;
	copy->base_type_decl = (AST_Type_Decl*)type_decl_ref;
}

void copy_biop_node(AST_Biop *copy, AST_Biop *biop, AST_Node *lhs, AST_Node *rhs)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(biop));
	copy->type = biop->type;
	copy->is_top_level = biop->is_top_level;
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

void copy_access_node(AST_Access *copy, AST_Access *access, AST_Node *base, AST_Node **args, int arg_count)
{
	int i;
	copy_ast_node_base(AST_BASE(copy), AST_BASE(access));
	copy->base = base;
	clear_array(AST_Node_Ptr)(&copy->args);
	for (i = 0; i < arg_count; ++i)
		push_array(AST_Node_Ptr)(&copy->args, args[i]);
	copy->is_member_access = access->is_member_access;
	copy->is_element_access = access->is_element_access;
	copy->is_array_access = access->is_array_access;
}

void copy_cond_node(AST_Cond *copy, AST_Cond *cond, AST_Node *expr, AST_Node *body, AST_Node *after_else)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(cond));
	ASSERT(!body || body->type == AST_scope);
	copy->expr = expr;
	copy->body = (AST_Scope*)body;
	copy->after_else = after_else;
	copy->implicit_scope = cond->implicit_scope;
}

void copy_loop_node(AST_Loop *copy, AST_Loop *loop, AST_Node *init, AST_Node *cond, AST_Node *incr, AST_Node *body)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(loop));
	ASSERT(!body || body->type == AST_scope);
	copy->init = init;
	copy->cond = cond;
	copy->incr = incr;
	copy->body = (AST_Scope*)body;
	copy->implicit_scope = loop->implicit_scope;
}

void copy_cast_node(AST_Cast *copy, AST_Cast *cast, AST_Node *type, AST_Node *target)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(cast));
	ASSERT(!type || type->type == AST_type);
	copy->type = (AST_Type*)type;
	copy->target = target;
}

void copy_typedef_node(AST_Typedef *copy, AST_Typedef *def, AST_Node *type, AST_Node *ident)
{
	copy_ast_node_base(AST_BASE(copy), AST_BASE(def));
	ASSERT(!type || type->type == AST_type);
	ASSERT(!ident || ident->type == AST_ident);
	copy->type = (AST_Type*)type;
	copy->ident = (AST_Ident*)ident;
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
		for (i = 0; i < access->args.size; ++i)
			destroy_node(access->args.data[i]);
	} break;

	case AST_cond: {
		CASTED_NODE(AST_Cond, cond, node);
		destroy_node(cond->expr);
		destroy_node(AST_BASE(cond->body));
		destroy_node(cond->after_else);
	} break;

	case AST_loop: {
		CASTED_NODE(AST_Loop, loop, node);
		destroy_node(loop->init);
		destroy_node(loop->cond);
		destroy_node(loop->incr);
		destroy_node(AST_BASE(loop->body));
	} break;

	case AST_cast: {
		CASTED_NODE(AST_Cast, cast, node);
		destroy_node(AST_BASE(cast->type));
		destroy_node(cast->target);
	} break;

	case AST_typedef: {
		CASTED_NODE(AST_Typedef, def, node);
		destroy_node(AST_BASE(def->type));
		destroy_node(AST_BASE(def->ident));
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

	case AST_type: break;
	case AST_type_decl: break;
	case AST_var_decl: break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		destroy_array(AST_Var_Decl_Ptr)(&decl->params);
	} break;

	case AST_literal: break;
	case AST_biop: break;
	case AST_control: break;

	case AST_call: {
		CASTED_NODE(AST_Call, call, node);
		destroy_array(AST_Node_Ptr)(&call->args);
	} break;

	case AST_access: {
		CASTED_NODE(AST_Access, access, node);
		destroy_array(AST_Node_Ptr)(&access->args);
	} break;

	case AST_cond: break;
	case AST_loop: break;
	case AST_cast: break;
	case AST_typedef: break;

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
		if (!ident->decl)
			break;
		ASSERT(ident->decl->type == AST_var_decl);
		{
			CASTED_NODE(AST_Var_Decl, decl, ident->decl);
			*ret = *decl->type;
			success = true;
		}
	} break;

	case AST_literal: {
		CASTED_NODE(AST_Literal, literal, expr);
		ret->base_type_decl = literal->base_type_decl;
		ASSERT(ret->base_type_decl);
		if (literal->type == Literal_string)
			++ret->ptr_depth;
		success = true;
	} break;

	case AST_access: {
		CASTED_NODE(AST_Access, access, expr);
		if (access->is_member_access) {
			ASSERT(access->args.size == 1);
			success = expr_type(ret, access->args.data[0]);
		} else if (access->is_element_access) {
			success = expr_type(ret, access->base);
			/* "Dereference" field -> matrix/scalar, matrix -> scalar */
			ret->base_type_decl = ret->base_type_decl->builtin_sub_type_decl;
		} else if (access->is_array_access) {
			success = expr_type(ret, access->base);
			--ret->ptr_depth;
		} else {
			/* Plain variable access */
			ASSERT(access->args.size == 0);
			success = expr_type(ret, access->base);
		}
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, expr);
		/* @todo Operation can yield different types than either of operands (2x1 * 1x2 matrices for example) */
		success = expr_type(ret, biop->rhs);
	} break;

	default:;
	}

	return success;
}

bool eval_const_expr(AST_Literal *ret, AST_Node *expr)
{
	bool success = false;
	memset(ret, 0, sizeof(*ret));

	switch (expr->type) {
	case AST_literal: {
		CASTED_NODE(AST_Literal, literal, expr);
		ret->type = literal->type;
		ret->value = literal->value;
		success = true;
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, biop, expr);
		/* @todo Operation can yield different types than either of operands (2x1 * 1x2 matrices for example) */
		if (biop->lhs && biop->rhs) {
			AST_Literal lhs, rhs;
			success = eval_const_expr(&lhs, biop->lhs);
			if (!success)
				break;
			success = eval_const_expr(&rhs, biop->rhs);
			if (!success)
				break;
			if (lhs.type != rhs.type) {
				success = false;
				break;
			}

			switch (biop->type) {
			case Token_add:
				switch (lhs.type) {
				case Literal_int: ret->value.integer = lhs.value.integer + rhs.value.integer; break;
				case Literal_float: ret->value.floating = lhs.value.floating + rhs.value.floating; break;
				default: FAIL(("Unhandled literal type %i", lhs.type));
				}
			break;
			case Token_sub:
				switch (lhs.type) {
				case Literal_int: ret->value.integer = lhs.value.integer - rhs.value.integer; break;
				case Literal_float: ret->value.floating = lhs.value.floating - rhs.value.floating; break;
				default: FAIL(("Unhandled literal type %i", lhs.type));
				}
			break;
			default: FAIL(("Unhandled biop type %i", biop->type));
			}
		} else {
			FAIL(("@todo unary const expr eval"));
		}
	} break;

	default:;
	}

	return success;
}

AST_Scope *create_ast()
{
	AST_Scope *root = create_scope_node();
	root->is_root = true;
	return root;
}

void destroy_ast(AST_Scope *node)
{ destroy_node((AST_Node*)node); }

typedef struct Copy_Ctx {
	Hash_Table(AST_Node_Ptr, AST_Node_Ptr) src_to_dst;
} Copy_Ctx;

AST_Node *copy_ast_impl(Copy_Ctx *ctx, AST_Node *node)
{
	/* @todo backend_c copy_excluding_types_and_funcs has almost identical code */
	if (node) {
		int i;
		AST_Node *copy = create_ast_node(node->type);

		/* @todo Do something for the massive number of allocations */
		Array(AST_Node_Ptr) subnodes = create_array(AST_Node_Ptr)(0);
		Array(AST_Node_Ptr) refnodes = create_array(AST_Node_Ptr)(0);

		set_tbl(AST_Node_Ptr, AST_Node_Ptr)(&ctx->src_to_dst, node, copy);

		push_immediate_subnodes(&subnodes, node);
		push_immediate_refnodes(&refnodes, node);

		for (i = 0; i < subnodes.size; ++i) {
			subnodes.data[i] = copy_ast_impl(ctx, subnodes.data[i]);
		}
		for (i = 0; i < refnodes.size; ++i) {
			AST_Node *remapped = get_tbl(AST_Node_Ptr, AST_Node_Ptr)(&ctx->src_to_dst, refnodes.data[i]);
			/* If referenced node is outside branch we're copying, don't change it */
			if (!remapped)
				remapped = refnodes.data[i]; 
			refnodes.data[i] = remapped;
		}

		copy_ast_node(	copy, node,
						subnodes.data, subnodes.size,
						refnodes.data, refnodes.size);

		destroy_array(AST_Node_Ptr)(&subnodes);
		destroy_array(AST_Node_Ptr)(&refnodes);

		return copy;
	}
	return NULL;
}

AST_Node *copy_ast(AST_Node *node)
{
	Copy_Ctx ctx = {{0}};
	AST_Node *ret;
	/* @todo Size should be something like TOTAL_NODE_COUNT*2 */
	ctx.src_to_dst = create_tbl(AST_Node_Ptr, AST_Node_Ptr)(NULL, NULL, 1024);
	ret = copy_ast_impl(&ctx, node);
	destroy_tbl(AST_Node_Ptr, AST_Node_Ptr)(&ctx.src_to_dst);
	return ret;
}

AST_Node *shallow_copy_ast(AST_Node *node)
{
	AST_Node *copy = create_ast_node(node->type);
	shallow_copy_ast_node(copy, node);
	return copy;
}

void move_ast(AST_Scope *dst, AST_Scope *src)
{
	/* Substitute subnodes in dst with subnodes of src, and destroy src */
	int i;
	for (i = 0; i < dst->nodes.size; ++i)
		destroy_node(dst->nodes.data[i]);

	clear_array(AST_Node_Ptr)(&dst->nodes);
	insert_array(AST_Node_Ptr)(&dst->nodes, 0, src->nodes.data, src->nodes.size);

	/* Don't destroy subnodes because they were moved */
	shallow_destroy_node(AST_BASE(src));
}

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
		insert_array(AST_Node_Ptr)(ret, ret->size, access->args.data, access->args.size);
	} break;

	case AST_cond: {
		CASTED_NODE(AST_Cond, cond, node);
		push_array(AST_Node_Ptr)(ret, cond->expr);
		push_array(AST_Node_Ptr)(ret, AST_BASE(cond->body));
		push_array(AST_Node_Ptr)(ret, cond->after_else);
	} break;

	case AST_loop: {
		CASTED_NODE(AST_Loop, loop, node);
		push_array(AST_Node_Ptr)(ret, loop->init);
		push_array(AST_Node_Ptr)(ret, loop->cond);
		push_array(AST_Node_Ptr)(ret, loop->incr);
		push_array(AST_Node_Ptr)(ret, AST_BASE(loop->body));
	} break;

	case AST_cast: {
		CASTED_NODE(AST_Cast, cast, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(cast->type));
		push_array(AST_Node_Ptr)(ret, cast->target);
	} break;

	case AST_typedef: {
		CASTED_NODE(AST_Typedef, def, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(def->type));
		push_array(AST_Node_Ptr)(ret, AST_BASE(def->ident));
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
		push_array(AST_Node_Ptr)(ret, AST_BASE(type->base_typedef));
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->builtin_sub_type_decl));
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->builtin_concrete_decl));
	} break;

	case AST_var_decl: break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(decl->builtin_concrete_decl));
	} break;

	case AST_literal: {
		CASTED_NODE(AST_Literal, literal, node);
		push_array(AST_Node_Ptr)(ret, AST_BASE(literal->base_type_decl));
	} break;

	case AST_biop: break;
	case AST_control: break;
	case AST_call: break;
	case AST_access: break;
	case AST_cond: break;
	case AST_loop: break;
	case AST_cast: break;
	case AST_typedef: break;

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

void find_subnodes_of_type(Array(AST_Node_Ptr) *ret, AST_Node_Type type, AST_Node *node)
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
		if (type->base_type_decl->is_builtin) {
			printf("builtin_type\n");
		} else {
			printf("type %s %i\n", type->base_type_decl->ident->text.data, type->ptr_depth);
		}
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		if (decl->is_builtin) {
			Builtin_Type bt = decl->builtin_type;
			printf("builtin_type_decl: is_field: %i dim: %i  is_matrix: %i rank: %i  is_int: %i  is_float: %i\n",
					bt.is_field, bt.field_dim,
					bt.is_matrix, bt.matrix_rank,
					bt.is_integer, bt.is_float);
		} else {
			printf("type_decl\n");
		}
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
			case Literal_float: printf("%f\n", literal->value.floating); break;
			case Literal_string: printf("%.*s\n", literal->value.string.len, literal->value.string.buf); break;
			case Literal_null: printf("NULL\n"); break;
			default: FAIL(("Unknown literal type: %i", literal->type));
		}
	} break;

	case AST_biop: {
		CASTED_NODE(AST_Biop, op, node);
		if (op->lhs && op->rhs) {
			printf("biop %s\n", tokentype_str(op->type));
			print_ast(op->lhs, indent + 2);
			print_ast(op->rhs, indent + 2);
		} else {
			printf("uop %s\n", tokentype_str(op->type));
			print_ast(op->rhs, indent + 2);
		}
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
		for (i = 0; i < access->args.size; ++i)
			print_ast(access->args.data[i], indent + 2);
	} break;

	case AST_cond: {
		CASTED_NODE(AST_Cond, cond, node);
		printf("cond\n");
		print_ast(cond->expr, indent + 2);
		print_ast(AST_BASE(cond->body), indent + 2);
		print_ast(cond->after_else, indent + 2);
	} break;

	case AST_loop: {
		CASTED_NODE(AST_Loop, loop, node);
		printf("loop\n");
		print_ast(loop->init, indent + 2);
		print_ast(loop->cond, indent + 2);
		print_ast(loop->incr, indent + 2);
		print_ast(AST_BASE(loop->body), indent + 2);
	} break;

	case AST_cast: {
		CASTED_NODE(AST_Cast, cast, node);
		printf("cast\n");
		print_ast(AST_BASE(cast->type), indent + 2);
		print_ast(cast->target, indent + 2);
	} break;

	case AST_typedef: {
		CASTED_NODE(AST_Typedef, def, node);
		printf("typedef\n");
		print_ast(AST_BASE(def->type), indent + 2);
		print_ast(AST_BASE(def->ident), indent + 2);
	} break;

	default: FAIL(("print_ast: Unknown node type %i", node->type));
	};
}

AST_Ident *create_ident_with_text(const char *fmt, ...)
{
	AST_Ident *ident = create_ident_node();
	va_list args;
	va_start(args, fmt);
	safe_vsprintf(&ident->text, fmt, args);
	va_end(args);
	return ident;
}

AST_Var_Decl *create_simple_var_decl(AST_Type_Decl *type_decl, const char *ident)
{
	AST_Var_Decl *decl = create_var_decl_node();
	decl->type = create_type_node();
	decl->type->base_type_decl = type_decl;
	decl->ident = create_ident_with_text(ident);
	return decl;
}

AST_Type_Decl *find_builtin_type_decl(Builtin_Type bt, AST_Scope *root)
{
	int i;
	for (i = 0; i < root->nodes.size; ++i) {
		AST_Node *node = root->nodes.data[i];
		if (node->type != AST_type_decl)
			continue;
		{
			CASTED_NODE(AST_Type_Decl, type_decl, node);
			if (!type_decl->is_builtin)
				continue;

			if (builtin_type_equals(type_decl->builtin_type, bt))
				return type_decl;
		}
	}
	FAIL(("find_builtin_type_decl: Builtin type not found"));
	return NULL;
}

AST_Literal *create_integer_literal(int value)
{
	AST_Literal *literal = create_literal_node();
	literal->type = Literal_int;
	literal->value.integer = value;
	return literal;
}

AST_Call *create_call_1(AST_Ident *ident, AST_Node *arg)
{
	AST_Call *call = create_call_node();
	call->ident = ident;
	push_array(AST_Node_Ptr)(&call->args, arg);
	return call;
}

AST_Control *create_return(AST_Node *expr)
{
	AST_Control *ret = create_control_node();
	ret->type = Token_kw_return;
	ret->value = expr;
	return ret;
}

AST_Biop *create_sizeof(AST_Node *expr)
{
	AST_Biop *op = create_biop_node();
	op->type = Token_kw_sizeof;
	op->rhs = expr;
	return op;
}

AST_Biop *create_deref(AST_Node *expr)
{
	AST_Biop *op = create_biop_node();
	op->type = Token_mul;
	op->rhs = expr;
	return op;
}

AST_Biop *create_assign(AST_Node *lhs, AST_Node *rhs)
{
	AST_Biop *op = create_biop_node();
	op->type = Token_assign;
	op->lhs = lhs;
	op->rhs = rhs;
	return op;
}

AST_Biop *create_mul(AST_Node *lhs, AST_Node *rhs)
{
	AST_Biop *op = create_biop_node();
	op->type = Token_mul;
	op->lhs = lhs;
	op->rhs = rhs;
	return op;
}

AST_Cast *create_cast(AST_Type *type, AST_Node *target)
{
	AST_Cast *cast = create_cast_node();
	cast->type = type;
	cast->target = target;
	return cast;
}

AST_Type *create_builtin_type(Builtin_Type bt, int ptr_depth, AST_Scope *root)
{
	AST_Type *type = create_type_node();
	type->base_type_decl = find_builtin_type_decl(bt, root);;
	type->ptr_depth = ptr_depth;
	ASSERT(type->base_type_decl);
	return type;
}

AST_Type *copy_and_modify_type(AST_Type *type, int delta_ptr_depth)
{
	AST_Type *copy = (AST_Type*)copy_ast(AST_BASE(type));
	copy->ptr_depth += delta_ptr_depth;
	return copy;
}

Builtin_Type void_builtin_type()
{
	Builtin_Type bt = {0};
	bt.is_void = true;
	return bt;
}

Builtin_Type int_builtin_type()
{
	Builtin_Type bt = {0};
	bt.is_integer = true;
	return bt;
}

Builtin_Type float_builtin_type()
{
	Builtin_Type bt = {0};
	bt.is_float = true;
	bt.bitness = 32;
	return bt;
}

Builtin_Type char_builtin_type()
{
	Builtin_Type bt = {0};
	bt.is_char = true;
	return bt;
}

AST_Node *create_chained_expr(AST_Node **elems, int elem_count, Token_Type chainop)
{
	int i;
	AST_Node *ret = NULL;
	for (i = 0; i < elem_count; ++i) {
		if (!ret) {
			ret = elems[i];
		} else {
			AST_Biop *chain = create_biop_node();
			chain->type = chainop;
			chain->lhs = ret;
			chain->rhs = elems[i];

			ret = AST_BASE(chain);
		}
	}
	return ret;
}

AST_Node *create_chained_expr_2(AST_Node **lhs_elems, AST_Node **rhs_elems, int elem_count, Token_Type biop, Token_Type chainop)
{
	int i;
	AST_Node *ret = NULL;
	for (i = 0; i < elem_count; ++i) {
		AST_Biop *inner = create_biop_node();
		inner->type = biop;
		inner->lhs = lhs_elems[i];
		inner->rhs = rhs_elems[i];

		if (!ret) {
			ret = AST_BASE(inner);
		} else {
			AST_Biop *outer = create_biop_node();
			outer->type = chainop;
			outer->lhs = ret;
			outer->rhs = AST_BASE(inner);

			ret = AST_BASE(outer);
		}
	}
	return ret;
}
