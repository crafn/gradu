#include "parse.h"

int str_to_int(const char *c, int len)
{
	const char *end = c + len;
	int value = 0;
	int sign = 1;
	if (*c == '+' || *c == '-') {
		if(*c == '-')
			sign = -1;
		c++;
	}
	while (c < end) {
		value *= 10;
		value += (int)(*c - '0');
		c++;
	}
	return value * sign;
}

int op_prec(Token_Type type)
{
	switch (type) {
		case Token_assign: return 1;
		case Token_add: return 2;
		case Token_mul: return 3;
		default: return -1;
	}
}

/* -1 left, 1 right */
int op_assoc(Token_Type type)
{
	switch (type) {
		case Token_assign: return 1; /* a = b = c  <=>  (a = (b = c)) */
		case Token_add: return -1;
		case Token_mul: return -1;
		default: return -1;
	}
}

bool is_op(Token_Type type)
{ return op_prec(type) >= 0; }

INTERNAL AST_Node *create_node_impl(AST_Node_Type type, int size)
{
	AST_Node *n = calloc(1, size);
	n->type = type;
	n->pre_comments = create_array(Token_Ptr)(0);
	n->post_comments = create_array(Token_Ptr)(0);
	return n;
}
#define CREATE_NODE(type, type_enum) ((type*)create_node_impl(type_enum, sizeof(type)))

AST_Scope *create_scope_node()
{
	AST_Scope *scope = CREATE_NODE(AST_Scope, AST_scope);
	scope->nodes = create_array(AST_Node_Ptr)(8);
	return scope;
}

AST_Ident *create_ident_node(Token *tok)
{
	AST_Ident * ident = CREATE_NODE(AST_Ident, AST_ident);
	if (tok) {
		ident->text_buf = tok->text_buf;
		ident->text_len = tok->text_len;
	}
	return ident;
}

AST_Type_Decl *create_type_decl_node()
{ return CREATE_NODE(AST_Type_Decl, AST_type_decl); }

AST_Var_Decl *create_var_decl_node()
{ return CREATE_NODE(AST_Var_Decl, AST_var_decl); }

AST_Func_Decl *create_func_decl_node()
{ return CREATE_NODE(AST_Func_Decl, AST_func_decl); }

AST_Literal *create_literal_node()
{ return CREATE_NODE(AST_Literal, AST_literal); }

AST_Biop *create_biop_node(Token_Type type, AST_Node *lhs, AST_Node *rhs)
{
	AST_Biop *biop = CREATE_NODE(AST_Biop, AST_biop);
	biop->type = type;
	biop->lhs = lhs;
	biop->rhs = rhs;
	return biop;
}

AST_Control *create_control_node(Token_Type type)
{
	AST_Control *control = CREATE_NODE(AST_Control, AST_control);
	control->type = type;
	return control;
}

AST_Call *create_call_node()
{
	AST_Call *call = CREATE_NODE(AST_Call, AST_call);
	call->args = create_array(AST_Node_Ptr)(0);
	return call;
}


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

AST_Scope *copy_scope_node(AST_Scope *scope, AST_Node **subnodes, int subnode_count)
{
	AST_Scope *copy = create_scope_node();
	int i;
	copy_ast_node_base(AST_BASE(copy), AST_BASE(scope));
	for (i = 0; i < subnode_count; ++i)
		push_array(AST_Node_Ptr)(&copy->nodes, subnodes[i]);
	copy->is_root = scope->is_root;
	return copy;
}

AST_Ident *copy_ident_node(AST_Ident *ident)
{
	AST_Ident *copy = create_ident_node(NULL);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(ident));
	copy->text_buf = ident->text_buf;
	copy->text_len = ident->text_len;
	/* @todo ident->decl as param. Now it will be NULL in the copy. */
	return copy;
}

AST_Type_Decl *copy_type_decl_node(AST_Type_Decl *decl, AST_Node *ident, AST_Node *body)
{
	AST_Type_Decl *copy = create_type_decl_node();
	ASSERT(ident->type == AST_ident);
	ASSERT(body->type == AST_scope);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->ident = (AST_Ident*)ident;
	copy->body = (AST_Scope*)body;
	return copy;
}

AST_Var_Decl *copy_var_decl_node(AST_Var_Decl *decl, AST_Node *type, AST_Node *ident, AST_Node *value)
{
	AST_Var_Decl *copy = create_var_decl_node();
	ASSERT(ident->type == AST_ident);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->type = type;
	copy->ptr_depth = decl->ptr_depth;
	copy->ident = (AST_Ident*)ident;
	copy->value = value;
	return copy;
}

AST_Func_Decl *copy_func_decl_node(AST_Func_Decl *decl, AST_Node *return_type, AST_Node *ident, AST_Node **params, int param_count, AST_Node *body)
{
	int i;
	AST_Func_Decl *copy = create_func_decl_node();
	ASSERT(ident->type == AST_ident);
	ASSERT(!body || body->type == AST_scope);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(decl));
	copy->return_type = return_type;
	copy->ident = (AST_Ident*)ident;
	for (i = 0; i < param_count; ++i) {
		ASSERT(params[i]->type == AST_var_decl);
		push_array(AST_Var_Decl_Ptr)(&copy->params, (AST_Var_Decl_Ptr)params[i]);
	}
	copy->body = (AST_Scope*)body;
	return copy;
}

AST_Literal *copy_literal_node(AST_Literal *literal)
{
	AST_Literal *copy = create_literal_node();
	copy_ast_node_base(AST_BASE(copy), AST_BASE(literal));
	*copy = *literal;
	return copy;
}

AST_Biop *copy_biop_node(AST_Biop *biop, AST_Node *lhs, AST_Node *rhs)
{
	AST_Biop *copy = create_biop_node(biop->type, lhs, rhs);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(biop));
	return copy;
}

AST_Control *copy_control_node(AST_Control *control, AST_Node *value)
{
	AST_Control *copy = create_control_node(control->type);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(control));
	copy->value = value;
	return copy;
}

AST_Call *copy_call_node(AST_Call *call, AST_Node *ident, AST_Node **args, int arg_count)
{
	int i;
	AST_Call *copy = create_call_node();
	ASSERT(ident->type == AST_ident);
	copy_ast_node_base(AST_BASE(copy), AST_BASE(call));
	copy->ident = (AST_Ident*)ident;
	for (i = 0; i < arg_count; ++i) {
		push_array(AST_Node_Ptr)(&copy->args, args[i]);
	}
	return copy;
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
		destroy_array(AST_Node_Ptr)(&scope->nodes);
	} break;

	case AST_ident: {
	} break;

	case AST_type_decl: {
		CASTED_NODE(AST_Type_Decl, decl, node);
		destroy_node(AST_BASE(decl->ident));
		destroy_node(AST_BASE(decl->body));
	} break;

	case AST_var_decl: {
		CASTED_NODE(AST_Var_Decl, decl, node);
		destroy_node(decl->type);
		destroy_node(AST_BASE(decl->ident));
		destroy_node(decl->value);
	} break;

	case AST_func_decl: {
		CASTED_NODE(AST_Func_Decl, decl, node);
		destroy_node(decl->return_type);
		destroy_node(AST_BASE(decl->ident));
		for (i = 0; i < decl->params.size; ++i)
			destroy_node(AST_BASE(decl->params.data[i]));
		destroy_array(AST_Var_Decl_Ptr)(&decl->params);
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
		destroy_array(AST_Node_Ptr)(&call->args);
	} break;
	default: FAIL(("destroy_node: Unknown node type %i", node->type));
	}
	destroy_array(Token_Ptr)(&node->pre_comments);
	destroy_array(Token_Ptr)(&node->post_comments);
	free(node);
}

DEFINE_ARRAY(AST_Node_Ptr)
DEFINE_ARRAY(AST_Var_Decl_Ptr)
DEFINE_ARRAY(Token_Ptr)

/* Mirrors call stack in parsing */
/* Used in searching, setting parent nodes, and backtracking */
typedef struct ParseStackFrame {
	Token *begin_tok;
	AST_Node **node; /* Ptr to node ptr */
} ParseStackFrame;

DECLARE_ARRAY(ParseStackFrame)
DEFINE_ARRAY(ParseStackFrame)

typedef struct Parse_Ctx {
	AST_Scope *root;
	Token *first_tok; /* @todo Consider having some Token_sof, corresponding to Token_eof*/
	Token *tok; /* Access with cur_tok */

	Array(char) error_msg;
	Token *error_tok;
	Array(ParseStackFrame) parse_stack;
} Parse_Ctx;


/* Token manipulation */

INTERNAL Token *cur_tok(Parse_Ctx *ctx)
{ return ctx->tok; }

INTERNAL void advance_tok(Parse_Ctx *ctx)
{
	ASSERT(ctx->tok->type != Token_eof);
	do {
		++ctx->tok;
	} while (is_comment_tok(ctx->tok->type));
}

INTERNAL bool accept_tok(Parse_Ctx *ctx, Token_Type type)
{
	if (ctx->tok->type == type) {
		advance_tok(ctx);
		return true;
	}
	return false;
}


/* Backtracking / stack traversing */

INTERNAL void begin_node_parsing(Parse_Ctx *ctx, AST_Node **node)
{
	ParseStackFrame frame = {0};
	ASSERT(node);
	frame.begin_tok = cur_tok(ctx);
	frame.node = node;
	push_array(ParseStackFrame)(&ctx->parse_stack, frame);
}

INTERNAL void end_node_parsing(Parse_Ctx *ctx)
{
	ParseStackFrame frame = pop_array(ParseStackFrame)(&ctx->parse_stack);
	ASSERT(frame.node);
	ASSERT(*frame.node);

	/* frame.node is used in end_node_parsing because node might not yet be created at the begin_node_parsing */
	(*frame.node)->begin_tok = frame.begin_tok;

	/*	Gather comments around node if this is the first time calling end_node_parsing with this node.
		It's possible to have multiple begin...end_node_parsing with the same node because of nesting,
		like when parsing statement: 'foo;', which yields parse_expr(parse_ident()). */
	if ((*frame.node)->pre_comments.size == 0) {
		Token *tok = frame.begin_tok - 1;
		/* Rewind first */
		while (tok >= ctx->first_tok && is_comment_tok(tok->type))
			--tok;
		++tok;
		while (is_comment_tok(tok->type) && tok->comment_bound_to == 1) {
			push_array(Token_Ptr)(&(*frame.node)->pre_comments, tok);
			++tok;
		}
	}
	if ((*frame.node)->post_comments.size == 0) {
		Token *tok = cur_tok(ctx);
		/* Cursor has been advanced past following comments, rewind */
		--tok;
		while (tok >= ctx->first_tok && is_comment_tok(tok->type))
			--tok;
		++tok;
		while (is_comment_tok(tok->type) && tok->comment_bound_to == -1) {
			push_array(Token_Ptr)(&(*frame.node)->post_comments, tok);
			++tok;
		}
	}
}

INTERNAL void cancel_node_parsing(Parse_Ctx *ctx)
{
	ParseStackFrame frame = pop_array(ParseStackFrame)(&ctx->parse_stack);
	ASSERT(frame.node);
	destroy_node(*frame.node);

	/* Backtrack */
	ctx->tok = frame.begin_tok;
}

void report_error(Parse_Ctx *ctx, const char *fmt, ...)
{
	Array(char) msg;
	va_list args;

	if (cur_tok(ctx) <= ctx->error_tok)
		return; /* Don't overwrite error generated from less succesfull parsing (heuristic) */

	va_start(args, fmt);
	msg = create_array(char)(0);
	safe_vsprintf(&msg, fmt, args);
	va_end(args);

	destroy_array(char)(&ctx->error_msg);
	ctx->error_msg = msg;
	ctx->error_tok = cur_tok(ctx);
}

INTERNAL bool is_decl(AST_Node *node)
{ return node->type == AST_type_decl || node->type == AST_var_decl || node->type == AST_func_decl; }

INTERNAL AST_Ident *decl_ident(AST_Node *node)
{
	ASSERT(is_decl(node));
	switch (node->type) {
		case AST_type_decl: {
			CASTED_NODE(AST_Type_Decl, decl, node);
			return decl->ident;
		} break;
		case AST_var_decl: {
			CASTED_NODE(AST_Var_Decl, decl, node);
			return decl->ident;
		} break;
		case AST_func_decl: {
			CASTED_NODE(AST_Func_Decl, decl, node);
			return decl->ident;
		} break;
		default: FAIL(("decl_ident: invalid node type %i", node->type));
	}
}

/* Find declaration visible in current parse scope */
INTERNAL AST_Node *find_decl_scoped(Parse_Ctx *ctx, const char *buf, int buf_len)
{
	int f, i;
	AST_Node *found_decl = NULL;
	for (f = ctx->parse_stack.size - 1; f >= 0; --f) {
		AST_Node *stack_node = *ctx->parse_stack.data[f].node;
		if (!stack_node || stack_node->type != AST_scope)
			continue;

		{
			CASTED_NODE(AST_Scope, scope, stack_node);
			for (i = 0; i < scope->nodes.size; ++i) {
				AST_Node *node = scope->nodes.data[i];
				if (!is_decl(node))
					continue;

				{
					AST_Ident *ident = decl_ident(node);
/*					printf("trying to match decl %.*s <=> %.*s\n",
							buf_len, buf,
							TOK_ARGS(ident));
							*/
					if (ident->text_len != buf_len)
						continue;
					if (strncmp(buf, ident->text_buf, buf_len))
						continue;

					/* Found declaration for identifier named 'buf' */
					found_decl = node;
					break;
				}
			}
		}
		if (found_decl)
			break;
	}
	return found_decl;
}


/* Parsing */
INTERNAL bool parse_ident(Parse_Ctx *ctx, AST_Node **ret, AST_Node *decl);
INTERNAL bool parse_type_decl(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_var_decl(Parse_Ctx *ctx, AST_Node **ret, bool is_param_decl);
INTERNAL bool parse_func_decl(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_block(Parse_Ctx *ctx, AST_Scope **ret);
INTERNAL bool parse_literal(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_expr(Parse_Ctx *ctx, AST_Node **ret, int min_prec);
INTERNAL bool parse_element(Parse_Ctx *ctx, AST_Node **ret);


/* If decl is NULL, then declaration is searched. */
/* Parse example: foo */
INTERNAL bool parse_ident(Parse_Ctx *ctx, AST_Node **ret, AST_Node *decl)
{
	Token *tok = cur_tok(ctx);
	AST_Ident *ident = create_ident_node(tok);

	begin_node_parsing(ctx, (AST_Node**)&ident);

	if (tok->type != Token_name) {
		report_error(ctx, "'%.*s' is not an identifier", TOK_ARGS(tok));
		goto mismatch;
	}

	if (decl) {
		ident->decl = decl;
	} else {
		ident->decl = find_decl_scoped(ctx, tok->text_buf, tok->text_len);
		if (!ident->decl) {
			report_error(ctx, "'%.*s' is not declared in this scope", TOK_ARGS(tok));
			goto mismatch;
		}
	}
	advance_tok(ctx);

	end_node_parsing(ctx);

	*ret = (AST_Node*)ident;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse examples: int test -- int func() { } -- struct foo { } */
INTERNAL bool parse_type_decl(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Type_Decl *decl = create_type_decl_node();
	begin_node_parsing(ctx, (AST_Node**)&decl);

	if (!accept_tok(ctx, Token_kw_struct))
		goto mismatch;

	if (!parse_ident(ctx, (AST_Node**)&decl->ident, AST_BASE(decl))) {
		report_error(ctx, "Expected type name, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	if (parse_block(ctx, &decl->body))
		;
	else {
		report_error(ctx, "Expected '{', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = (AST_Node*)decl;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_var_decl(Parse_Ctx *ctx, AST_Node **ret, bool is_param_decl)
{
	AST_Var_Decl *decl = create_var_decl_node();
	begin_node_parsing(ctx, (AST_Node**)&decl);

	/* @todo ptrs, typedefs, const, types with multiple identifiers... */

	if (!parse_ident(ctx, (AST_Node**)&decl->type, NULL)) {
		report_error(ctx, "Expected type name, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	/* @todo Move to function and use in function declaration also */
	while (accept_tok(ctx, Token_mul))
		++decl->ptr_depth;


	if (!parse_ident(ctx, (AST_Node**)&decl->ident, AST_BASE(decl))) {
		report_error(ctx, "Expected identifier, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	if (!is_param_decl) {
		if (!accept_tok(ctx, Token_semi)) {
			report_error(ctx, "Expected ';' before '%.*s'", TOK_ARGS(cur_tok(ctx)));
			goto mismatch;
		}
	}

	end_node_parsing(ctx);

	*ret = (AST_Node*)decl;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_func_decl(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Func_Decl *decl = create_func_decl_node();
	begin_node_parsing(ctx, (AST_Node**)&decl);

	if (!parse_ident(ctx, (AST_Node**)&decl->return_type, NULL)) {
		report_error(ctx, "Expected type name, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	if (!parse_ident(ctx, (AST_Node**)&decl->ident, AST_BASE(decl))) {
		report_error(ctx, "Expected identifier, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	if (!accept_tok(ctx, Token_open_paren)) {
		report_error(ctx, "Expected '(', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	{ /* Parse parameter declaration list */
		while (cur_tok(ctx)->type != Token_close_paren) {
			AST_Var_Decl *param_decl = NULL;
			if (cur_tok(ctx)->type == Token_comma)
				advance_tok(ctx);

			if (!parse_var_decl(ctx, (AST_Node**)&param_decl, true))
				goto mismatch;

			push_array(AST_Var_Decl_Ptr)(&decl->params, param_decl);
		}
	}

	if (!accept_tok(ctx, Token_close_paren)) {
		report_error(ctx, "Expected ')', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	if (cur_tok(ctx)->type == Token_semi) {
		/* No body */
		accept_tok(ctx, Token_semi);
	} else if (parse_block(ctx, &decl->body)) {
		;
	} else {
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = (AST_Node*)decl;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: { .... } */
INTERNAL bool parse_block(Parse_Ctx *ctx, AST_Scope **ret)
{
	AST_Scope *scope = create_scope_node();

	begin_node_parsing(ctx, (AST_Node**)&scope);

	if (!accept_tok(ctx, Token_open_brace)) {
		report_error(ctx, "Expected '{', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	while (!accept_tok(ctx, Token_close_brace)) {
		AST_Node *element = NULL;
		if (!parse_element(ctx, &element))
			goto mismatch;
		push_array(AST_Node_Ptr)(&scope->nodes, element);
	}

	end_node_parsing(ctx);

	*ret = scope;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: 1234 */
INTERNAL bool parse_literal(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Literal *literal = create_literal_node();
	Token *tok = cur_tok(ctx);

	begin_node_parsing(ctx, (AST_Node**)&literal);

	switch (tok->type) {
		case Token_number:
			literal->type = Literal_int;
			literal->value.integer = str_to_int(tok->text_buf, tok->text_len);
		break;
		case Token_string:
			literal->type = Literal_string;
			literal->value.string.buf = tok->text_buf;
			literal->value.string.len = tok->text_len;
		break;
		default:
			report_error(ctx, "Expected literal, got '%.*s'", TOK_ARGS(tok));
			goto mismatch;
	}
	advance_tok(ctx);

	end_node_parsing(ctx);

	*ret = (AST_Node*)literal;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: var = 5 + 3 * 2; */
INTERNAL bool parse_expr(Parse_Ctx *ctx, AST_Node **ret, int min_prec)
{
	AST_Node *expr = NULL;

	begin_node_parsing(ctx, &expr);

	if (parse_literal(ctx, &expr)) {
		;
	} else if (parse_ident(ctx, &expr, NULL)) {
		CASTED_NODE(AST_Ident, ident, expr);
		if (ident->decl->type == AST_type_decl) {
			report_error(ctx, "Expression can't start with a type name (%.*s)", TOK_ARGS(ident));
			goto mismatch;
		} else if (ident->decl->type == AST_func_decl) {
			/* This is a function call */
			AST_Call *call = create_call_node();
			call->ident = ident;
			expr = AST_BASE(call);

			if (!accept_tok(ctx, Token_open_paren)) {
				report_error(ctx, "Expected '(', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
				goto mismatch;
			}

			{ /* Parse argument list */
				while (cur_tok(ctx)->type != Token_close_paren) {
					AST_Node *arg = NULL;
					if (cur_tok(ctx)->type == Token_comma)
						advance_tok(ctx);

					if (!parse_expr(ctx, (AST_Node**)&arg, true))
						goto mismatch;

					push_array(AST_Node_Ptr)(&call->args, arg);
				}
			}

			if (!accept_tok(ctx, Token_close_paren)) {
				report_error(ctx, "Expected ')', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
				goto mismatch;
			}
		}
	} else {
		report_error(ctx, "Expected identifier or literal, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}
	/* @todo ^ parse parens */

	while (	is_op(cur_tok(ctx)->type) &&
			op_prec(cur_tok(ctx)->type) >= min_prec) {
		AST_Node *rhs = NULL;
		Token *tok = cur_tok(ctx);
		int prec = op_prec(tok->type);
		int assoc = op_assoc(tok->type);
		int next_min_prec;
		if (assoc == -1)
			next_min_prec = prec + 1;
		else
			next_min_prec = prec;
		advance_tok(ctx);

		if (!parse_expr(ctx, &rhs, next_min_prec))
			goto mismatch;

		expr = AST_BASE(create_biop_node(tok->type, expr, rhs));
	}

	accept_tok(ctx, Token_semi);

	end_node_parsing(ctx);

	*ret = expr;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: return 42; */
INTERNAL bool parse_control(Parse_Ctx *ctx, AST_Node **ret)
{
	Token *tok = cur_tok(ctx);
	AST_Control *control = create_control_node(tok->type);

	begin_node_parsing(ctx, (AST_Node**)&control);

	switch (tok->type) {
		case Token_kw_return: {
			advance_tok(ctx);
			if (!parse_element(ctx, &control->value)) {
				report_error(ctx, "Expected return value, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
				goto mismatch;
			}
		} break;
		case Token_kw_goto: {
			advance_tok(ctx);
			if (!parse_element(ctx, &control->value)) {
				report_error(ctx, "Expected goto label, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
				goto mismatch;
			}
		} break;
		case Token_kw_continue:
		case Token_kw_break:
			advance_tok(ctx);
		break;
		default:
			report_error(ctx, "Expected control statement, got '%.*s'", TOK_ARGS(tok));
			goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = (AST_Node*)control;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse the next self-contained thing - var decl, function decl, statement, expr... */
INTERNAL bool parse_element(Parse_Ctx *ctx, AST_Node **ret)
{
	AST_Node *result = NULL;

	begin_node_parsing(ctx, &result);

	if (parse_type_decl(ctx, &result))
		;
	else if (parse_var_decl(ctx, &result, false))
		;
	else if (parse_func_decl(ctx, &result))
		;
	else if (parse_expr(ctx, &result, 0))
		;
	else if (parse_control(ctx, &result))
		;
	else 
		goto mismatch;

	end_node_parsing(ctx);
	*ret = result;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

AST_Scope *parse_tokens(Token *toks)
{
	bool failure = false;
	Parse_Ctx ctx = {0};
	AST_Scope *root = create_ast_tree();

	ctx.root = root;
	ctx.parse_stack = create_array(ParseStackFrame)(32);
	ctx.tok = ctx.first_tok = toks;
	if (is_comment_tok(ctx.tok->type))
		advance_tok(&ctx);

	begin_node_parsing(&ctx, (AST_Node**)&root);

	while (ctx.tok->type != Token_eof) {
		AST_Node *elem = NULL;
		if (!parse_element(&ctx, &elem)) {
			failure = true;
			break;
		}
		push_array(AST_Node_Ptr)(&root->nodes, elem); 
	}
	end_node_parsing(&ctx);

	if (failure) {
		Token *tok = ctx.error_tok;
		const char *msg = ctx.error_msg.data;
		if (tok && msg) {
			printf("Error at line %i near token '%.*s':\n   %s\n",
					tok->line, TOK_ARGS(tok), msg);
		} else {
			printf("Internal parser error (excuse)\n");
		}
		printf("Compilation failed\n");
		destroy_ast_tree(root);
		root = NULL;
	}

	destroy_array(char)(&ctx.error_msg);
	destroy_array(ParseStackFrame)(&ctx.parse_stack);

	return root;
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

void print_ast(AST_Node *node, int indent)
{
	int i;
	if (!node)
		return;

	print_indent(indent);

	switch (node->type) {
		case AST_scope: {
			CASTED_NODE(AST_Scope, scope, node);
			printf("scope\n");
			for (i = 0; i < scope->nodes.size; ++i)
				print_ast(scope->nodes.data[i], indent + 2);
		} break;
		case AST_ident: {
			CASTED_NODE(AST_Ident, ident, node);
			printf("ident: %.*s\n", TOK_ARGS(ident));
		} break;
		case AST_type_decl: {
			CASTED_NODE(AST_Type_Decl, decl, node);
			printf("type_decl\n");
			print_ast(AST_BASE(decl->ident), indent + 2);
			print_ast(AST_BASE(decl->body), indent + 2);
		} break;
		case AST_var_decl: {
			CASTED_NODE(AST_Var_Decl, decl, node);
			printf("var_decl\n");
			print_ast(decl->type, indent + 2);
			print_ast(AST_BASE(decl->ident), indent + 2);
			print_ast(decl->value, indent + 2);
		} break;
		case AST_func_decl: {
			CASTED_NODE(AST_Func_Decl, decl, node);
			printf("func_decl\n");
			print_ast(decl->return_type, indent + 2);
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
		default: FAIL(("print_ast: Unknown node type %i", node->type));
	};
}

