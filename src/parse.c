#include "parse.h"

int str_to_int(Buf_Str text)
{
	const char *c = text.buf;
	const char *end = c + text.len;
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


/* Mirrors call stack in parsing */
/* Used in searching, setting parent nodes, and backtracking */
typedef struct Parse_Stack_Frame {
	Token *begin_tok;
	AST_Node **node; /* Ptr to node ptr */
} Parse_Stack_Frame;

DECLARE_ARRAY(Parse_Stack_Frame)
DEFINE_ARRAY(Parse_Stack_Frame)

typedef AST_Type_Decl* AST_Type_Decl_Ptr;
DECLARE_ARRAY(AST_Type_Decl_Ptr)
DEFINE_ARRAY(AST_Type_Decl_Ptr)

typedef struct Parse_Ctx {
	AST_Scope *root;
	Token *first_tok; /* @todo Consider having some Token_sof, corresponding to Token_eof*/
	Token *tok; /* Access with cur_tok */
	int expr_depth;

	/* Builtin types are generated while parsing */
	Array(AST_Type_Decl_Ptr) builtin_decls;

	Array(char) error_msg;
	Token *error_tok;
	Array(Parse_Stack_Frame) parse_stack;
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
	Parse_Stack_Frame frame = {0};
	ASSERT(node);
	frame.begin_tok = cur_tok(ctx);
	frame.node = node;
	push_array(Parse_Stack_Frame)(&ctx->parse_stack, frame);
}

INTERNAL void end_node_parsing(Parse_Ctx *ctx)
{
	Parse_Stack_Frame frame = pop_array(Parse_Stack_Frame)(&ctx->parse_stack);
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
	Parse_Stack_Frame frame = pop_array(Parse_Stack_Frame)(&ctx->parse_stack);
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
INTERNAL AST_Node *find_decl_scoped(Parse_Ctx *ctx, Buf_Str name)
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
/*					printf("trying to match decl %.*s <=> %s\n",
							BUF_STR_ARGS(buf),
							ident->text.data);
							*/
					if (!buf_str_equals(c_str_to_buf_str(ident->text.data), name))
						continue;

					/* Found declaration for the name */
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

/* @todo AST_Node -> specific node type */
INTERNAL bool parse_ident(Parse_Ctx *ctx, AST_Node **ret, AST_Node *decl, AST_Scope *search_scope);
INTERNAL bool parse_type_decl(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_var_decl(Parse_Ctx *ctx, AST_Node **ret, bool is_param_decl);
INTERNAL bool parse_type_and_ident(Parse_Ctx *ctx, AST_Type **ret_type, AST_Ident **ret_ident, AST_Node *enclosing_decl);
INTERNAL bool parse_func_decl(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_block(Parse_Ctx *ctx, AST_Scope **ret);
INTERNAL bool parse_literal(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_expr(Parse_Ctx *ctx, AST_Node **ret, int min_prec);
INTERNAL bool parse_control(Parse_Ctx *ctx, AST_Node **ret);
INTERNAL bool parse_element(Parse_Ctx *ctx, AST_Node **ret);


/* If decl is NULL, then declaration is searched. */
/* Parse example: foo */
INTERNAL bool parse_ident(Parse_Ctx *ctx, AST_Node **ret, AST_Node *decl, AST_Scope *search_scope)
{
	Token *tok = cur_tok(ctx);
	AST_Ident *ident = create_ident_node();
	append_str(&ident->text, "%.*s", BUF_STR_ARGS(tok->text));

	begin_node_parsing(ctx, (AST_Node**)&ident);

	if (tok->type != Token_name) {
		report_error(ctx, "'%.*s' is not an identifier", BUF_STR_ARGS(tok->text));
		goto mismatch;
	}

	if (decl) {
		ident->decl = decl;
	} else {
		if (search_scope) {
			/* Search from given scope */
			int i;
			for (i = 0; i < search_scope->nodes.size; ++i) {
				AST_Node *subnode = search_scope->nodes.data[i];
				if (!is_decl(subnode))
					continue;
				if (strcmp(decl_ident(subnode)->text.data, ident->text.data))
					continue;

				ident->decl = subnode;
				break;
			}
		} else {
			/* Search from current scope */
			ident->decl = find_decl_scoped(ctx, tok->text);
		}
	}
	if (!ident->decl) {
		report_error(ctx, "'%.*s' is not declared in this scope", BUF_STR_ARGS(tok->text));
		goto mismatch;
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

	if (!parse_ident(ctx, (AST_Node**)&decl->ident, AST_BASE(decl), NULL)) {
		report_error(ctx, "Expected type name, got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
		goto mismatch;
	}

	if (parse_block(ctx, &decl->body))
		;
	else {
		report_error(ctx, "Expected '{', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret = (AST_Node*)decl;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Type and ident in same function because of cases like 'int (*foo)()' */
INTERNAL bool parse_type_and_ident(Parse_Ctx *ctx, AST_Type **ret_type, AST_Ident **ret_ident, AST_Node *enclosing_decl)
{
	AST_Type *type = create_type_node();
	begin_node_parsing(ctx, (AST_Node**)&type);

	/* @todo ptr-to-funcs, const (?), types with multiple identifiers... */

	{ /* Type */
		AST_Node *found_decl = NULL;
		bool is_builtin = false;
		Builtin_Type bt = {0};
		bool recognized = true;

		if (accept_tok(ctx, Token_kw_const))
			type->is_const = true;

		/* Gather all builtin type specifiers (like int, matrix(), field()) */
		while (recognized) {
			Token *tok = cur_tok(ctx);

			switch (tok->type) {
			case Token_kw_void:
				bt.is_void = true;
				advance_tok(ctx);
			break;
			case Token_kw_int:
				bt.is_integer = true;
				bt.bitness = 0; /* Not specified */
				advance_tok(ctx);
			break;
			case Token_kw_size_t:
				bt.is_integer = true;
				bt.bitness = sizeof(size_t)*8; /* @todo Assuming target is same architecture than host */
				bt.is_unsigned = true;
				advance_tok(ctx);
			break;
			case Token_kw_char:
				bt.is_char = true;
				bt.bitness = 8;
				advance_tok(ctx);
			break;
			case Token_kw_float:
				bt.is_float = true;
				bt.bitness = 32;
				advance_tok(ctx);
			break;
			case Token_kw_matrix: {
				bt.is_matrix = true;
				advance_tok(ctx);

				if (!accept_tok(ctx, Token_open_paren)) {
					report_error(ctx, "Expected '(', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
					goto mismatch;
				}

				{ /* Parse dimension list */
					/* @todo Support constant expressions */
					while (cur_tok(ctx)->type == Token_number) {
						int dim = str_to_int(cur_tok(ctx)->text);
						bt.matrix_dim[bt.matrix_rank++] = dim;
						advance_tok(ctx);

						if (bt.matrix_rank > MAX_MATRIX_RANK) {
							report_error(ctx, "Too high rank for a matrix. Max is %i", MAX_MATRIX_RANK);
							goto mismatch;
						}

						if (!accept_tok(ctx, Token_comma))
							break;
					}
				}

				if (!accept_tok(ctx, Token_close_paren)) {
					report_error(ctx, "Expected ')', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
					goto mismatch;
				}
			} break;
			default: recognized = false;
			}
			if (recognized)
				is_builtin = true;
		}

		if (is_builtin) {
			int i;
			/* Search for builtin declaration from existing decls */
			for (i = 0; i < ctx->builtin_decls.size; ++i) {
				AST_Type_Decl *decl = ctx->builtin_decls.data[i];
				Builtin_Type t = decl->builtin_type;
				if (builtin_type_equals(t, bt)) {
					found_decl = (AST_Node*)decl;
					break;
				}
			}
			/* Create new builtin decl if not found */
			if (!found_decl) {
				AST_Type_Decl *decl = create_type_decl_node();
				/* Note that the declaration doesn't have ident -- it's up to backend to generate it */
				decl->is_builtin = true;
				decl->builtin_type = bt;
				push_array(AST_Type_Decl_Ptr)(&ctx->builtin_decls, decl);

				found_decl = AST_BASE(decl);
			}
		} else {
			found_decl = find_decl_scoped(ctx, cur_tok(ctx)->text);
			if (found_decl)
				advance_tok(ctx);
		}
		if (!found_decl || found_decl->type != AST_type_decl) {
			report_error(ctx, "'%.*s' is not declared in this scope", BUF_STR_ARGS(cur_tok(ctx)->text));
			goto mismatch;
		}
		type->base_type_decl = (AST_Type_Decl*)found_decl;
	}

	/* Pointer * */
	while (accept_tok(ctx, Token_mul))
		++type->ptr_depth;

	/* Variable name */
	if (!parse_ident(ctx, (AST_Node**)ret_ident, enclosing_decl, NULL)) {
		report_error(ctx, "Expected identifier, got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
		goto mismatch;
	}

	end_node_parsing(ctx);

	*ret_type = type;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

INTERNAL bool parse_var_decl(Parse_Ctx *ctx, AST_Node **ret, bool is_param_decl)
{
	AST_Var_Decl *decl = create_var_decl_node();
	begin_node_parsing(ctx, (AST_Node**)&decl);

	if (!parse_type_and_ident(ctx, &decl->type, &decl->ident, AST_BASE(decl)))
		goto mismatch;

	if (!is_param_decl) {
		if (!accept_tok(ctx, Token_semi)) {
			report_error(ctx, "Expected ';' before '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
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

	if (!parse_type_and_ident(ctx, &decl->return_type, &decl->ident, AST_BASE(decl)))
		goto mismatch;

	if (!accept_tok(ctx, Token_open_paren)) {
		report_error(ctx, "Expected '(', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
		goto mismatch;
	}

	{ /* Parse parameter declaration list */
		while (cur_tok(ctx)->type != Token_close_paren) {
			AST_Var_Decl *param_decl = NULL;
			if (cur_tok(ctx)->type == Token_comma)
				advance_tok(ctx);

			if (accept_tok(ctx, Token_ellipsis)) {
				decl->ellipsis = true;
				continue;
			}

			if (!parse_var_decl(ctx, (AST_Node**)&param_decl, true))
				goto mismatch;

			push_array(AST_Var_Decl_Ptr)(&decl->params, param_decl);
		}
	}

	if (!accept_tok(ctx, Token_close_paren)) {
		report_error(ctx, "Expected ')', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
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
		report_error(ctx, "Expected '{', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
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
			literal->value.integer = str_to_int(tok->text);
		break;
		case Token_string:
			literal->type = Literal_string;
			literal->value.string= tok->text;
		break;
		default:
			report_error(ctx, "Expected literal, got '%.*s'", BUF_STR_ARGS(tok->text));
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

/* Parses 'foo, bar)' */
INTERNAL bool parse_arg_list(Parse_Ctx *ctx, Array(AST_Node_Ptr) *ret)
{
	while (cur_tok(ctx)->type != Token_close_paren) {
		AST_Node *arg = NULL;
		if (cur_tok(ctx)->type == Token_comma)
			advance_tok(ctx);

		if (!parse_expr(ctx, (AST_Node**)&arg, true))
			goto mismatch;

		push_array(AST_Node_Ptr)(ret, arg);
	}

	if (!accept_tok(ctx, Token_close_paren)) {
		report_error(ctx, "Expected ')', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
		goto mismatch;
	}

	return true;

mismatch:
	return false;
}

/* Parse example: var = 5 + 3 * 2; */
INTERNAL bool parse_expr(Parse_Ctx *ctx, AST_Node **ret, int min_prec)
{
	AST_Node *expr = NULL;

	++ctx->expr_depth;
	begin_node_parsing(ctx, &expr);

	if (parse_literal(ctx, &expr)) {
		;
	} else if (parse_ident(ctx, &expr, NULL, NULL)) {
		CASTED_NODE(AST_Ident, ident, expr);
		if (ident->decl->type == AST_type_decl) {
			report_error(ctx, "Expression can't start with a type name (%s)", ident->text.data);
			goto mismatch;
		} else if (ident->decl->type == AST_func_decl) {
			/* This is a function call */
			AST_Call *call = create_call_node();
			call->ident = ident;
			expr = AST_BASE(call);

			if (!accept_tok(ctx, Token_open_paren)) {
				report_error(ctx, "Expected '(', got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
				goto mismatch;
			}

			if (!parse_arg_list(ctx, &call->args))
				goto mismatch;

		} else if (ident->decl->type == AST_var_decl) {
			/* This is variable access */
			/* @todo This might have to be moved to main expr loop below as '.' operator
			 * with correct associativity and precedence. Think 'a.b.c'. */

			AST_Access *access = create_access_node();
			access->base = AST_BASE(ident);
			expr = AST_BASE(access);

			if (accept_tok(ctx, Token_dot)) {
				access->is_member_access = true;
			} else if (accept_tok(ctx, Token_right_arrow)) {
				access->is_member_access = true;
			} else if (accept_tok(ctx, Token_open_paren)) {
				access->is_element_access = true;
			} else {
				;
			}

			if (access->is_member_access) {
				AST_Ident *sub = NULL;
				AST_Node *base_decl = ident->decl;
				if (base_decl->type != AST_var_decl) {
					report_error(ctx, "@todo: good message for this error");
					goto mismatch;
				}

				{
					CASTED_NODE(AST_Var_Decl, base_var_decl, base_decl);
					AST_Scope *base_type_scope = base_var_decl->type->base_type_decl->body;
					ASSERT(base_type_scope);
					if (!parse_ident(ctx, (AST_Node**)&sub, NULL, base_type_scope))
						goto mismatch;
					push_array(AST_Node_Ptr)(&access->args, AST_BASE(sub));
				}
			} else if (access->is_element_access) {
				if (!parse_arg_list(ctx, &access->args))
					goto mismatch;
			}
		}
	} else {
		report_error(ctx, "Expected identifier or literal, got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
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

		{
			/* @todo Type checking */
			/* @todo Operator overloading */
			AST_Biop *biop = create_biop_node();
			biop->type = tok->type;
			biop->lhs = expr;
			biop->rhs = rhs;
			biop->is_top_level = (ctx->expr_depth == 1);
			expr = AST_BASE(biop);
		}
	}

	accept_tok(ctx, Token_semi);

	end_node_parsing(ctx);
	--ctx->expr_depth;

	*ret = expr;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	--ctx->expr_depth;
	return false;
}

/* Parse example: return 42; */
INTERNAL bool parse_control(Parse_Ctx *ctx, AST_Node **ret)
{
	Token *tok = cur_tok(ctx);
	AST_Control *control = create_control_node();
	control->type = tok->type;

	begin_node_parsing(ctx, (AST_Node**)&control);

	switch (tok->type) {
		case Token_kw_return: {
			advance_tok(ctx);
			if (!parse_element(ctx, &control->value)) {
				report_error(ctx, "Expected return value, got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
				goto mismatch;
			}
		} break;
		case Token_kw_goto: {
			advance_tok(ctx);
			if (!parse_element(ctx, &control->value)) {
				report_error(ctx, "Expected goto label, got '%.*s'", BUF_STR_ARGS(cur_tok(ctx)->text));
				goto mismatch;
			}
		} break;
		case Token_kw_continue:
		case Token_kw_break:
			advance_tok(ctx);
		break;
		default:
			report_error(ctx, "Expected control statement, got '%.*s'", BUF_STR_ARGS(tok->text));
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

	/* @todo Heuristic */
	if (accept_tok(ctx, Token_semi)) {
		report_error(ctx, "Unexpected ';'");
		goto mismatch;
	} else if (parse_type_decl(ctx, &result))
		;
	else if (parse_var_decl(ctx, &result, false))
		;
	else if (parse_func_decl(ctx, &result))
		;
	else if (parse_expr(ctx, &result, 0))
		;
	else if (parse_control(ctx, &result))
		;
	else if (parse_block(ctx, (AST_Scope**)&result))
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
	AST_Scope *root = create_ast();

	ctx.root = root;
	ctx.builtin_decls = create_array(AST_Type_Decl_Ptr)(32);
	ctx.parse_stack = create_array(Parse_Stack_Frame)(32);
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

	{ /* Insert builtin declarations to beginning of root node */
		Array(AST_Node_Ptr) new_nodes = create_array(AST_Node_Ptr)(ctx.builtin_decls.size + root->nodes.size);
		int i;
		/* @todo Array insert function */
		for (i = 0; i < ctx.builtin_decls.size; ++i) {
			push_array(AST_Node_Ptr)(&new_nodes, AST_BASE(ctx.builtin_decls.data[i]));
		}
		for (i = 0; i < root->nodes.size; ++i) {
			push_array(AST_Node_Ptr)(&new_nodes, root->nodes.data[i]);
		}
		destroy_array(AST_Node_Ptr)(&root->nodes);
		root->nodes = new_nodes;
	}

	if (failure) {
		Token *tok = ctx.error_tok;
		const char *msg = ctx.error_msg.data;
		if (tok && msg) {
			printf("Error at line %i near token '%.*s':\n   %s\n",
					tok->line, BUF_STR_ARGS(tok->text), msg);
		} else {
			printf("Internal parser error (excuse)\n");
		}
		printf("Compilation failed\n");
		destroy_ast(root);
		root = NULL;
	}

	destroy_array(char)(&ctx.error_msg);
	destroy_array(Parse_Stack_Frame)(&ctx.parse_stack);
	destroy_array(AST_Type_Decl_Ptr)(&ctx.builtin_decls);

	return root;
}

