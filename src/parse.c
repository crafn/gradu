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

int op_prec(TokenType type)
{
	switch (type) {
		case TokenType_assign: return 1;
		case TokenType_add: return 2;
		case TokenType_mul: return 3;
		default: return -1;
	}
}

/* -1 left, 1 right */
int op_assoc(TokenType type)
{
	switch (type) {
		case TokenType_assign: return 1; /* a = b = c  <=>  (a = (b = c)) */
		case TokenType_add: return -1;
		case TokenType_mul: return -1;
		default: return -1;
	}
}

bool is_op(TokenType type)
{ return op_prec(type) >= 0; }

INTERNAL AstNode *create_node_impl(AstNodeType type, int size)
{
	AstNode *n = calloc(1, size);
	n->type = type;
	return n;
}
#define CREATE_NODE(type, type_enum) ((type*)create_node_impl(type_enum, sizeof(type)))

ScopeAstNode *create_scope_node()
{
	ScopeAstNode *scope = CREATE_NODE(ScopeAstNode, AstNodeType_scope);
	scope->nodes = create_array(AstNodePtr)(8);
	return scope;
}

IdentAstNode *create_ident_node(Token *tok)
{
	IdentAstNode * ident = CREATE_NODE(IdentAstNode, AstNodeType_ident);
	if (tok) {
		ident->text_buf = tok->text_buf;
		ident->text_len = tok->text_len;
	}
	return ident;
}

DeclAstNode *create_decl_node()
{ return CREATE_NODE(DeclAstNode, AstNodeType_decl); }

LiteralAstNode *create_literal_node()
{ return CREATE_NODE(LiteralAstNode, AstNodeType_literal); }

BiopAstNode *create_biop_node(TokenType type, AstNode *lhs, AstNode *rhs)
{
	BiopAstNode *biop = CREATE_NODE(BiopAstNode, AstNodeType_biop);
	biop->type = type;
	biop->lhs = lhs;
	biop->rhs = rhs;
	return biop;
}

/* Node copying */

ScopeAstNode *copy_scope_node(ScopeAstNode *scope, AstNode **subnodes, int subnode_count)
{
	ScopeAstNode *copy = create_scope_node();
	int i;
	for (i = 0; i < subnode_count; ++i)
		push_array(AstNodePtr)(&copy->nodes, subnodes[i]);
	copy->is_root = scope->is_root;
	return copy;
}

IdentAstNode *copy_ident_node(IdentAstNode *ident)
{
	IdentAstNode *copy = create_ident_node(NULL);
	copy->text_buf = ident->text_buf;
	copy->text_len = ident->text_len;
	/* @todo ident->decl as param. Now it will be NULL in the copy. */
	return copy;
}

DeclAstNode *copy_decl_node(DeclAstNode *decl, AstNode *type, AstNode *ident, AstNode *value)
{
	DeclAstNode *copy = create_decl_node();
	ASSERT(ident->type == AstNodeType_ident);
	copy->type = type;
	copy->ident = (IdentAstNode*)ident;
	copy->value = value;
	copy->is_type_decl = decl->is_type_decl;
	copy->is_var_decl = decl->is_var_decl;
	copy->is_func_decl = decl->is_func_decl;
	return copy;
}

LiteralAstNode *copy_literal_node(LiteralAstNode *literal)
{
	LiteralAstNode *copy = create_literal_node();
	*copy = *literal;
	return copy;
}

BiopAstNode *copy_biop_node(BiopAstNode *biop, AstNode *lhs, AstNode *rhs)
{
	BiopAstNode *copy = create_biop_node(biop->type, lhs, rhs);
	return copy;
}

/* Recursive */
void destroy_node(AstNode *node)
{
	if (!node)
		return;
	switch (node->type) {
		case AstNodeType_scope: {
			CASTED_NODE(ScopeAstNode, scope, node);
			int i;
			for (i = 0; i < scope->nodes.size; ++i)
				destroy_node(scope->nodes.data[i]);
			destroy_array(AstNodePtr)(&scope->nodes);
		} break;
		case AstNodeType_ident: {
		} break;
		case AstNodeType_decl: {
			CASTED_NODE(DeclAstNode, decl, node);
			destroy_node(decl->type);
			destroy_node(AST_BASE(decl->ident));
			destroy_node(decl->value);
		} break;
		case AstNodeType_literal: {
		} break;
		case AstNodeType_biop: {
			CASTED_NODE(BiopAstNode, op, node);
			destroy_node(op->lhs);
			destroy_node(op->rhs);
		} break;
		default: FAIL(("destroy_node: Unknown node type %i", node->type));
	}
	free(node);
}

DEFINE_ARRAY(AstNodePtr)

/* Mirrors call stack in parsing */
/* Used in searching, setting parent nodes, and backtracking */
typedef struct ParseStackFrame {
	Token *begin_tok;
	AstNode *node;
} ParseStackFrame;

DECLARE_ARRAY(ParseStackFrame)
DEFINE_ARRAY(ParseStackFrame)

typedef struct ParseCtx {
	ScopeAstNode *root;
	Token *tok; /* Access with cur_tok */

	Array(char) error_msg;
	Token *error_tok;
	Array(ParseStackFrame) parse_stack;
} ParseCtx;


/* Token manipulation */

INTERNAL Token *cur_tok(ParseCtx *ctx)
{ return ctx->tok; }

/*
INTERNAL Token *next_tok(Token *tok)
{
	if (tok->type != TokenType_eof)
		return tok + 1;
	return tok;
}
*/

INTERNAL void advance_tok(ParseCtx *ctx)
{
	ASSERT(ctx->tok->type != TokenType_eof);
	++ctx->tok;
}

INTERNAL bool accept_tok(ParseCtx *ctx, TokenType type)
{
	if (ctx->tok->type == type) {
		advance_tok(ctx);
		return true;
	}
	return false;
}


/* Backtracking / stack traversing */

INTERNAL void begin_node_parsing(ParseCtx *ctx, AstNode *node)
{
	ParseStackFrame frame = {0};
	frame.begin_tok = cur_tok(ctx);
	frame.node = node;
	push_array(ParseStackFrame)(&ctx->parse_stack, frame);
}

INTERNAL void end_node_parsing(ParseCtx *ctx)
{
	pop_array(ParseStackFrame)(&ctx->parse_stack);
}

INTERNAL void cancel_node_parsing(ParseCtx *ctx)
{
	ParseStackFrame frame = pop_array(ParseStackFrame)(&ctx->parse_stack);
	destroy_node(frame.node);
	ctx->tok = frame.begin_tok;
}

void report_error(ParseCtx *ctx, const char *fmt, ...)
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

/* Find declaration visible in current parse scope */
INTERNAL DeclAstNode *find_decl_scoped(ParseCtx *ctx, const char *buf, int buf_len)
{
	int f, i;
	DeclAstNode *found_decl = NULL;
	for (f = ctx->parse_stack.size - 1; f >= 0; --f) {
		AstNode *stack_node = ctx->parse_stack.data[f].node;
		if (!stack_node || stack_node->type != AstNodeType_scope)
			continue;

		{
			CASTED_NODE(ScopeAstNode, scope, stack_node);
			for (i = 0; i < scope->nodes.size; ++i) {
				AstNode *node = scope->nodes.data[i];
				if (node->type != AstNodeType_decl)
					continue;

				{
					CASTED_NODE(DeclAstNode, decl, node);
					/*printf("trying to match decl %.*s <=> %.*s\n",
							buf_len, buf,
							TOK_ARGS(decl->ident),
							*/
					if (decl->ident->text_len != buf_len)
						continue;
					if (strncmp(buf, decl->ident->text_buf, buf_len))
						continue;

					/* Found declaration for identifier named 'buf' */
					found_decl = decl;
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
INTERNAL bool parse_ident(ParseCtx *ctx, AstNode **ret, DeclAstNode *decl);
INTERNAL bool parse_decl(ParseCtx *ctx, AstNode **ret);
INTERNAL bool parse_block(ParseCtx *ctx, AstNode **ret);
INTERNAL bool parse_literal(ParseCtx *ctx, AstNode **ret);
INTERNAL bool parse_expr(ParseCtx *ctx, AstNode **ret, int min_prec);
INTERNAL AstNode *parse_element(ParseCtx *ctx);


/* If decl is NULL, then declaration is searched. */
/* Parse example: foo */
INTERNAL bool parse_ident(ParseCtx *ctx, AstNode **ret, DeclAstNode *decl)
{
	Token *tok = cur_tok(ctx);
	IdentAstNode *ident = create_ident_node(tok);

	begin_node_parsing(ctx, AST_BASE(ident));

	if (tok->type != TokenType_name) {
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

	*ret = (AstNode*)ident;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse examples: int test -- int func() { } -- struct foo { } */
INTERNAL bool parse_decl(ParseCtx *ctx, AstNode **ret)
{
	DeclAstNode *decl = create_decl_node();
	IdentAstNode *type = NULL;
	IdentAstNode *ident = NULL;
	AstNode *value = NULL;

	begin_node_parsing(ctx, AST_BASE(decl));


	/* @todo ptrs, typedefs, const, types with multiple identifiers... */

	if (accept_tok(ctx, TokenType_kw_struct)) {
		/* This is a struct definition */

		if (!parse_ident(ctx, (AstNode**)&ident, decl)) {
			report_error(ctx, "Expected type name, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
			goto mismatch;
		}
		decl->ident = ident;
		decl->is_type_decl = true;

		if (parse_block(ctx, &value))
			decl->value = value;
		 else {
			report_error(ctx, "Expected '{', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
			goto mismatch;
		 }
	} else {
		/* This is variable or function declaration */

		if (!parse_ident(ctx, (AstNode**)&type, NULL)) { /* @todo Ptr to type decl */
			report_error(ctx, "Expected type name, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
			goto mismatch;
		}
		decl->type = (AstNode*)type;

		if (!parse_ident(ctx, (AstNode**)&ident, decl)) {
			report_error(ctx, "Expected identifier, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
			goto mismatch;
		}
		decl->ident = ident;

		if (accept_tok(ctx, TokenType_open_paren)) {
			decl->is_func_decl = true;

			if (!accept_tok(ctx, TokenType_close_paren)) {
				report_error(ctx, "Expected ')', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
				goto mismatch;
			}

			if (cur_tok(ctx)->type == TokenType_semi) {
				/* No body */
			} else if (parse_block(ctx, &value)) {
				/* Body parsed */
				decl->value = value;
			} else {
				goto mismatch;
			}
		} else {
			decl->is_var_decl = true;
			if (!accept_tok(ctx, TokenType_semi)) {
				report_error(ctx, "Expected ';' before '%.*s'", TOK_ARGS(cur_tok(ctx)));
				goto mismatch;
			}
		}
	}

	end_node_parsing(ctx);

	*ret = (AstNode*)decl;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: { .... } */
INTERNAL bool parse_block(ParseCtx *ctx, AstNode **ret)
{
	ScopeAstNode *scope = create_scope_node();

	begin_node_parsing(ctx, AST_BASE(scope));

	if (!accept_tok(ctx, TokenType_open_brace)) {
		report_error(ctx, "Expected '{', got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}

	while (!accept_tok(ctx, TokenType_close_brace)) {
		AstNode *element = parse_element(ctx);
		if (!element)
			goto mismatch;
		push_array(AstNodePtr)(&scope->nodes, element);
	}

	end_node_parsing(ctx);

	*ret = (AstNode*)scope;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: 1234 */
INTERNAL bool parse_literal(ParseCtx *ctx, AstNode **ret)
{
	LiteralAstNode *literal = create_literal_node();
	Token *tok = cur_tok(ctx);

	begin_node_parsing(ctx, AST_BASE(literal));

	switch (tok->type) {
		case TokenType_number:
			literal->type = LiteralType_int;
			literal->value.integer = str_to_int(tok->text_buf, tok->text_len);
		break;
		default:
			report_error(ctx, "Expected literal, got '%.*s'", TOK_ARGS(tok));
			goto mismatch;
	}
	advance_tok(ctx);

	end_node_parsing(ctx);

	*ret = (AstNode*)literal;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	return false;
}

/* Parse example: var = 5 + 3 * 2 */
INTERNAL bool parse_expr(ParseCtx *ctx, AstNode **ret, int min_prec)
{
	AstNode *expr = NULL;

	begin_node_parsing(ctx, NULL); /* @todo ExprAstNode enclosing all expressions? */

	if (parse_literal(ctx, &expr)) {
		;
	} else if (parse_ident(ctx, &expr, NULL)) {
		CASTED_NODE(IdentAstNode, ident, expr);
		if (ident->decl->is_type_decl) {
			report_error(ctx, "Expression can't start with a type name (%.*s)", TOK_ARGS(ident));
			goto mismatch;
		}
	} else {
		report_error(ctx, "Expected identifier or literal, got '%.*s'", TOK_ARGS(cur_tok(ctx)));
		goto mismatch;
	}
	/* @todo ^ parse parens */

	while (	is_op(cur_tok(ctx)->type) &&
			op_prec(cur_tok(ctx)->type) >= min_prec) {
		AstNode *rhs = NULL;
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

	end_node_parsing(ctx);

	*ret = expr;
	return true;

mismatch:
	cancel_node_parsing(ctx);
	destroy_node(expr);
	return false;
}

/* Parse the next self-contained thing - var decl, function decl, statement, expr... */
INTERNAL AstNode *parse_element(ParseCtx *ctx)
{
	AstNode *result = NULL;

	begin_node_parsing(ctx, NULL);

	if (parse_decl(ctx, &result))
		accept_tok(ctx, TokenType_semi);
	else if (parse_expr(ctx, &result, 0))
		accept_tok(ctx, TokenType_semi);
	else 
		goto mismatch;

	end_node_parsing(ctx);
	return result;

mismatch:
	cancel_node_parsing(ctx);
	return NULL;
}

ScopeAstNode *parse_tokens(Token *toks)
{
	bool failure = false;
	ParseCtx ctx = {0};
	ScopeAstNode *root = create_ast_tree();

	ctx.root = root;
	ctx.tok = toks;
	ctx.parse_stack = create_array(ParseStackFrame)(32);

	begin_node_parsing(&ctx, AST_BASE(root));

	while (ctx.tok->type != TokenType_eof) {
		AstNode *elem = parse_element(&ctx);
		if (!elem) {
			failure = true;
			break;
		}
		push_array(AstNodePtr)(&root->nodes, elem); 
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

ScopeAstNode *create_ast_tree()
{
	ScopeAstNode *root = create_scope_node();
	root->is_root = true;
	return root;
}

void destroy_ast_tree(ScopeAstNode *node)
{ destroy_node((AstNode*)node); }

INTERNAL void print_indent(int indent)
{ printf("%*s", indent, ""); }

void print_ast(AstNode *node, int indent)
{
	int i;
	if (!node)
		return;

	print_indent(indent);

	switch (node->type) {
		case AstNodeType_scope: {
			CASTED_NODE(ScopeAstNode, scope, node);
			printf("scope\n");
			for (i = 0; i < scope->nodes.size; ++i)
				print_ast(scope->nodes.data[i], indent + 2);
		} break;
		case AstNodeType_ident: {
			CASTED_NODE(IdentAstNode, ident, node);
			printf("ident: %.*s\n", TOK_ARGS(ident));
		} break;
		case AstNodeType_decl: {
			CASTED_NODE(DeclAstNode, decl, node);
			printf("decl\n");
			print_ast(decl->type, indent + 2);
			print_ast(AST_BASE(decl->ident), indent + 2);
			print_ast(decl->value, indent + 2);
		} break;
		case AstNodeType_literal: {
			CASTED_NODE(LiteralAstNode, literal, node);
			printf("literal: ");
			switch (literal->type) {
				case LiteralType_int: printf("%i\n", literal->value.integer); break;
				default: FAIL(("Unknown literal type"));
			}
		} break;
		case AstNodeType_biop: {
			CASTED_NODE(BiopAstNode, op, node);
			printf("biop: %s\n", tokentype_str(op->type));
			print_ast(op->lhs, indent + 2);
			print_ast(op->rhs, indent + 2);
		} break;
		default: FAIL(("print_ast: Unknown node type %i", node->type));
	};
}
