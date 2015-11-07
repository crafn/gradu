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

/* Usage: CASTED_NODE(IdentAstNode, ident, generic_node); printf("%c", ident->text_buf[0]); */
#define CASTED_NODE(type, name, assign) \
	type *name = (type*)assign

INTERNAL AstNode *create_node_impl(AstNodeType type, int size)
{
	AstNode *n = calloc(1, size);
	n->type = type;
	return n;
}
#define CREATE_NODE(type, type_enum) ((type*)create_node_impl(type_enum, sizeof(type)))

INTERNAL IdentAstNode *create_ident_node(Token *tok)
{
	IdentAstNode * ident = CREATE_NODE(IdentAstNode, AstNodeType_ident);
	ident->text_buf = tok->text_buf;
	ident->text_len = tok->text_len;
	return ident;
}

INTERNAL DeclAstNode *create_decl_node()
{ return CREATE_NODE(DeclAstNode, AstNodeType_decl); }

INTERNAL BlockAstNode *create_block_node()
{
	BlockAstNode *block = CREATE_NODE(BlockAstNode, AstNodeType_block);
	block->nodes = create_array(AstNodePtr)(8);
	return block;
}

INTERNAL LiteralAstNode *create_literal_node()
{ return CREATE_NODE(LiteralAstNode, AstNodeType_literal); }

INTERNAL BiopAstNode *create_biop_node()
{ return CREATE_NODE(BiopAstNode, AstNodeType_biop); }

/* Recursive */
INTERNAL void destroy_node(AstNode *node)
{
	if (!node)
		return;
	switch (node->type) {
		case AstNodeType_root: {
			CASTED_NODE(RootAstNode, root, node);
			int i;
			for (i = 0; i < root->nodes.size; ++i)
				destroy_node(root->nodes.data[i]);
			destroy_array(AstNodePtr)(&root->nodes);
		} break;
		case AstNodeType_ident: {
		} break;
		case AstNodeType_decl: {
			CASTED_NODE(DeclAstNode, decl, node);
			destroy_node(decl->type);
			destroy_node(&decl->ident->b);
			destroy_node(decl->value);
		} break;
		case AstNodeType_block: {
			CASTED_NODE(BlockAstNode, block, node);
			int i;
			for (i = 0; i < block->nodes.size; ++i)
				destroy_node(block->nodes.data[i]);
			destroy_array(AstNodePtr)(&block->nodes);

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

typedef Token *TokenPtr;
DECLARE_ARRAY(TokenPtr)
DEFINE_ARRAY(TokenPtr)

typedef struct ParseCtx {
	RootAstNode *root;
	Token *tok; /* Access with cur_tok */
	Token *most_advanced_token_used;
	Array(TokenPtr) backtrack_stack;
} ParseCtx;


/* Token manipulation */

INTERNAL Token *cur_tok(ParseCtx *ctx)
{ return ctx->tok; }

INTERNAL Token *next_tok(Token *tok)
{
	if (tok->type != TokenType_eof)
		return tok + 1;
	return tok;
}

INTERNAL void advance_tok(ParseCtx *ctx)
{
	assert(ctx->tok->type != TokenType_eof);
	++ctx->tok;
	if (ctx->tok > ctx->most_advanced_token_used)
		ctx->most_advanced_token_used = ctx->tok;
}

INTERNAL bool accept_tok(ParseCtx *ctx, TokenType type)
{
	if (ctx->tok->type == type) {
		advance_tok(ctx);
		return true;
	}
	return false;
}


/* Backtracking */

INTERNAL void push_backtrack(ParseCtx *ctx)
{ push_array(TokenPtr)(&ctx->backtrack_stack, cur_tok(ctx)); }

INTERNAL void pop_backtrack(ParseCtx *ctx)
{ pop_array(TokenPtr)(&ctx->backtrack_stack); }

INTERNAL void do_backtrack(ParseCtx *ctx)
{ ctx->tok = pop_array(TokenPtr)(&ctx->backtrack_stack); }


/* Parsing */
INTERNAL bool parse_var_decl(ParseCtx *ctx, AstNode **ret);
INTERNAL bool parse_func_decl(ParseCtx *ctx, AstNode **ret);
INTERNAL bool parse_block(ParseCtx *ctx, AstNode **ret);
INTERNAL bool parse_literal(ParseCtx *ctx, AstNode **ret);
INTERNAL bool parse_expr(ParseCtx *ctx, AstNode **ret, int min_prec);
INTERNAL AstNode *parse_element(ParseCtx *ctx);

/* Parse example: int test */
INTERNAL bool parse_var_decl(ParseCtx *ctx, AstNode **ret)
{
	IdentAstNode *type = NULL;
	DeclAstNode *decl = NULL;
	IdentAstNode *ident = NULL;

	push_backtrack(ctx);

	decl = create_decl_node();

	/* @todo ptrs, typedefs, const, types with multiple identifiers... */

	/* Expect type name */
	if (cur_tok(ctx)->type != TokenType_name)
		goto mismatch;
	type = create_ident_node(cur_tok(ctx));
	advance_tok(ctx);
	decl->type = (AstNode*)type;

	/* Expect variable name */
	if (cur_tok(ctx)->type != TokenType_name)
		goto mismatch;
	ident = create_ident_node(cur_tok(ctx));
	advance_tok(ctx);
	decl->ident = ident;

	/* @todo Consider merging var decl and func decl parsing */
	if (cur_tok(ctx)->type == TokenType_open_paren)
		goto mismatch;

	pop_backtrack(ctx);
	*ret = (AstNode*)decl;
	return true;

mismatch:
	do_backtrack(ctx);
	destroy_node((AstNode*)decl);
	return false;
}

/* Parse example: int foo(int a, int b) { return 1; } */
INTERNAL bool parse_func_decl(ParseCtx *ctx, AstNode **ret)
{
	IdentAstNode *type = NULL;
	DeclAstNode *decl = NULL;
	IdentAstNode *ident = NULL;
	AstNode *body = NULL;

	push_backtrack(ctx);

	decl = create_decl_node();

	/* Expect type name */
	if (cur_tok(ctx)->type != TokenType_name)
		goto mismatch;
	type = create_ident_node(cur_tok(ctx));
	advance_tok(ctx);
	decl->type = (AstNode*)type;

	/* Expect variable name */
	if (cur_tok(ctx)->type != TokenType_name)
		goto mismatch;
	ident = create_ident_node(cur_tok(ctx));
	advance_tok(ctx);
	decl->ident = ident;

	if (!accept_tok(ctx, TokenType_open_paren))
		goto mismatch;
	if (!accept_tok(ctx, TokenType_close_paren))
		goto mismatch;

	if (cur_tok(ctx)->type == TokenType_semi) {
		/* No body */
	} else if (parse_block(ctx, &body)) {
		/* Body parsed */
		decl->value = body;
	} else {
		goto mismatch;
	}

	pop_backtrack(ctx);
	*ret = (AstNode*)decl;
	return true;

mismatch:
	do_backtrack(ctx);
	destroy_node((AstNode*)decl);
	return false;
}

/* Parse example: { .... } */
INTERNAL bool parse_block(ParseCtx *ctx, AstNode **ret)
{
	BlockAstNode *block = NULL;

	push_backtrack(ctx);
	block = create_block_node();

	if (!accept_tok(ctx, TokenType_open_brace))
		goto mismatch;

	while (!accept_tok(ctx, TokenType_close_brace)) {
		AstNode *element = parse_element(ctx);
		if (!element)
			goto mismatch;
		push_array(AstNodePtr)(&block->nodes, element);
	}

	pop_backtrack(ctx);

	*ret = (AstNode*)block;
	return true;

mismatch:
	do_backtrack(ctx);
	destroy_node(&block->b);
	return false;
}

/* Parse example: 1234 */
INTERNAL bool parse_literal(ParseCtx *ctx, AstNode **ret)
{
	LiteralAstNode *literal = NULL;
	Token *tok = cur_tok(ctx);

	push_backtrack(ctx);

	literal = create_literal_node();
	switch (tok->type) {
		case TokenType_number:
			literal->type = LiteralType_int;
			literal->value.integer = str_to_int(tok->text_buf, tok->text_len);
		break;
		default: goto mismatch;
	}
	advance_tok(ctx);

	pop_backtrack(ctx);

	*ret = (AstNode*)literal;
	return true;

mismatch:
	do_backtrack(ctx);
	destroy_node(&literal->b);
	return false;
}

/* Parse example: var = 5 + 3 * 2 */
INTERNAL bool parse_expr(ParseCtx *ctx, AstNode **ret, int min_prec)
{
	AstNode *expr = NULL;

	push_backtrack(ctx);

	if (parse_literal(ctx, &expr)) {
		;
	} else if (cur_tok(ctx)->type == TokenType_name) {
		expr = (AstNode*)create_ident_node(cur_tok(ctx));
		advance_tok(ctx);
	} else {
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

		{
			BiopAstNode *biop = create_biop_node();
				biop->type = tok->type;
				biop->lhs = expr;
				biop->rhs = rhs;
			expr = &biop->b;
		}
	}

	pop_backtrack(ctx);

	*ret = expr;
	return true;
mismatch:
	do_backtrack(ctx);
	destroy_node(expr);
	return false;
}

/* Parse the next self-contained thing - var decl, function decl, statement, expr... */
INTERNAL AstNode *parse_element(ParseCtx *ctx)
{
	AstNode *result = NULL;

	push_backtrack(ctx);

	if (parse_var_decl(ctx, &result))
		accept_tok(ctx, TokenType_semi);
	else if (parse_func_decl(ctx, &result))
		accept_tok(ctx, TokenType_semi);
	else if (parse_expr(ctx, &result, 0))
		accept_tok(ctx, TokenType_semi);
	else if (parse_literal(ctx, &result)) /* @todo parse_expr does this? */
		accept_tok(ctx, TokenType_semi);
	else {
		Token *begin = cur_tok(ctx);
		Token *end = next_tok(ctx->most_advanced_token_used);
		printf("Parsing of '%.*s' at line %i failed\n",
				(int)(end->text_buf - begin->text_buf + end->text_len), begin->text_buf,
				begin->line);
#if 0
		printf("Current AST:\n");
		print_ast(&ctx->root->b, 2);
#endif
	}

	pop_backtrack(ctx);
	return result;
}

RootAstNode *parse_tokens(Token *toks)
{
	bool failure = false;
	ParseCtx ctx = {0};
	RootAstNode *root = malloc(sizeof(*root));

	root->nodes = create_array(AstNodePtr)(128);
	root->b.type = AstNodeType_root;

	ctx.root = root;
	ctx.tok = toks;
	ctx.backtrack_stack = create_array(TokenPtr)(32);

	while (ctx.tok->type != TokenType_eof) {
		AstNode *elem = parse_element(&ctx);
		if (!elem) {
			failure = true;
			break;
		}
		push_array(AstNodePtr)(&root->nodes, elem); 
	}
	destroy_array(TokenPtr)(&ctx.backtrack_stack);

	if (failure) {
		printf("Compilation failed\n");
		destroy_ast_tree(root);
		root = NULL;
	}
	return root;
}

void destroy_ast_tree(RootAstNode *node)
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
		case AstNodeType_root: {
			CASTED_NODE(RootAstNode, root, node);
			printf("root\n");
			for (i = 0; i < root->nodes.size; ++i)
				print_ast(root->nodes.data[i], indent + 2);
		} break;
		case AstNodeType_ident: {
			CASTED_NODE(IdentAstNode, ident, node);
			printf("ident: %.*s\n", ident->text_len, ident->text_buf);
		} break;
		case AstNodeType_decl: {
			CASTED_NODE(DeclAstNode, decl, node);
			printf("decl\n");
			print_ast(decl->type, indent + 2);
			print_ast(&decl->ident->b, indent + 2);
			print_ast(decl->value, indent + 2);
		} break;
		case AstNodeType_block: {
			CASTED_NODE(BlockAstNode, block, node);
			printf("block\n");
			for (i = 0; i < block->nodes.size; ++i)
				print_ast(block->nodes.data[i], indent + 2);
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

