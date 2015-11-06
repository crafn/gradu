#include "parse.h"

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

/* Recursive */
INTERNAL void destroy_node(AstNode *node)
{
	if (!node)
		return;
	switch (node->type) {
		case AstNodeType_root: {
			int i;
			CASTED_NODE(RootAstNode, root, node);
			for (i = 0; i < root->nodes.size; ++i)
				free(root->nodes.data[i]);
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
		default: FAIL(("Unknown node type %i", node->type));
	}
	free(node);
}

DEFINE_ARRAY(AstNodePtr)

typedef Token *TokenPtr;
DECLARE_ARRAY(TokenPtr)
DEFINE_ARRAY(TokenPtr)

typedef struct ParseCtx {
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

/* Parse example: int test; */
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

	if (!accept_tok(ctx, TokenType_semi))
		goto mismatch;

	pop_backtrack(ctx);
	*ret = (AstNode*)decl;
	return true;

mismatch:
	do_backtrack(ctx);
	destroy_node((AstNode*)decl);
	return false;
}

/* Parse example: int foo(); */
INTERNAL bool parse_func_decl(ParseCtx *ctx, AstNode **ret)
{
	IdentAstNode *type = NULL;
	DeclAstNode *decl = NULL;
	IdentAstNode *ident = NULL;

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
	if (!accept_tok(ctx, TokenType_semi))
		goto mismatch;

	pop_backtrack(ctx);
	*ret = (AstNode*)decl;
	return true;

mismatch:
	do_backtrack(ctx);
	destroy_node((AstNode*)decl);
	return false;
}


/* Parse the next self-contained thing - var decl, function decl, statement, expr... */
INTERNAL AstNode *parse_element(ParseCtx *ctx)
{
	AstNode *result = NULL;
	if (parse_var_decl(ctx, &result))
		;
	else if (parse_func_decl(ctx, &result))
		;
	else {
		Token *begin = cur_tok(ctx);
		Token *end = next_tok(ctx->most_advanced_token_used);
		printf("Parsing of '%.*s' at line %i failed\n",
				(int)(end->text_buf - begin->text_buf + end->text_len), begin->text_buf,
				begin->line);
	}
	return result;
}

RootAstNode *parse_tokens(Token *toks)
{
	bool failure = false;
	ParseCtx ctx = {0};
	RootAstNode *root = malloc(sizeof(*root));

	root->nodes = create_array(AstNodePtr)(128);
	root->b.type = AstNodeType_root;

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
{
	destroy_node((AstNode*)node);
}

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
		default: FAIL(("Unknown ast node type %i", node->type));
	};
}

