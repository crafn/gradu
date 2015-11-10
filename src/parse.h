#ifndef PARSE_H
#define PARSE_H

#include "core.h"
#include "tokenize.h"

typedef enum {
	AstNodeType_scope,
	AstNodeType_ident,
	AstNodeType_decl,
	AstNodeType_literal,
	AstNodeType_biop
		/*
	AstNodeType_uop,
	AstNodeType_ctrl_stmt,
	AstNodeType_call,
	AstNodeType_label,
	AstNodeType_comment
	*/
} AstNodeType;

struct AstNode;
struct DeclAstNode;
typedef struct AstNode *AstNodePtr;

DECLARE_ARRAY(AstNodePtr)

typedef struct AstNode {
	AstNodeType type;
	Token *begin_tok;
} AstNode;

typedef struct ScopeAstNode {
	AstNode b;
	Array(AstNodePtr) nodes;
	bool is_root;
} ScopeAstNode;

typedef struct IdentAstNode {
	AstNode b;
	const char *text_buf;
	int text_len;

	struct DeclAstNode *decl; /* Pointer to node which declares this identifier */
} IdentAstNode;

typedef struct DeclAstNode {
	AstNode b;
	AstNode *type;
	IdentAstNode *ident;
	AstNode *value;

	bool is_type_decl;
	bool is_var_decl;
	bool is_func_decl;
} DeclAstNode;

typedef enum {
	LiteralType_int
} LiteralType;

typedef struct LiteralAstNode {
	AstNode b;
	LiteralType type;
	union {
		/* @todo Different integer sizes etc */
		int integer;
	} value;
} LiteralAstNode;

typedef struct BiopAstNode {
	AstNode b;
	TokenType type;
	AstNode *lhs;
	AstNode *rhs;
} BiopAstNode;

/* Usage: CASTED_NODE(IdentAstNode, ident, generic_node); printf("%c", ident->text_buf[0]); */
#define CASTED_NODE(type, name, assign) \
	type *name = (type*)assign
#define AST_BASE(node) (&(node)->b)

/* Created AST will have pointers to tokens */
ScopeAstNode *parse_tokens(Token *toks);

ScopeAstNode *create_ast_tree();
void destroy_ast_tree(ScopeAstNode *node);

ScopeAstNode *create_scope_node();
IdentAstNode *create_ident_node(Token *tok);
DeclAstNode *create_decl_node();
LiteralAstNode *create_literal_node();
BiopAstNode *create_biop_node(TokenType type, AstNode *lhs, AstNode *rhs);

ScopeAstNode *copy_scope_node(ScopeAstNode *scope, AstNode **subnodes, int subnode_count);
IdentAstNode *copy_ident_node(IdentAstNode *ident);
DeclAstNode *copy_decl_node(DeclAstNode *decl, AstNode *type, AstNode *ident, AstNode *value);
LiteralAstNode *copy_literal_node(LiteralAstNode *literal);
BiopAstNode *copy_biop_node(BiopAstNode *biop, AstNode *lhs, AstNode *rhs);

void destroy_node(AstNode *node);

void print_ast(AstNode *node, int indent);

#endif
