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
} AstNode;

typedef struct ScopeAstNode {
	AstNode b;
	Array(AstNodePtr) nodes;
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

/* Created AST will have pointers to tokens */
ScopeAstNode *parse_tokens(Token *toks);
void destroy_ast_tree(ScopeAstNode *node);

void print_ast(AstNode *node, int indent);

#endif
