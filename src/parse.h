#ifndef PARSE_H
#define PARSE_H

#include "core.h"
#include "tokenize.h"

typedef enum {
	AstNodeType_root,
	AstNodeType_ident,
	AstNodeType_decl
	/*AstNodeType_block,
	AstNodeType_literal,
	AstNodeType_uop,
	AstNodeType_biop,
	AstNodeType_ctrl_stmt,
	AstNodeType_call,
	AstNodeType_label,
	AstNodeType_comment
	*/
} AstNodeType;

struct AstNode;
typedef struct AstNode *AstNodePtr;

DECLARE_ARRAY(AstNodePtr)

typedef struct AstNode {
	AstNodeType type;
} AstNode;

typedef struct RootAstNode {
	AstNode b;
	Array(AstNodePtr) nodes;
} RootAstNode;

typedef struct IdentAstNode {
	AstNode b;
	const char *text_buf;
	int text_len;
} IdentAstNode;

typedef struct DeclAstNode {
	AstNode b;
	AstNode *type;
	IdentAstNode *ident;
	AstNode *value;
} DeclAstNode;


/* Created AST will have pointers to tokens */
RootAstNode *parse_tokens(Token *toks);
void destroy_ast_tree(RootAstNode *node);

void print_ast(AstNode *node, int indent);

#endif
