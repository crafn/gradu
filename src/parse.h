#ifndef PARSE_H
#define PARSE_H

#include "core.h"
#include "tokenize.h"

typedef enum {
	AST_scope,
	AST_ident,
	AST_decl,
	AST_literal,
	AST_biop,
	AST_control
		/*
	AST_uop,
	AST_call,
	AST_label,
	*/
} AST_Node_Type;

struct AST_Node;
struct AST_Decl;
typedef struct AST_Node *AST_Node_Ptr;
typedef struct Token *Token_Ptr;

DECLARE_ARRAY(AST_Node_Ptr)
DECLARE_ARRAY(Token_Ptr)

typedef struct AST_Node {
	AST_Node_Type type;

	/* Information for human-readable output */

	Token *begin_tok;
	/* Comments on the previous line(s) (like this comment) */
	Array(Token_Ptr) pre_comments;
	Array(Token_Ptr) post_comments; /* On the same line (like this comment) */
} AST_Node;

typedef struct AST_Scope {
	AST_Node b;
	Array(AST_Node_Ptr) nodes;
	bool is_root;
} AST_Scope;

typedef struct AST_Ident {
	AST_Node b;
	const char *text_buf;
	int text_len;

	struct AST_Decl *decl; /* Pointer to node which declares this identifier */
} AST_Ident;

typedef struct AST_Decl {
	AST_Node b;
	AST_Node *type;
	AST_Ident *ident;
	AST_Node *value;

	bool is_type_decl;
	bool is_var_decl;
	bool is_func_decl;
} AST_Decl;

typedef enum {
	Literal_int
} Literal_Type;

typedef struct AST_Literal {
	AST_Node b;
	Literal_Type type;
	union {
		/* @todo Different integer sizes etc */
		int integer;
	} value;
} AST_Literal;

typedef struct AST_Biop {
	AST_Node b;
	Token_Type type;
	AST_Node *lhs;
	AST_Node *rhs;
} AST_Biop;

typedef struct AST_Control {
	AST_Node b;
	Token_Type type;
	/* For return and goto */
	AST_Node *value;
} AST_Control;

/* Usage: CASTED_NODE(AST_Ident, ident, generic_node); printf("%c", ident->text_buf[0]); */
#define CASTED_NODE(type, name, assign) \
	type *name = (type*)assign
#define AST_BASE(node) (&(node)->b)

/* Created AST will have pointers to tokens */
AST_Scope *parse_tokens(Token *toks);

AST_Scope *create_ast_tree();
void destroy_ast_tree(AST_Scope *node);

AST_Scope *create_scope_node();
AST_Ident *create_ident_node(Token *tok);
AST_Decl *create_decl_node();
AST_Literal *create_literal_node();
AST_Biop *create_biop_node(Token_Type type, AST_Node *lhs, AST_Node *rhs);

AST_Scope *copy_scope_node(AST_Scope *scope, AST_Node **subnodes, int subnode_count);
AST_Ident *copy_ident_node(AST_Ident *ident);
AST_Decl *copy_decl_node(AST_Decl *decl, AST_Node *type, AST_Node *ident, AST_Node *value);
AST_Literal *copy_literal_node(AST_Literal *literal);
AST_Biop *copy_biop_node(AST_Biop *biop, AST_Node *lhs, AST_Node *rhs);
AST_Control *copy_control_node(AST_Control *control, AST_Node *value);

void destroy_node(AST_Node *node);

void print_ast(AST_Node *node, int indent);

#endif
