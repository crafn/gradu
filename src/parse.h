#ifndef PARSE_H
#define PARSE_H

#include "core.h"
#include "tokenize.h"

typedef enum {
	AST_scope,
	AST_ident,
	AST_type,
	AST_type_decl,
	AST_var_decl,
	AST_func_decl,
	AST_literal,
	AST_biop,
	AST_control,
	AST_call
		/*
	AST_uop,
	AST_label,
	*/
} AST_Node_Type;

struct AST_Node;
struct AST_Decl;
typedef struct AST_Node *AST_Node_Ptr;
typedef struct Token *Token_Ptr;

DECLARE_ARRAY(AST_Node_Ptr)
DECLARE_ARRAY(Token_Ptr)

/* Base "class" for every AST node type */
typedef struct AST_Node {
	AST_Node_Type type;

	/* Information for human-readable output */

	Token *begin_tok;
	/* Comments on the previous line(s) (like this comment) */
	Array(Token_Ptr) pre_comments;
	Array(Token_Ptr) post_comments; /* On the same line (like this comment) */
} AST_Node;

/* { ... } */
typedef struct AST_Scope {
	AST_Node b;
	Array(AST_Node_Ptr) nodes;
	bool is_root;
} AST_Scope;

/* Identifier */
typedef struct AST_Ident {
	AST_Node b;
	const char *text_buf;
	int text_len;

	struct AST_Node *decl; /* Not owned */
} AST_Ident;

struct AST_Type_Decl;

typedef struct AST_Type {
	AST_Node b;
	/* Pointer to 'struct Foo { ... }' in type 'Foo **' */
	/* Decided to use type decl directly instead of identifiers, because
	 * type names can be multiple keywords long in source code (long int etc.)*/
	struct AST_Type_Decl *base_type_decl; /* Not owned */
	int ptr_depth;
} AST_Type;

typedef struct Builtin_Type {
	bool is_void;
	bool is_integer;
	bool is_char; /* int8_t != char */
	bool is_float;
	int bitness;

	bool is_unsigned;

	/* When adding members, remember to update parse_type_and_ident! */

	/*
	bool is_scalar : 1;
	bool is_matrix;

	bool is_field;
	*/
} Builtin_Type;

/* Type declaration / definition */
typedef struct AST_Type_Decl {
	AST_Node b;
	AST_Ident *ident;
	AST_Scope *body;

	bool is_builtin; /* void, int, char etc. */
	Builtin_Type builtin_type;
} AST_Type_Decl;

/* Variable declaration / definition */
typedef struct AST_Var_Decl {
	AST_Node b;

	AST_Type *type;
	AST_Ident *ident;
	AST_Node *value;
} AST_Var_Decl;

typedef AST_Var_Decl *AST_Var_Decl_Ptr;
DECLARE_ARRAY(AST_Var_Decl_Ptr)

/* Function declaration / definition */
typedef struct AST_Func_Decl {
	AST_Node b;
	AST_Type *return_type;
	AST_Ident *ident;
	Array(AST_Var_Decl_Ptr) params;
	AST_Scope *body;
} AST_Func_Decl;

typedef enum {
	Literal_int,
	Literal_string
} Literal_Type;

/* Number / string literal */
typedef struct AST_Literal {
	AST_Node b;
	Literal_Type type;
	union {
		/* @todo Different integer sizes etc */
		int integer;
		struct {
			const char *buf;
			int len;
		} string;
	} value;
} AST_Literal;

/* Binary operation */
typedef struct AST_Biop {
	AST_Node b;
	Token_Type type;
	AST_Node *lhs;
	AST_Node *rhs;
} AST_Biop;

/* return, goto, continue, break */
typedef struct AST_Control {
	AST_Node b;
	Token_Type type;
	/* For return and goto */
	AST_Node *value;
} AST_Control;

/* Function call */
typedef struct AST_Call {
	AST_Node b;
	AST_Ident *ident;
	Array(AST_Node_Ptr) args;
} AST_Call;

/* Usage: CASTED_NODE(AST_Ident, ident, generic_node); printf("%c", ident->text_buf[0]); */
#define CASTED_NODE(type, name, assign) \
	type *name = (type*)assign
#define AST_BASE(node) (&(node)->b)

/* Created AST will have pointers to tokens */
AST_Scope *parse_tokens(Token *toks);

AST_Scope *create_ast_tree();
void destroy_ast_tree(AST_Scope *node);

AST_Scope *create_scope_node();
AST_Ident *create_ident_node();
AST_Type *create_type_node();
AST_Type_Decl *create_type_decl_node();
AST_Var_Decl *create_var_decl_node();
AST_Func_Decl *create_func_decl_node();
AST_Literal *create_literal_node();
AST_Biop *create_biop_node();
AST_Control *create_control_node();
AST_Call *create_call_node();

void copy_scope_node(AST_Scope *copy, AST_Scope *scope, AST_Node **subnodes, int subnode_count);
void copy_ident_node(AST_Ident *copy, AST_Ident *ident, AST_Node *ref_to_decl);
void copy_type_node(AST_Type *copy, AST_Type *type, AST_Node *ref_to_base_type_decl);
void copy_type_decl_node(AST_Type_Decl *copy, AST_Type_Decl *decl, AST_Node *ident, AST_Node *body);
void copy_var_decl_node(AST_Var_Decl *copy, AST_Var_Decl *decl, AST_Node *type, AST_Node *ident, AST_Node *value);
void copy_func_decl_node(AST_Func_Decl *copy, AST_Func_Decl *decl, AST_Node *type, AST_Node *ident, AST_Node **params, int param_count, AST_Node *body);
void copy_literal_node(AST_Literal *copy, AST_Literal *literal);
void copy_biop_node(AST_Biop *copy, AST_Biop *biop, AST_Node *lhs, AST_Node *rhs);
void copy_control_node(AST_Control *copy, AST_Control *control, AST_Node *value);
void copy_call_node(AST_Call *copy, AST_Call *call, AST_Node *ident, AST_Node **args, int arg_count);

/* Recursive */
void destroy_node(AST_Node *node);

void print_ast(AST_Node *node, int indent);

#endif
