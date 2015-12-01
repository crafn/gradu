#ifndef AST_H
#define AST_H

#include "core.h"
#include "tokenize.h"

typedef enum {
	AST_none,
	AST_scope,
	AST_ident,
	AST_type,
	AST_type_decl,
	AST_var_decl,
	AST_func_decl,
	AST_literal,
	AST_biop,
	AST_control,
	AST_call,
	AST_access,
	AST_cond,
	AST_loop,
	AST_cast,
	AST_typedef,
	AST_parallel
} AST_Node_Type;

struct AST_Node;
struct AST_Decl;
typedef struct AST_Node *AST_Node_Ptr;
typedef struct Token *Token_Ptr;

DECLARE_ARRAY(AST_Node_Ptr)
DECLARE_ARRAY(Token_Ptr)
DECLARE_HASH_TABLE(AST_Node_Ptr, AST_Node_Ptr)
static U32 hash(AST_Node_Ptr)(AST_Node_Ptr node) { return hash(Void_Ptr)(node); }

/* Base "class" for every AST node type */
typedef struct AST_Node {
	AST_Node_Type type;

	/* Information for human-readable output */

	/* @todo Inline struct with needed info. Then generated code can have vertical spacing etc. */
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
	/* @todo Change to Array(char) */
	Array(char) text; /* NULL-terminated */

	struct AST_Node *decl; /* Not owned */
} AST_Ident;

struct AST_Type_Decl;
struct AST_Typedef;

typedef struct AST_Type {
	AST_Node b;
	/* Pointer to 'struct Foo { ... }' in type 'Foo **' */
	/* Decided to use type decl directly instead of identifiers, because
	 * type names can be multiple keywords long in source code (long int etc.)*/
	struct AST_Type_Decl *base_type_decl; /* Not owned */
	struct AST_Typedef *base_typedef; /* Not owned. Records the chain of typedefs for backend. */
	int ptr_depth;
	/* @todo 2-dimensional arrays, pointers to arrays, ... (?) */
	int array_size; /* 0 for no array */
	bool is_const; /* Just to propagate consts to output */

	/* When adding members, remember to update type_node_equals! */
} AST_Type;
bool type_node_equals(AST_Type a, AST_Type b);

typedef struct Builtin_Type {
	bool is_void;
	bool is_integer;
	bool is_char; /* int8_t != char */
	bool is_float;
	int bitness; /* Zero for "not explicitly specified" */
	bool is_unsigned;

	bool is_matrix;
#define MAX_MATRIX_RANK 10 /* Could be made dynamic */
	int matrix_rank;
	int matrix_dim[MAX_MATRIX_RANK];

	bool is_field;
	int field_dim;

	/* When adding members, remember to update builtin_type_equals! */
} Builtin_Type;
bool builtin_type_equals(Builtin_Type a, Builtin_Type b);

/* Type declaration / definition */
typedef struct AST_Type_Decl {
	AST_Node b;
	AST_Ident *ident;
	AST_Scope *body;

	/* 'body' and 'ident' are NULL for builtin types */
	bool is_builtin; /* void, int, char etc. */
	Builtin_Type builtin_type;
	struct AST_Type_Decl *builtin_sub_type_decl; /* Not owner. Matrix/scalar for a field. Scalar for a matrix. */
	struct AST_Type_Decl *builtin_concrete_decl; /* Not owned. Backend can use this to point generated types. */
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
	bool ellipsis;
	AST_Scope *body;

	bool is_builtin; /* Field allocation and deallocation functions */
	struct AST_Func_Decl *builtin_concrete_decl; /* Not owned. Backend can use this to point generated types. */
} AST_Func_Decl;

typedef enum {
	Literal_int,
	Literal_float,
	Literal_string,
	Literal_null
} Literal_Type;

/* Number / string literal */
typedef struct AST_Literal {
	AST_Node b;
	Literal_Type type;
	union {
		/* @todo Different integer sizes etc */
		int integer;
		double floating;
		Buf_Str string;
	} value;

	struct AST_Type_Decl *base_type_decl; /* Not owned. 'expr_type' needs this. */
} AST_Literal;

/* Binary operation */
/* @todo Rename to AST_Expr */
typedef struct AST_Biop {
	AST_Node b;
	Token_Type type;
	AST_Node *lhs; /* NULL for unary operations like '-5' */
	AST_Node *rhs; /* NULL for unary operations like 'i++' */

	bool is_top_level; /* This is not part of another expression */
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

/* Variable/array/member access */
typedef struct AST_Access {
	AST_Node b;

	/* base -- base.arg -- base->arg -- base[arg] -- some_matrix(arg1, arg2)*/
	AST_Node *base;
	Array(AST_Node_Ptr) args;
	/* 'base.sub' -> Access(Access(Ident(base)), Ident(sub))
	 * 'base' is wrapped in an extra Access, because then '(base + 0)->sub' and '(base)->sub' and 'base.sub'
	 * are handled uniformly that way (every expression has two Access nodes) */

	bool is_member_access;
	bool is_element_access; /* Matrix or field element access */
	bool is_array_access;

	bool implicit_deref; /* 'a->b' or 'field_ptr(1, 2)' */

	/* @todo Field access etc. */
} AST_Access;

/* if */
typedef struct AST_Cond {
	AST_Node b;

	AST_Node *expr;
	AST_Scope *body;
	bool implicit_scope; /* Original source had no { } */

	/* Must be AST_Scope or AST_Cond or NULL */
	AST_Node *after_else;
} AST_Cond;

/* while/for/do-while */
typedef struct AST_Loop {
	AST_Node b;

	/* while-loop has only 'cond' */
	AST_Node *init;
	AST_Node *cond;
	AST_Node *incr;

	AST_Scope *body;
	bool implicit_scope; /* Original source had no { } */

	/* @todo do-while */
} AST_Loop;

typedef struct AST_Cast {
	AST_Node b;

	AST_Type *type;
	AST_Node *target;
} AST_Cast;

typedef struct AST_Typedef {
	AST_Node b;

	AST_Type *type;
	AST_Ident *ident;
} AST_Typedef;

/* for_field */
typedef struct AST_Parallel {
	AST_Node b;

	AST_Node *output;
	AST_Node *input;

	AST_Scope *body;

	int dim; /* Essentially output field dimension */
} AST_Parallel;

/* Usage: CASTED_NODE(AST_Ident, ident, generic_node); printf("%c", ident->text_buf[0]); */
#define CASTED_NODE(type, name, assign) \
	type *name = (type*)assign
#define AST_BASE(node) (&(node)->b)

AST_Scope *create_ast();
void destroy_ast(AST_Scope *node);
AST_Node *copy_ast(AST_Node *node);
AST_Node *shallow_copy_ast(AST_Node *node);
void move_ast(AST_Scope *dst, AST_Scope *src);

AST_Node *create_ast_node(AST_Node_Type type);
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
AST_Access *create_access_node();
AST_Cond *create_cond_node();
AST_Loop *create_loop_node();
AST_Cast *create_cast_node();
AST_Typedef *create_typedef_node();
AST_Parallel *create_parallel_node();

/* Copies only stuff in AST_Node structure. Useful for copying comments to another node, for example. */
void copy_ast_node_base(AST_Node *dst, AST_Node *src);
/* Calls a specific copy_*_node */
/* 'subnodes' and 'refnodes' should contain same nodes as a specific copy_*_node */
void copy_ast_node(AST_Node *copy, AST_Node *node, AST_Node **subnodes, int subnode_count, AST_Node **refnodes, int refnode_count);
void shallow_copy_ast_node(AST_Node *copy, AST_Node* node);
/* Copy and source nodes can be the same in copying functions. This just updates the subnodes. */
/* First param: destination
 * Second param: source
 * Subnode params: new subnodes for destination
 * Refnode params: new refnodes for destination */
void copy_scope_node(AST_Scope *copy, AST_Scope *scope, AST_Node **subnodes, int subnode_count);
void copy_ident_node(AST_Ident *copy, AST_Ident *ident, AST_Node *ref_to_decl);
void copy_type_node(AST_Type *copy, AST_Type *type, AST_Node *ref_to_base_type_decl, AST_Node *ref_to_base_typedef);
void copy_type_decl_node(AST_Type_Decl *copy, AST_Type_Decl *decl, AST_Node *ident, AST_Node *body, AST_Node *builtin_sub_decl_ref, AST_Node *backend_decl_ref);
void copy_var_decl_node(AST_Var_Decl *copy, AST_Var_Decl *decl, AST_Node *type, AST_Node *ident, AST_Node *value);
void copy_func_decl_node(AST_Func_Decl *copy, AST_Func_Decl *decl, AST_Node *type, AST_Node *ident, AST_Node *body, AST_Node **params, int param_count, AST_Node *backend_decl_ref);
void copy_literal_node(AST_Literal *copy, AST_Literal *literal, AST_Node *type_decl_ref);
void copy_biop_node(AST_Biop *copy, AST_Biop *biop, AST_Node *lhs, AST_Node *rhs);
void copy_control_node(AST_Control *copy, AST_Control *control, AST_Node *value);
void copy_call_node(AST_Call *copy, AST_Call *call, AST_Node *ident, AST_Node **args, int arg_count);
void copy_access_node(AST_Access *copy, AST_Access *access, AST_Node *base, AST_Node **args, int arg_count);
void copy_cond_node(AST_Cond *copy, AST_Cond *cond, AST_Node *expr, AST_Node *body, AST_Node *after_else);
void copy_loop_node(AST_Loop *copy, AST_Loop *loop, AST_Node *init, AST_Node *cond, AST_Node *incr, AST_Node *body);
void copy_cast_node(AST_Cast *copy, AST_Cast *cast, AST_Node *type, AST_Node *target);
void copy_typedef_node(AST_Typedef *copy, AST_Typedef *def, AST_Node *type, AST_Node *ident);
void copy_parallel_node(AST_Parallel *copy, AST_Parallel *parallel, AST_Node *output, AST_Node *input, AST_Node *body);

/* Recursive */
/* @todo Use destroy_ast for this */
void destroy_node(AST_Node *node);
/* Use this to destroy the original node after shallow_copy_ast_node.
 * Doesn't destroy any owned nodes. */
void shallow_destroy_node(AST_Node *node);

/* Evaluation */
/* Don't destroy nodes returned by evaluation, they are not constructed by create_*, just containers of data */

bool expr_type(AST_Type *ret, AST_Node *expr);
bool eval_const_expr(AST_Literal *ret, AST_Node *expr);

bool is_decl(AST_Node *node);
AST_Ident *decl_ident(AST_Node *node);


/* AST traversing utils */

typedef struct AST_Parent_Map {
	Hash_Table(AST_Node_Ptr, AST_Node_Ptr) table; /* Use find_- and set_parent_node to access */
	Array(AST_Node_Ptr) builtin_decls; /* Builtin decls are separate, because they're created during parsing */
	/* @todo That ^ could maybe be removed. */
} AST_Parent_Map;

AST_Parent_Map create_parent_map(AST_Node *root);
void destroy_parent_map(AST_Parent_Map *map);
AST_Node *find_parent_node(AST_Parent_Map *map, AST_Node *node);
void set_parent_node(AST_Parent_Map *map, AST_Node *sub, AST_Node *parent);

AST_Ident *resolve_ident(AST_Parent_Map *map, AST_Ident *ident);
/* Resolves call to specific overload */
AST_Call *resolve_call(AST_Parent_Map *map, AST_Call *call, AST_Type *return_type_hint);
/* Resove all unresolved things in AST. Call this after inserting unresolved nodes into AST. */
void resolve_ast(AST_Scope *root);


void push_immediate_subnodes(Array(AST_Node_Ptr) *ret, AST_Node *node);
/* Gathers immediate referenced (not owned) nodes */
void push_immediate_refnodes(Array(AST_Node_Ptr) *ret, AST_Node *node);
/* Gathers the whole subnode tree to array */
void push_subnodes(Array(AST_Node_Ptr) *ret, AST_Node *node, bool push_before_recursing);
/* Rewrites nodes in tree, old_nodes[i] -> new_nodes[i]
 * Doesn't free or allocate any nodes.
 * Doesn't recurse into old_nodes. They can be dangling.
 * Is recursive, so if some new_nodes[i] contain old_nodes[k], it will also be replaced. */
AST_Node *replace_nodes_in_ast(AST_Node *node, AST_Node **old_nodes, AST_Node **new_nodes, int node_count);

/* Innermost first */
void find_subnodes_of_type(Array(AST_Node_Ptr) *ret, AST_Node_Type type, AST_Node *node);

/* Debug */
void print_ast(AST_Node *node, int indent);

/* Convenience functions */

AST_Ident *create_ident_with_text(AST_Node *decl, const char *fmt, ...);
AST_Var_Decl *create_simple_var_decl(AST_Type_Decl *type_decl, const char *ident);
AST_Var_Decl *create_var_decl(AST_Type_Decl *type_decl, AST_Ident *ident, AST_Node *value);
AST_Type_Decl *find_builtin_type_decl(Builtin_Type bt, AST_Scope *root);
AST_Literal *create_integer_literal(int value, AST_Scope *root);
AST_Call *create_call_1(AST_Ident *ident, AST_Node *arg);
AST_Call *create_call_2(AST_Ident *ident, AST_Node *arg1, AST_Node *arg2);
AST_Control *create_return(AST_Node *expr);
AST_Biop *create_sizeof(AST_Node *expr);
AST_Biop *create_deref(AST_Node *expr);
AST_Biop *create_biop(Token_Type type, AST_Node *lhs, AST_Node *rhs);
AST_Biop *create_assign(AST_Node *lhs, AST_Node *rhs);
AST_Biop *create_mul(AST_Node *lhs, AST_Node *rhs);
AST_Biop *create_less_than(AST_Node *lhs, AST_Node *rhs);
AST_Biop *create_pre_increment(AST_Node *expr);
AST_Cast *create_cast(AST_Type *type, AST_Node *target);
AST_Type *create_builtin_type(Builtin_Type bt, int ptr_depth, AST_Scope *root);
AST_Type *copy_and_modify_type(AST_Type *type, int delta_ptr_depth);
AST_Type *create_simple_type(AST_Type_Decl *type_decl);
AST_Loop *create_for_loop(AST_Var_Decl *index, AST_Node *max_expr, AST_Scope *body);
AST_Node *try_create_access(AST_Node *node);
AST_Access *create_element_access_1(AST_Node *base, AST_Node *arg);
AST_Scope *create_scope_1(AST_Node *expr);

Builtin_Type void_builtin_type();
Builtin_Type int_builtin_type();
Builtin_Type float_builtin_type();
Builtin_Type char_builtin_type();

/* elem[0] chainop elem[1] */
AST_Node *create_chained_expr(AST_Node **elems, int elem_count, Token_Type chainop);

/* (lhs[0] biop rhs[0]) chainop (lhs[1] biop rhs[1]) */
AST_Node *create_chained_expr_2(AST_Node **lhs_elems, AST_Node **rhs_elems, int elem_count, Token_Type biop, Token_Type chainop);

#endif
