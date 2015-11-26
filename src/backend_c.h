#ifndef BACKEND_C_H
#define BACKEND_C_H

#include "core.h"
#include "parse.h"

/* Utils for other c-like backends */
void lift_types_and_funcs_to_global_scope(AST_Scope *root);
void add_builtin_c_decls_to_global_scope(AST_Scope *root, bool func_decls);
void apply_c_operator_overloading(AST_Scope *root, bool convert_mat_expr);
/* Type name for builtin type */
void append_builtin_type_c_str(Array(char) *buf, Builtin_Type bt);
/* Function name for expression */
void append_expr_c_func_name(Array(char) *buf, AST_Node *expr);
void append_c_stdlib_includes(Array(char) *buf);
bool ast_to_c_str(Array(char) *buf, int indent, AST_Node *node);

Array(char) gen_c_code(AST_Scope *root);

#endif
