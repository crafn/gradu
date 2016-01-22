#ifndef BACKEND_C_H
#define BACKEND_C_H

#include "core.h"
#include "parse.h"

AST_Var_Decl *is_device_field_member_decl(AST_Type_Decl *field_decl);

/* Utils for other c-like backends */
void lift_var_decls(AST_Scope *root);
void parallel_loops_to_ordinary(AST_Scope *root);
void lift_types_and_funcs_to_global_scope(AST_Scope *root);
void add_builtin_c_decls_to_global_scope(AST_Scope *root, bool cpu_device_impl);
void apply_c_operator_overloading(AST_Scope *root, bool convert_mat_expr);
/* Type name for builtin type */
void append_builtin_type_c_str(QC_Array(char) *buf, Builtin_Type bt);
/* Function name for expression */
void append_expr_c_func_name(QC_Array(char) *buf, AST_Node *expr);
void append_c_stdlib_includes(QC_Array(char) *buf);
bool ast_to_c_str(QC_Array(char) *buf, int indent, AST_Node *node);

/* @todo Flag determining C99 or C89 */
QC_Array(char) gen_c_code(AST_Scope *root);


#endif
