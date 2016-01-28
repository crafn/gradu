#ifndef QC_PARSE_H
#define QC_PARSE_H

#include "ast.h"
#include "tokenize.h"

int qc_biop_prec(QC_Token_Type type);
int qc_biop_assoc(QC_Token_Type type);

/* dont_reference_tokens: Allowing AST to reference tokens will give e.g. line number for each node */
/* @todo dont_reference_tokens -> self_contained (no token refs, no source refs) */
/* @todo Don't print error messages, return them */
QC_AST_Scope *qc_parse_tokens(QC_Token *toks, QC_Bool dont_reference_tokens);


/* Convenience functions */

/* `string` can be like "foo.bar.x = 123" or a whole program.
   Note that parsing pieces of a program doesn't necessarily result in
   a full or even correct AST because C is context sensitive. */
QC_AST_Node *qc_parse_string(const char *string);

#endif
