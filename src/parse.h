#ifndef QC_PARSE_H
#define QC_PARSE_H

#include "ast.h"
#include "tokenize.h"

int qc_biop_prec(QC_Token_Type type);
int qc_biop_assoc(QC_Token_Type type);

/* Created QC_AST will have pointers to tokens */
QC_AST_Scope *qc_parse_tokens(QC_Token *toks);

#endif
