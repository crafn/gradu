#ifndef PARSE_H
#define PARSE_H

#include "ast.h"
#include "tokenize.h"

int biop_prec(QC_Token_Type type);
int biop_assoc(QC_Token_Type type);

/* Created QC_AST will have pointers to tokens */
QC_AST_Scope *parse_tokens(QC_Token *toks);

#endif
