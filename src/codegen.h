#ifndef CODEGEN_H
#define CODEGEN_H

#include "core.h"
#include "parse.h"

Array(char) gen_c_code(AST_Scope *root);

#endif
