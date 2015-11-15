#ifndef BACKEND_C_H
#define BACKEND_C_H

#include "core.h"
#include "parse.h"

Array(char) gen_c_code(AST_Scope *root);

#endif
