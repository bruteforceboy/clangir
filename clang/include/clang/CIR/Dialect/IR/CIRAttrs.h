//===- CIRAttrs.h - MLIR CIR Attrs ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRATTRS_H
#define CLANG_CIR_DIALECT_IR_CIRATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIRTypeInterfaces.h"

namespace cir {
inline constexpr uint32_t DefaultGlobalCtorDtorPriority = 65535;
} // namespace cir

//===----------------------------------------------------------------------===//
// CIR Dialect Attrs
//===----------------------------------------------------------------------===//

namespace clang {
class FunctionDecl;
class RecordDecl;
class VarDecl;
} // namespace clang

namespace cir {
class ArrayType;
class BoolType;
class ComplexType;
class DataMemberType;
class IntType;
class MethodType;
class PointerType;
class RecordType;
class VectorType;
} // namespace cir

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIRATTRS_H
