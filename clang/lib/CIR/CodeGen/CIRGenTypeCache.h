//===--- CIRGenTypeCache.h - Commonly used LLVM types and info -*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This structure provides a set of common types useful during CIR emission.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGENTYPECACHE_H
#define LLVM_CLANG_LIB_CIR_CODEGENTYPECACHE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "clang/AST/CharUnits.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

namespace clang::CIRGen {

/// This structure provides a set of types that are commonly used
/// during IR emission. It's initialized once in CodeGenModule's
/// constructor and then copied around into new CIRGenFunction's.
struct CIRGenTypeCache {
  CIRGenTypeCache() {}

  /// void
  cir::VoidType VoidTy;
  // char, int, short, long, __int128
  cir::IntType SInt8Ty, SInt16Ty, SInt32Ty, SInt64Ty, SInt128Ty;
  // usigned char, unsigned, unsigned short, unsigned long, unsigned __int128
  cir::IntType UInt8Ty, UInt16Ty, UInt32Ty, UInt64Ty, UInt128Ty;
  /// half, bfloat, float, double, fp80
  cir::FP16Type FP16Ty;
  cir::BF16Type BFloat16Ty;
  cir::SingleType FloatTy;
  cir::DoubleType DoubleTy;
  cir::FP80Type FP80Ty;
  cir::FP128Type FP128Ty;

  /// int
  mlir::Type UIntTy;

  /// char
  mlir::Type UCharTy;

  /// intptr_t, size_t, and ptrdiff_t, which we assume are the same size.
  union {
    mlir::Type UIntPtrTy;
    mlir::Type SizeTy;
  };

  mlir::Type PtrDiffTy;

  /// void* in address space 0
  cir::PointerType VoidPtrTy;
  cir::PointerType UInt8PtrTy;

  /// void** in address space 0
  cir::PointerType VoidPtrPtrTy;
  cir::PointerType UInt8PtrPtrTy;

  /// void* in alloca address space
  cir::PointerType AllocaVoidPtrTy;
  cir::PointerType AllocaInt8PtrTy;

  /// void* in default globals address space
  //   union {
  //     cir::PointerType GlobalsVoidPtrTy;
  //     cir::PointerType GlobalsInt8PtrTy;
  //   };

  /// void* in the address space for constant globals
  //   cir::PointerType ConstGlobalsPtrTy;

  /// The size and alignment of the builtin C type 'int'.  This comes
  /// up enough in various ABI lowering tasks to be worth pre-computing.
  //   union {
  //     unsigned char IntSizeInBytes;
  //     unsigned char IntAlignInBytes;
  //   };
  //   clang::CharUnits getIntSize() const {
  //     return clang::CharUnits::fromQuantity(IntSizeInBytes);
  //   }
  //   clang::CharUnits getIntAlign() const {
  //     return clang::CharUnits::fromQuantity(IntAlignInBytes);
  //   }

  /// The width of a pointer into the generic address space.
  //   unsigned char PointerWidthInBits;

  /// The size and alignment of a pointer into the generic address space.
  union {
    unsigned char PointerAlignInBytes;
    unsigned char PointerSizeInBytes;
  };

  /// The size and alignment of size_t.
  union {
    unsigned char SizeSizeInBytes; // sizeof(size_t)
    unsigned char SizeAlignInBytes;
  };

  cir::AddressSpace CIRAllocaAddressSpace;

  clang::CharUnits getSizeSize() const {
    return clang::CharUnits::fromQuantity(SizeSizeInBytes);
  }
  clang::CharUnits getSizeAlign() const {
    return clang::CharUnits::fromQuantity(SizeAlignInBytes);
  }
  clang::CharUnits getPointerSize() const {
    return clang::CharUnits::fromQuantity(PointerSizeInBytes);
  }
  clang::CharUnits getPointerAlign() const {
    return clang::CharUnits::fromQuantity(PointerAlignInBytes);
  }

  cir::AddressSpace getCIRAllocaAddressSpace() const {
    return CIRAllocaAddressSpace;
  }
};

} // namespace clang::CIRGen

#endif
