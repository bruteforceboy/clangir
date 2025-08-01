//====- LowerToLLVM.cpp - Lowering from CIR to LLVMIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to LLVMIR.
//
//===----------------------------------------------------------------------===//
#include "LowerToLLVM.h"
#include "LoweringHelpers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <cstdint>
#include <deque>
#include <optional>
#include <set>

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

namespace {

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(mlir::Region &region,
                        mlir::function_ref<void(mlir::Operation *)> callback) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<Ops...>(op))
      return mlir::WalkResult::skip();
    callback(op);
    return mlir::WalkResult::advance();
  });
}

/// Convert from a CIR comparison kind to an LLVM IR integral comparison kind.
mlir::LLVM::ICmpPredicate convertCmpKindToICmpPredicate(cir::CmpOpKind kind,
                                                        bool isSigned) {
  using CIR = cir::CmpOpKind;
  using LLVMICmp = mlir::LLVM::ICmpPredicate;
  switch (kind) {
  case CIR::eq:
    return LLVMICmp::eq;
  case CIR::ne:
    return LLVMICmp::ne;
  case CIR::lt:
    return (isSigned ? LLVMICmp::slt : LLVMICmp::ult);
  case CIR::le:
    return (isSigned ? LLVMICmp::sle : LLVMICmp::ule);
  case CIR::gt:
    return (isSigned ? LLVMICmp::sgt : LLVMICmp::ugt);
  case CIR::ge:
    return (isSigned ? LLVMICmp::sge : LLVMICmp::uge);
  }
  llvm_unreachable("Unknown CmpOpKind");
}

/// Convert from a CIR comparison kind to an LLVM IR floating-point comparison
/// kind.
mlir::LLVM::FCmpPredicate convertCmpKindToFCmpPredicate(cir::CmpOpKind kind) {
  using CIR = cir::CmpOpKind;
  using LLVMFCmp = mlir::LLVM::FCmpPredicate;
  switch (kind) {
  case CIR::eq:
    return LLVMFCmp::oeq;
  case CIR::ne:
    return LLVMFCmp::une;
  case CIR::lt:
    return LLVMFCmp::olt;
  case CIR::le:
    return LLVMFCmp::ole;
  case CIR::gt:
    return LLVMFCmp::ogt;
  case CIR::ge:
    return LLVMFCmp::oge;
  }
  llvm_unreachable("Unknown CmpOpKind");
}

/// If the given type is a vector type, return the vector's element type.
/// Otherwise return the given type unchanged.
mlir::Type elementTypeIfVector(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
      .Case<cir::VectorType, mlir::VectorType>(
          [](auto p) { return p.getElementType(); })
      .Default([](mlir::Type p) { return p; });
}

mlir::LLVM::Visibility
lowerCIRVisibilityToLLVMVisibility(cir::VisibilityKind visibilityKind) {
  switch (visibilityKind) {
  case cir::VisibilityKind::Default:
    return ::mlir::LLVM::Visibility::Default;
  case cir::VisibilityKind::Hidden:
    return ::mlir::LLVM::Visibility::Hidden;
  case cir::VisibilityKind::Protected:
    return ::mlir::LLVM::Visibility::Protected;
  }
}

// Make sure the LLVM function we are about to create a call for actually
// exists, if not create one. Returns a function
void getOrCreateLLVMFuncOp(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Operation *srcOp, llvm::StringRef fnName,
                           mlir::Type fnTy) {
  auto modOp = srcOp->getParentOfType<mlir::ModuleOp>();
  auto enclosingFnOp = srcOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  auto *sourceSymbol = mlir::SymbolTable::lookupSymbolIn(modOp, fnName);
  if (!sourceSymbol) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(enclosingFnOp);
    rewriter.create<mlir::LLVM::LLVMFuncOp>(srcOp->getLoc(), fnName, fnTy);
  }
}

static constexpr StringRef llvmMetadataSectionName = "llvm.metadata";

// Create a string global for annotation related string.
mlir::LLVM::GlobalOp
getAnnotationStringGlobal(mlir::StringAttr strAttr, mlir::ModuleOp &module,
                          llvm::StringMap<mlir::LLVM::GlobalOp> &globalsMap,
                          mlir::OpBuilder &globalVarBuilder,
                          mlir::Location &loc, bool isArg = false) {
  llvm::StringRef str = strAttr.getValue();
  if (!globalsMap.contains(str)) {
    auto llvmStrTy = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(module.getContext(), 8), str.size() + 1);
    auto strGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
        loc, llvmStrTy,
        /*isConstant=*/true, mlir::LLVM::Linkage::Private,
        ".str" +
            (globalsMap.empty() ? ""
                                : "." + std::to_string(globalsMap.size())) +
            ".annotation" + (isArg ? ".arg" : ""),
        mlir::StringAttr::get(module.getContext(), std::string(str) + '\0'),
        /*alignment=*/isArg ? 1 : 0);
    if (!isArg)
      strGlobalOp.setSection(llvmMetadataSectionName);
    strGlobalOp.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
    strGlobalOp.setDsoLocal(true);
    globalsMap[str] = strGlobalOp;
  }
  return globalsMap[str];
}

mlir::LLVM::GlobalOp getOrCreateAnnotationArgsVar(
    mlir::Location &loc, mlir::ModuleOp &module,
    mlir::OpBuilder &globalVarBuilder,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap,
    mlir::ArrayAttr argsAttr) {
  if (argsVarMap.contains(argsAttr))
    return argsVarMap[argsAttr];

  mlir::LLVM::LLVMPointerType annoPtrTy =
      mlir::LLVM::LLVMPointerType::get(globalVarBuilder.getContext());
  llvm::SmallVector<mlir::Type> argStrutFldTypes;
  llvm::SmallVector<mlir::Value> argStrutFields;
  for (mlir::Attribute arg : argsAttr) {
    if (auto strArgAttr = mlir::dyn_cast<mlir::StringAttr>(arg)) {
      // Call getAnnotationStringGlobal here to make sure
      // have a global for this string before
      // creation of the args var.
      getAnnotationStringGlobal(strArgAttr, module, argStringGlobalsMap,
                                globalVarBuilder, loc, true);
      // This will become a ptr to the global string
      argStrutFldTypes.push_back(annoPtrTy);
    } else if (auto intArgAttr = mlir::dyn_cast<mlir::IntegerAttr>(arg)) {
      argStrutFldTypes.push_back(intArgAttr.getType());
    } else {
      llvm_unreachable("Unsupported annotation arg type");
    }
  }

  mlir::LLVM::LLVMStructType argsStructTy =
      mlir::LLVM::LLVMStructType::getLiteral(globalVarBuilder.getContext(),
                                             argStrutFldTypes);
  auto argsGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
      loc, argsStructTy, true, mlir::LLVM::Linkage::Private,
      ".args" +
          (argsVarMap.empty() ? "" : "." + std::to_string(argsVarMap.size())) +
          ".annotation",
      mlir::Attribute());
  argsGlobalOp.setSection(llvmMetadataSectionName);
  argsGlobalOp.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
  argsGlobalOp.setDsoLocal(true);

  // Create the initializer for this args global
  argsGlobalOp.getRegion().push_back(new mlir::Block());
  mlir::OpBuilder argsInitBuilder(module.getContext());
  argsInitBuilder.setInsertionPointToEnd(argsGlobalOp.getInitializerBlock());

  mlir::Value argsStructInit =
      argsInitBuilder.create<mlir::LLVM::UndefOp>(loc, argsStructTy);
  int idx = 0;
  for (mlir::Attribute arg : argsAttr) {
    if (auto strArgAttr = mlir::dyn_cast<mlir::StringAttr>(arg)) {
      // This would be simply return with existing map entry value
      // from argStringGlobalsMap as string global is already
      // created in the previous loop.
      mlir::LLVM::GlobalOp argStrVar = getAnnotationStringGlobal(
          strArgAttr, module, argStringGlobalsMap, globalVarBuilder, loc, true);
      auto argStrVarAddr = argsInitBuilder.create<mlir::LLVM::AddressOfOp>(
          loc, annoPtrTy, argStrVar.getSymName());
      argsStructInit = argsInitBuilder.create<mlir::LLVM::InsertValueOp>(
          loc, argsStructInit, argStrVarAddr, idx++);
    } else if (auto intArgAttr = mlir::dyn_cast<mlir::IntegerAttr>(arg)) {
      auto intArgFld = argsInitBuilder.create<mlir::LLVM::ConstantOp>(
          loc, intArgAttr.getType(), intArgAttr.getValue());
      argsStructInit = argsInitBuilder.create<mlir::LLVM::InsertValueOp>(
          loc, argsStructInit, intArgFld, idx++);
    } else {
      llvm_unreachable("Unsupported annotation arg type");
    }
  }
  argsInitBuilder.create<mlir::LLVM::ReturnOp>(loc, argsStructInit);
  argsVarMap[argsAttr] = argsGlobalOp;
  return argsGlobalOp;
}

/// Lower an annotation value to a series of LLVM globals, `outVals` contains
/// all values which are either used to build other globals or for intrisic call
/// arguments.
void lowerAnnotationValue(
    mlir::Location &localLoc, mlir::Location annotLoc,
    cir::AnnotationAttr annotation, mlir::ModuleOp &module,
    mlir::OpBuilder &varInitBuilder, mlir::OpBuilder &globalVarBuilder,
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap,
    SmallVectorImpl<mlir::Value> &outVals) {
  mlir::LLVM::LLVMPointerType annoPtrTy =
      mlir::LLVM::LLVMPointerType::get(globalVarBuilder.getContext());
  // First field is either a global name or a alloca address and is handled
  // by the caller, this function deals with content from `AnnotationAttr`
  // only.

  // The second field is ptr to the annotation name
  mlir::StringAttr annotationName = annotation.getName();
  auto annotationNameFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
      localLoc, annoPtrTy,
      getAnnotationStringGlobal(annotationName, module, stringGlobalsMap,
                                globalVarBuilder, localLoc)
          .getSymName());
  outVals.push_back(annotationNameFld->getResult(0));

  // The third field is ptr to the translation unit name,
  // and the fourth field is the line number
  if (mlir::isa<mlir::FusedLoc>(annotLoc)) {
    auto FusedLoc = mlir::cast<mlir::FusedLoc>(annotLoc);
    annotLoc = FusedLoc.getLocations()[0];
  }
  auto annotFileLoc = mlir::cast<mlir::FileLineColLoc>(annotLoc);
  assert(annotFileLoc && "annotation value has to be FileLineColLoc");
  // To be consistent with clang code gen, we add trailing null char
  auto fileName = mlir::StringAttr::get(
      module.getContext(), std::string(annotFileLoc.getFilename().getValue()));
  auto fileNameFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
      localLoc, annoPtrTy,
      getAnnotationStringGlobal(fileName, module, stringGlobalsMap,
                                globalVarBuilder, localLoc)
          .getSymName());
  outVals.push_back(fileNameFld->getResult(0));

  unsigned int lineNo = annotFileLoc.getLine();
  auto lineNoFld = varInitBuilder.create<mlir::LLVM::ConstantOp>(
      localLoc, globalVarBuilder.getI32Type(), lineNo);
  outVals.push_back(lineNoFld->getResult(0));

  // The fifth field is ptr to the annotation args var, it could be null
  if (annotation.isNoArgs()) {
    auto nullPtrFld =
        varInitBuilder.create<mlir::LLVM::ZeroOp>(localLoc, annoPtrTy);
    outVals.push_back(nullPtrFld->getResult(0));
  } else {
    mlir::ArrayAttr argsAttr = annotation.getArgs();
    mlir::LLVM::GlobalOp annotArgsVar =
        getOrCreateAnnotationArgsVar(localLoc, module, globalVarBuilder,
                                     argStringGlobalsMap, argsVarMap, argsAttr);
    auto argsVarView = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
        localLoc, annoPtrTy, annotArgsVar.getSymName());
    outVals.push_back(argsVarView->getResult(0));
  }
}

// Get addrspace by converting a pointer type.
// TODO: The approach here is a little hacky. We should access the target info
// directly to convert the address space of global op, similar to what we do
// for type converter.
unsigned getGlobalOpTargetAddrSpace(mlir::ConversionPatternRewriter &rewriter,
                                    const mlir::TypeConverter *converter,
                                    cir::GlobalOp op) {
  auto tempPtrTy = cir::PointerType::get(rewriter.getContext(), op.getSymType(),
                                         op.getAddrSpace());
  return cast<mlir::LLVM::LLVMPointerType>(converter->convertType(tempPtrTy))
      .getAddressSpace();
}

/// Given a type convertor and a data layout, convert the given type to a type
/// that is suitable for memory operations. For example, this can be used to
/// lower cir.bool accesses to i8.
static mlir::Type convertTypeForMemory(const mlir::TypeConverter &converter,
                                       mlir::DataLayout const &dataLayout,
                                       mlir::Type type) {
  // TODO(cir): Handle other types similarly to clang's codegen
  // convertTypeForMemory
  if (isa<cir::BoolType>(type)) {
    return mlir::IntegerType::get(type.getContext(),
                                  dataLayout.getTypeSizeInBits(type));
  }

  return converter.convertType(type);
}

/// Emits the value from memory as expected by its users. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitFromMemory(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::DataLayout const &dataLayout,
                                  cir::LoadOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitFromMemory
  if (auto boolTy = mlir::dyn_cast<cir::BoolType>(op.getType())) {
    // Create a cast value from specified size in datalayout to i1
    assert(value.getType().isInteger(dataLayout.getTypeSizeInBits(boolTy)));
    return createIntCast(rewriter, value, rewriter.getI1Type());
  }

  return value;
}

/// Emits a value to memory with the expected scalar type. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitToMemory(mlir::ConversionPatternRewriter &rewriter,
                                mlir::DataLayout const &dataLayout,
                                mlir::Type origType, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitToMemory
  if (auto boolTy = mlir::dyn_cast<cir::BoolType>(origType)) {
    // Create zext of value from i1 to i8
    auto memType =
        rewriter.getIntegerType(dataLayout.getTypeSizeInBits(boolTy));
    return createIntCast(rewriter, value, memType);
  }

  return value;
}

std::optional<llvm::StringRef>
getLLVMSyncScope(std::optional<cir::MemScopeKind> syncScope) {
  if (syncScope.has_value())
    return syncScope.value() == cir::MemScopeKind::SingleThread ? "singlethread"
                                                                : "";
  return std::nullopt;
}
} // namespace

//===----------------------------------------------------------------------===//
// Visitors for Lowering CIR Const Attributes
//===----------------------------------------------------------------------===//

/// Emits a value to memory with the expected scalar type. Should be called when
/// the memory represetnation of a CIR attribute's type is not equal to its
/// scalar representation.
static mlir::Value
emitCirAttrToMemory(mlir::Operation *parentOp, mlir::Attribute attr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter,
                    mlir::DataLayout const &dataLayout) {

  mlir::Value loweredValue =
      lowerCirAttrAsValue(parentOp, attr, rewriter, converter, dataLayout);
  if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(attr)) {
    return emitToMemory(rewriter, dataLayout, boolAttr.getType(), loweredValue);
  }

  return loweredValue;
}

/// Switches on the type of attribute and calls the appropriate conversion.
class CirAttrToValue {
public:
  CirAttrToValue(mlir::Operation *parentOp,
                 mlir::ConversionPatternRewriter &rewriter,
                 const mlir::TypeConverter *converter,
                 mlir::DataLayout const &dataLayout)
      : parentOp(parentOp), rewriter(rewriter), converter(converter),
        dataLayout(dataLayout) {}

  mlir::Value visit(mlir::Attribute attr) {
    return llvm::TypeSwitch<mlir::Attribute, mlir::Value>(attr)
        .Case<cir::IntAttr, cir::FPAttr, cir::ConstPtrAttr,
              cir::ConstRecordAttr, cir::ConstArrayAttr, cir::ConstVectorAttr,
              cir::BoolAttr, cir::ZeroAttr, cir::UndefAttr, cir::PoisonAttr,
              cir::GlobalViewAttr, cir::VTableAttr, cir::TypeInfoAttr,
              cir::ComplexAttr>([&](auto attrT) { return visitCirAttr(attrT); })
        .Default([&](auto attrT) { return mlir::Value(); });
  }

  mlir::Value visitCirAttr(cir::IntAttr attr);
  mlir::Value visitCirAttr(cir::FPAttr attr);
  mlir::Value visitCirAttr(cir::ConstPtrAttr attr);
  mlir::Value visitCirAttr(cir::ConstRecordAttr attr);
  mlir::Value visitCirAttr(cir::ConstArrayAttr attr);
  mlir::Value visitCirAttr(cir::ConstVectorAttr attr);
  mlir::Value visitCirAttr(cir::ComplexAttr attr);
  mlir::Value visitCirAttr(cir::BoolAttr attr);
  mlir::Value visitCirAttr(cir::ZeroAttr attr);
  mlir::Value visitCirAttr(cir::UndefAttr attr);
  mlir::Value visitCirAttr(cir::PoisonAttr attr);
  mlir::Value visitCirAttr(cir::GlobalViewAttr attr);
  mlir::Value visitCirAttr(cir::VTableAttr attr);
  mlir::Value visitCirAttr(cir::TypeInfoAttr attr);

private:
  mlir::Operation *parentOp;
  mlir::ConversionPatternRewriter &rewriter;
  const mlir::TypeConverter *converter;
  mlir::DataLayout const &dataLayout;
};

/// IntAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::IntAttr intAttr) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(intAttr.getType()), intAttr.getValue());
}

/// BoolAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::BoolAttr boolAttr) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(boolAttr.getType()), boolAttr.getValue());
}

/// ConstPtrAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::ConstPtrAttr ptrAttr) {
  auto loc = parentOp->getLoc();
  if (ptrAttr.isNullValue()) {
    return rewriter.create<mlir::LLVM::ZeroOp>(
        loc, converter->convertType(ptrAttr.getType()));
  }
  mlir::DataLayout layout(parentOp->getParentOfType<mlir::ModuleOp>());
  mlir::Value ptrVal = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getIntegerType(layout.getTypeSizeInBits(ptrAttr.getType())),
      ptrAttr.getValue().getInt());
  return rewriter.create<mlir::LLVM::IntToPtrOp>(
      loc, converter->convertType(ptrAttr.getType()), ptrVal);
}

/// FPAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::FPAttr fltAttr) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(fltAttr.getType()), fltAttr.getValue());
}

/// ZeroAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::ZeroAttr zeroAttr) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ZeroOp>(
      loc, converter->convertType(zeroAttr.getType()));
}

/// UndefAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::UndefAttr undefAttr) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::UndefOp>(
      loc, converter->convertType(undefAttr.getType()));
}

/// PoisonAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::PoisonAttr poisonAttr) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::PoisonOp>(
      loc, converter->convertType(poisonAttr.getType()));
}

/// ConstRecord visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::ConstRecordAttr constRecord) {
  auto llvmTy = converter->convertType(constRecord.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  // Iteratively lower each constant element of the record.
  for (auto [idx, elt] : llvm::enumerate(constRecord.getMembers())) {
    mlir::Value init =
        emitCirAttrToMemory(parentOp, elt, rewriter, converter, dataLayout);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// VTableAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::VTableAttr vtableArr) {
  auto llvmTy = converter->convertType(vtableArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(vtableArr.getVtableData())) {
    mlir::Value init = visit(elt);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// TypeInfoAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::TypeInfoAttr typeinfoArr) {
  auto llvmTy = converter->convertType(typeinfoArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(typeinfoArr.getData())) {
    mlir::Value init = visit(elt);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// ConstArrayAttr visitor
mlir::Value CirAttrToValue::visitCirAttr(cir::ConstArrayAttr constArr) {
  auto llvmTy = converter->convertType(constArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result;

  if (constArr.getTrailingZerosNum() > 0) {
    auto arrayTy = constArr.getType();
    result = rewriter.create<mlir::LLVM::ZeroOp>(
        loc, converter->convertType(arrayTy));
  } else {
    result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);
  }

  // Iteratively lower each constant element of the array.
  if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts())) {
    for (auto [idx, elt] : llvm::enumerate(arrayAttr)) {
      mlir::Value init =
          emitCirAttrToMemory(parentOp, elt, rewriter, converter, dataLayout);
      result =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
    }
  }
  // TODO(cir): this diverges from traditional lowering. Normally the string
  // would be a global constant that is memcopied.
  else if (auto strAttr =
               mlir::dyn_cast<mlir::StringAttr>(constArr.getElts())) {
    auto arrayTy = mlir::dyn_cast<cir::ArrayType>(strAttr.getType());
    assert(arrayTy && "String attribute must have an array type");
    auto eltTy = arrayTy.getElementType();
    for (auto [idx, elt] : llvm::enumerate(strAttr)) {
      auto init = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, converter->convertType(eltTy), elt);
      result =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
    }
  } else {
    llvm_unreachable("unexpected ConstArrayAttr elements");
  }

  return result;
}

// ConstVectorAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::ConstVectorAttr constVec) {
  auto llvmTy = converter->convertType(constVec.getType());
  auto loc = parentOp->getLoc();
  SmallVector<mlir::Attribute> mlirValues;
  for (auto elementAttr : constVec.getElts()) {
    mlir::Attribute mlirAttr;
    if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(elementAttr)) {
      mlirAttr = rewriter.getIntegerAttr(
          converter->convertType(intAttr.getType()), intAttr.getValue());
    } else if (auto floatAttr = mlir::dyn_cast<cir::FPAttr>(elementAttr)) {
      mlirAttr = rewriter.getFloatAttr(
          converter->convertType(floatAttr.getType()), floatAttr.getValue());
    } else {
      llvm_unreachable(
          "vector constant with an element that is neither an int nor a float");
    }
    mlirValues.push_back(mlirAttr);
  }
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, llvmTy,
      mlir::DenseElementsAttr::get(mlir::cast<mlir::ShapedType>(llvmTy),
                                   mlirValues));
}

mlir::Value CirAttrToValue::visitCirAttr(cir::ComplexAttr complexAttr) {
  auto complexType = mlir::cast<cir::ComplexType>(complexAttr.getType());
  mlir::Type complexElemTy = complexType.getElementType();
  mlir::Type complexElemLLVMTy = converter->convertType(complexElemTy);

  mlir::Attribute components[2];
  if (const auto intType = mlir::dyn_cast<cir::IntType>(complexElemTy)) {
    components[0] = rewriter.getIntegerAttr(
        complexElemLLVMTy,
        mlir::cast<cir::IntAttr>(complexAttr.getReal()).getValue());
    components[1] = rewriter.getIntegerAttr(
        complexElemLLVMTy,
        mlir::cast<cir::IntAttr>(complexAttr.getImag()).getValue());
  } else {
    components[0] = rewriter.getFloatAttr(
        complexElemLLVMTy,
        mlir::cast<cir::FPAttr>(complexAttr.getReal()).getValue());
    components[1] = rewriter.getFloatAttr(
        complexElemLLVMTy,
        mlir::cast<cir::FPAttr>(complexAttr.getImag()).getValue());
  }

  mlir::Location loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(complexAttr.getType()),
      rewriter.getArrayAttr(components));
}
// GlobalViewAttr visitor.
mlir::Value CirAttrToValue::visitCirAttr(cir::GlobalViewAttr globalAttr) {
  auto module = parentOp->getParentOfType<mlir::ModuleOp>();
  mlir::Type sourceType;
  unsigned sourceAddrSpace = 0;
  llvm::StringRef symName;
  auto *sourceSymbol =
      mlir::SymbolTable::lookupSymbolIn(module, globalAttr.getSymbol());
  if (auto llvmSymbol = dyn_cast<mlir::LLVM::GlobalOp>(sourceSymbol)) {
    sourceType = llvmSymbol.getType();
    symName = llvmSymbol.getSymName();
    sourceAddrSpace = llvmSymbol.getAddrSpace();
  } else if (auto cirSymbol = dyn_cast<cir::GlobalOp>(sourceSymbol)) {
    sourceType =
        convertTypeForMemory(*converter, dataLayout, cirSymbol.getSymType());
    symName = cirSymbol.getSymName();
    sourceAddrSpace =
        getGlobalOpTargetAddrSpace(rewriter, converter, cirSymbol);
  } else if (auto llvmFun = dyn_cast<mlir::LLVM::LLVMFuncOp>(sourceSymbol)) {
    sourceType = llvmFun.getFunctionType();
    symName = llvmFun.getSymName();
    sourceAddrSpace = 0;
  } else if (auto fun = dyn_cast<cir::FuncOp>(sourceSymbol)) {
    sourceType = converter->convertType(fun.getFunctionType());
    symName = fun.getSymName();
    sourceAddrSpace = 0;
  } else if (auto alias = dyn_cast<mlir::LLVM::AliasOp>(sourceSymbol)) {
    // FIXME: It seems unusual that we need to handle an LLVM dialect op here,
    // but that's the way it happens. Should we be doing something somewhere
    // else to lower the GlobalViewAttr sooner?
    sourceType = alias.getType();
    symName = alias.getSymName();
    sourceAddrSpace = 0;
  } else {
    llvm_unreachable("Unexpected GlobalOp type");
  }

  auto loc = parentOp->getLoc();
  mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
      loc,
      mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), sourceAddrSpace),
      symName);

  if (globalAttr.getIndices()) {
    llvm::SmallVector<mlir::LLVM::GEPArg> indices;

    if (isa<mlir::LLVM::LLVMArrayType, mlir::LLVM::LLVMStructType>(sourceType))
      indices.push_back(0);

    for (auto idx : globalAttr.getIndices()) {
      auto intAttr = dyn_cast<mlir::IntegerAttr>(idx);
      assert(intAttr && "index must be integers");
      indices.push_back(intAttr.getValue().getSExtValue());
    }
    auto resTy = addrOp.getType();
    auto eltTy = converter->convertType(sourceType);
    addrOp = rewriter.create<mlir::LLVM::GEPOp>(loc, resTy, eltTy, addrOp,
                                                indices, true);
  }

  if (auto intTy = mlir::dyn_cast<cir::IntType>(globalAttr.getType())) {
    auto llvmDstTy = converter->convertType(globalAttr.getType());
    return rewriter.create<mlir::LLVM::PtrToIntOp>(parentOp->getLoc(),
                                                   llvmDstTy, addrOp);
  }

  if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(globalAttr.getType())) {
    auto llvmEltTy =
        convertTypeForMemory(*converter, dataLayout, ptrTy.getPointee());

    if (llvmEltTy == sourceType)
      return addrOp;

    auto llvmDstTy = converter->convertType(globalAttr.getType());
    return rewriter.create<mlir::LLVM::BitcastOp>(parentOp->getLoc(), llvmDstTy,
                                                  addrOp);
  }

  llvm_unreachable("Expecting pointer or integer type for GlobalViewAttr");
}

/// Switches on the type of attribute and calls the appropriate conversion.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp,
                                const mlir::Attribute attr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter,
                                mlir::DataLayout const &dataLayout) {
  CirAttrToValue valueConverter(parentOp, rewriter, converter, dataLayout);
  auto value = valueConverter.visit(attr);
  if (!value)
    llvm_unreachable("unhandled attribute type");
  return value;
}

//===----------------------------------------------------------------------===//

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage) {
  using CIR = cir::GlobalLinkageKind;
  using LLVM = mlir::LLVM::Linkage;

  switch (linkage) {
  case CIR::AvailableExternallyLinkage:
    return LLVM::AvailableExternally;
  case CIR::CommonLinkage:
    return LLVM::Common;
  case CIR::ExternalLinkage:
    return LLVM::External;
  case CIR::ExternalWeakLinkage:
    return LLVM::ExternWeak;
  case CIR::InternalLinkage:
    return LLVM::Internal;
  case CIR::LinkOnceAnyLinkage:
    return LLVM::Linkonce;
  case CIR::LinkOnceODRLinkage:
    return LLVM::LinkonceODR;
  case CIR::PrivateLinkage:
    return LLVM::Private;
  case CIR::WeakAnyLinkage:
    return LLVM::Weak;
  case CIR::WeakODRLinkage:
    return LLVM::WeakODR;
  };
}

mlir::LLVM::CConv convertCallingConv(cir::CallingConv callingConv) {
  using CIR = cir::CallingConv;
  using LLVM = mlir::LLVM::CConv;

  switch (callingConv) {
  case CIR::C:
    return LLVM::C;
  case CIR::SpirKernel:
    return LLVM::SPIR_KERNEL;
  case CIR::SpirFunction:
    return LLVM::SPIR_FUNC;
  case CIR::OpenCLKernel:
    llvm_unreachable("NYI");
  case CIR::PTXKernel:
    return LLVM::PTX_Kernel;
  }
  llvm_unreachable("Unknown calling convention");
}

void convertSideEffectForCall(mlir::Operation *callOp,
                              cir::SideEffect sideEffect,
                              mlir::LLVM::MemoryEffectsAttr &memoryEffect,
                              bool &noUnwind, bool &willReturn) {
  using mlir::LLVM::ModRefInfo;

  switch (sideEffect) {
  case cir::SideEffect::All:
    memoryEffect = {};
    noUnwind = false;
    willReturn = false;
    break;

  case cir::SideEffect::Pure:
    memoryEffect = mlir::LLVM::MemoryEffectsAttr::get(
        callOp->getContext(), /*other=*/ModRefInfo::Ref,
        /*argMem=*/ModRefInfo::Ref,
        /*inaccessibleMem=*/ModRefInfo::Ref);
    noUnwind = true;
    willReturn = true;
    break;

  case cir::SideEffect::Const:
    memoryEffect = mlir::LLVM::MemoryEffectsAttr::get(
        callOp->getContext(), /*other=*/ModRefInfo::NoModRef,
        /*argMem=*/ModRefInfo::NoModRef,
        /*inaccessibleMem=*/ModRefInfo::NoModRef);
    noUnwind = true;
    willReturn = true;
    break;
  }
}

mlir::LogicalResult CIRToLLVMCopyOpLowering::matchAndRewrite(
    cir::CopyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const mlir::Value length = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI32Type(), op.getLength());
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyOp>(
      op, adaptor.getDst(), adaptor.getSrc(), length, op.getIsVolatile());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemCpyOpLowering::matchAndRewrite(
    cir::MemCpyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
      /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemChrOpLowering::matchAndRewrite(
    cir::MemChrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  llvm::SmallVector<mlir::Type> arguments;
  const mlir::TypeConverter *converter = getTypeConverter();
  mlir::Type srcTy = converter->convertType(op.getSrc().getType());
  mlir::Type patternTy = converter->convertType(op.getPattern().getType());
  mlir::Type lenTy = converter->convertType(op.getLen().getType());
  auto fnTy =
      mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {srcTy, patternTy, lenTy},
                                        /*isVarArg=*/false);
  llvm::StringRef fnName = "memchr";
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{llvmPtrTy}, fnName,
      mlir::ValueRange{adaptor.getSrc(), adaptor.getPattern(),
                       adaptor.getLen()});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemMoveOpLowering::matchAndRewrite(
    cir::MemMoveOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemmoveOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
      /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemCpyInlineOpLowering::matchAndRewrite(
    cir::MemCpyInlineOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyInlineOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLenAttr(),
      /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemSetOpLowering::matchAndRewrite(
    cir::MemSetOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto converted = rewriter.create<mlir::LLVM::TruncOp>(
      op.getLoc(), mlir::IntegerType::get(op.getContext(), 8),
      adaptor.getVal());
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemsetOp>(op, adaptor.getDst(),
                                                    converted, adaptor.getLen(),
                                                    /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemSetInlineOpLowering::matchAndRewrite(
    cir::MemSetInlineOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto converted = rewriter.create<mlir::LLVM::TruncOp>(
      op.getLoc(), mlir::IntegerType::get(op.getContext(), 8),
      adaptor.getVal());
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemsetInlineOp>(
      op, adaptor.getDst(), converted, adaptor.getLenAttr(),
      /*isVolatile=*/false);
  return mlir::success();
}

static mlir::Value getLLVMIntCast(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Value llvmSrc, mlir::Type llvmDstIntTy,
                                  bool isUnsigned, uint64_t cirSrcWidth,
                                  uint64_t cirDstIntWidth) {
  if (cirSrcWidth == cirDstIntWidth)
    return llvmSrc;

  auto loc = llvmSrc.getLoc();
  if (cirSrcWidth < cirDstIntWidth) {
    if (isUnsigned)
      return rewriter.create<mlir::LLVM::ZExtOp>(loc, llvmDstIntTy, llvmSrc);
    return rewriter.create<mlir::LLVM::SExtOp>(loc, llvmDstIntTy, llvmSrc);
  }

  // Otherwise truncate
  return rewriter.create<mlir::LLVM::TruncOp>(loc, llvmDstIntTy, llvmSrc);
}

static mlir::Value promoteIndex(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Value index, uint64_t layoutWidth,
                                bool isUnsigned) {
  auto indexOp = index.getDefiningOp();
  if (!indexOp)
    return index;

  auto indexType = mlir::cast<mlir::IntegerType>(index.getType());
  auto width = indexType.getWidth();
  if (layoutWidth == width)
    return index;

  // If the index definition is a unary minus (index = sub 0, x), then we need
  // to
  bool rewriteSub = false;
  auto sub = mlir::dyn_cast<mlir::LLVM::SubOp>(indexOp);
  if (sub) {
    if (auto lhsConst =
            sub.getOperand(0).getDefiningOp<mlir::LLVM::ConstantOp>()) {
      auto lhsConstInt = mlir::dyn_cast<mlir::IntegerAttr>(lhsConst.getValue());
      if (lhsConstInt && lhsConstInt.getValue() == 0) {
        rewriteSub = true;
        index = sub.getOperand(1);
      }
    }
  }

  // Handle the cast
  auto llvmDstType = mlir::IntegerType::get(rewriter.getContext(), layoutWidth);
  index = getLLVMIntCast(rewriter, index, llvmDstType, isUnsigned, width,
                         layoutWidth);

  if (rewriteSub) {
    index = rewriter.create<mlir::LLVM::SubOp>(
        index.getLoc(),
        rewriter.create<mlir::LLVM::ConstantOp>(index.getLoc(), index.getType(),
                                                0),
        index);
    // TODO: check if the sub is trivially dead now.
    rewriter.eraseOp(sub);
  }

  return index;
}

mlir::LogicalResult CIRToLLVMPtrStrideOpLowering::matchAndRewrite(
    cir::PtrStrideOp ptrStrideOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto *tc = getTypeConverter();
  const auto resultTy = tc->convertType(ptrStrideOp.getType());
  auto elementTy =
      convertTypeForMemory(*tc, dataLayout, ptrStrideOp.getElementTy());
  auto *ctx = elementTy.getContext();

  // void and function types doesn't really have a layout to use in GEPs,
  // make it i8 instead.
  if (mlir::isa<mlir::LLVM::LLVMVoidType>(elementTy) ||
      mlir::isa<mlir::LLVM::LLVMFunctionType>(elementTy))
    elementTy = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signless);

  // Zero-extend, sign-extend or trunc the pointer value.
  auto index = adaptor.getStride();
  mlir::DataLayout LLVMLayout(ptrStrideOp->getParentOfType<mlir::ModuleOp>());
  if (auto layoutWidth =
          LLVMLayout.getTypeIndexBitwidth(adaptor.getBase().getType())) {
    bool isUnsigned = false;
    if (auto strideTy =
            mlir::dyn_cast<cir::IntType>(ptrStrideOp.getOperand(1).getType()))
      isUnsigned = strideTy.isUnsigned();
    index = promoteIndex(rewriter, index, *layoutWidth, isUnsigned);
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      ptrStrideOp, resultTy, elementTy, adaptor.getBase(), index);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGetElementOpLowering::matchAndRewrite(
    cir::GetElementOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  if (auto arrayTy =
          mlir::dyn_cast<cir::ArrayType>(op.getBaseType().getPointee())) {
    auto *tc = getTypeConverter();
    const auto llResultTy = tc->convertType(op.getType());
    auto elementTy = convertTypeForMemory(*tc, dataLayout, op.getElementType());
    auto *ctx = elementTy.getContext();

    // void and function types doesn't really have a layout to use in GEPs,
    // make it i8 instead.
    if (mlir::isa<mlir::LLVM::LLVMVoidType>(elementTy) ||
        mlir::isa<mlir::LLVM::LLVMFunctionType>(elementTy))
      elementTy = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signless);

    // Zero-extend, sign-extend or trunc the index value.
    auto index = adaptor.getIndex();
    mlir::DataLayout LLVMLayout(op->getParentOfType<mlir::ModuleOp>());
    if (auto layoutWidth =
            LLVMLayout.getTypeIndexBitwidth(adaptor.getBase().getType())) {
      bool isUnsigned = false;
      if (auto strideTy = dyn_cast<cir::IntType>(op.getOperand(1).getType()))
        isUnsigned = strideTy.isUnsigned();
      index = promoteIndex(rewriter, index, *layoutWidth, isUnsigned);
    }

    // Since the base address is a pointer to an aggregate, the first
    // offset is always zero. The second offset tell us which member it
    // will access.
    const auto llArrayTy = getTypeConverter()->convertType(arrayTy);
    llvm::SmallVector<mlir::LLVM::GEPArg, 2> offset{0, index};
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, llResultTy, llArrayTy,
                                                   adaptor.getBase(), offset);

    return mlir::success();
  }

  llvm_unreachable("NYI, GetElementOp lowering to LLVM for non-Array");
}

mlir::LogicalResult CIRToLLVMBaseClassAddrOpLowering::matchAndRewrite(
    cir::BaseClassAddrOp baseClassOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto resultType =
      getTypeConverter()->convertType(baseClassOp.getType());
  mlir::Value derivedAddr = adaptor.getDerivedAddr();
  llvm::SmallVector<mlir::LLVM::GEPArg, 1> offset = {
      adaptor.getOffset().getZExtValue()};
  mlir::Type byteType = mlir::IntegerType::get(resultType.getContext(), 8,
                                               mlir::IntegerType::Signless);
  if (adaptor.getOffset().getZExtValue() == 0) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(
        baseClassOp, resultType, adaptor.getDerivedAddr());
    return mlir::success();
  }

  if (baseClassOp.getAssumeNotNull()) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        baseClassOp, resultType, byteType, derivedAddr, offset);
  } else {
    auto loc = baseClassOp.getLoc();
    mlir::Value isNull = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, derivedAddr,
        rewriter.create<mlir::LLVM::ZeroOp>(loc, derivedAddr.getType()));
    mlir::Value adjusted = rewriter.create<mlir::LLVM::GEPOp>(
        loc, resultType, byteType, derivedAddr, offset);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(baseClassOp, isNull,
                                                      derivedAddr, adjusted);
  }
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMDerivedClassAddrOpLowering::matchAndRewrite(
    cir::DerivedClassAddrOp derivedClassOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto resultType =
      getTypeConverter()->convertType(derivedClassOp.getType());
  mlir::Value baseAddr = adaptor.getBaseAddr();
  int64_t offsetVal = adaptor.getOffset().getZExtValue() * -1;
  llvm::SmallVector<mlir::LLVM::GEPArg, 1> offset = {offsetVal};
  mlir::Type byteType = mlir::IntegerType::get(resultType.getContext(), 8,
                                               mlir::IntegerType::Signless);
  if (derivedClassOp.getAssumeNotNull()) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(derivedClassOp, resultType,
                                                   byteType, baseAddr, offset);
  } else {
    auto loc = derivedClassOp.getLoc();
    mlir::Value isNull = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, baseAddr,
        rewriter.create<mlir::LLVM::ZeroOp>(loc, baseAddr.getType()));
    mlir::Value adjusted = rewriter.create<mlir::LLVM::GEPOp>(
        loc, resultType, byteType, baseAddr, offset);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(derivedClassOp, isNull,
                                                      baseAddr, adjusted);
  }
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBaseDataMemberOpLowering::matchAndRewrite(
    cir::BaseDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerMod->getCXXABI().lowerBaseDataMember(op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMDerivedDataMemberOpLowering::matchAndRewrite(
    cir::DerivedDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerMod->getCXXABI().lowerDerivedDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBaseMethodOpLowering::matchAndRewrite(
    cir::BaseMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerMod->getCXXABI().lowerBaseMethod(op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMDerivedMethodOpLowering::matchAndRewrite(
    cir::DerivedMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerMod->getCXXABI().lowerDerivedMethod(op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

static mlir::Value
getValueForVTableSymbol(mlir::Operation *op,
                        mlir::ConversionPatternRewriter &rewriter,
                        const mlir::TypeConverter *converter,
                        mlir::FlatSymbolRefAttr nameAttr, mlir::Type &eltType) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto *symbol = mlir::SymbolTable::lookupSymbolIn(module, nameAttr);
  if (auto llvmSymbol = dyn_cast<mlir::LLVM::GlobalOp>(symbol)) {
    eltType = llvmSymbol.getType();
  } else if (auto cirSymbol = dyn_cast<cir::GlobalOp>(symbol)) {
    eltType = converter->convertType(cirSymbol.getSymType());
  }
  return rewriter.create<mlir::LLVM::AddressOfOp>(
      op->getLoc(), mlir::LLVM::LLVMPointerType::get(op->getContext()),
      nameAttr.getValue());
}

mlir::LogicalResult CIRToLLVMVTTAddrPointOpLowering::matchAndRewrite(
    cir::VTTAddrPointOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const mlir::Type resultType = getTypeConverter()->convertType(op.getType());
  llvm::SmallVector<mlir::LLVM::GEPArg> offsets;
  mlir::Type eltType;
  mlir::Value llvmAddr = adaptor.getSymAddr();

  if (op.getSymAddr()) {
    if (op.getOffset() == 0) {
      rewriter.replaceOp(op, {llvmAddr});
      return mlir::success();
    }

    offsets.push_back(adaptor.getOffset());
    eltType = mlir::IntegerType::get(resultType.getContext(), 8,
                                     mlir::IntegerType::Signless);
  } else {
    llvmAddr = getValueForVTableSymbol(op, rewriter, getTypeConverter(),
                                       op.getNameAttr(), eltType);
    assert(eltType && "Shouldn't ever be missing an eltType here");
    offsets.push_back(0);
    offsets.push_back(adaptor.getOffset());
  }
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, resultType, eltType,
                                                 llvmAddr, offsets, true);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBrCondOpLowering::matchAndRewrite(
    cir::BrCondOp brOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value i1Condition;

  auto hasOneUse = false;

  if (auto defOp = brOp.getCond().getDefiningOp())
    hasOneUse = defOp->getResult(0).hasOneUse();

  if (auto zext = adaptor.getCond().getDefiningOp<mlir::LLVM::ZExtOp>()) {
    if (zext->use_empty() &&
        zext->getOperand(0).getType() == rewriter.getI1Type()) {
      i1Condition = zext->getOperand(0);
      if (hasOneUse)
        rewriter.eraseOp(zext);
    }
  }

  if (!i1Condition)
    i1Condition = adaptor.getCond();

  rewriter.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
      brOp, i1Condition, brOp.getDestTrue(), adaptor.getDestOperandsTrue(),
      brOp.getDestFalse(), adaptor.getDestOperandsFalse());

  return mlir::success();
}

mlir::Type CIRToLLVMCastOpLowering::convertTy(mlir::Type ty) const {
  return getTypeConverter()->convertType(ty);
}

mlir::LogicalResult CIRToLLVMCastOpLowering::matchAndRewrite(
    cir::CastOp castOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // For arithmetic conversions, LLVM IR uses the same instruction to convert
  // both individual scalars and entire vectors. This lowering pass handles
  // both situations.

  auto src = adaptor.getSrc();

  switch (castOp.getKind()) {
  case cir::CastKind::array_to_ptrdecay: {
    const auto ptrTy = mlir::cast<cir::PointerType>(castOp.getType());
    auto sourceValue = adaptor.getSrc();
    auto targetType = convertTy(ptrTy);
    auto elementTy = convertTypeForMemory(*getTypeConverter(), dataLayout,
                                          ptrTy.getPointee());
    auto offset = llvm::SmallVector<mlir::LLVM::GEPArg>{0};
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        castOp, targetType, elementTy, sourceValue, offset);
    break;
  }
  case cir::CastKind::int_to_bool: {
    auto zero = rewriter.create<cir::ConstantOp>(
        src.getLoc(), cir::IntAttr::get(castOp.getSrc().getType(), 0));
    rewriter.replaceOpWithNewOp<cir::CmpOp>(
        castOp, cir::BoolType::get(getContext()), cir::CmpOpKind::ne,
        castOp.getSrc(), zero);
    break;
  }
  case cir::CastKind::integral: {
    auto srcType = castOp.getSrc().getType();
    auto dstType = castOp.getType();
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstType = getTypeConverter()->convertType(dstType);
    cir::IntType srcIntType =
        mlir::cast<cir::IntType>(elementTypeIfVector(srcType));
    cir::IntType dstIntType =
        mlir::cast<cir::IntType>(elementTypeIfVector(dstType));
    rewriter.replaceOp(castOp, getLLVMIntCast(rewriter, llvmSrcVal, llvmDstType,
                                              srcIntType.isUnsigned(),
                                              srcIntType.getWidth(),
                                              dstIntType.getWidth()));
    break;
  }
  case cir::CastKind::floating: {
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstTy = getTypeConverter()->convertType(castOp.getType());

    auto srcTy = elementTypeIfVector(castOp.getSrc().getType());
    auto dstTy = elementTypeIfVector(castOp.getType());

    if (!mlir::isa<cir::FPTypeInterface>(dstTy) ||
        !mlir::isa<cir::FPTypeInterface>(srcTy))
      return castOp.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

    auto getFloatWidth = [](mlir::Type ty) -> unsigned {
      return mlir::cast<cir::FPTypeInterface>(ty).getWidth();
    };

    if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPExtOp>(castOp, llvmDstTy,
                                                       llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::int_to_ptr: {
    auto dstTy = mlir::cast<cir::PointerType>(castOp.getType());
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::ptr_to_int: {
    auto dstTy = mlir::cast<cir::IntType>(castOp.getType());
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::float_to_bool: {
    auto llvmSrcVal = adaptor.getSrc();
    auto kind = mlir::LLVM::FCmpPredicate::une;

    // Check if float is not equal to zero.
    auto zeroFloat = rewriter.create<mlir::LLVM::ConstantOp>(
        castOp.getLoc(), llvmSrcVal.getType(),
        mlir::FloatAttr::get(llvmSrcVal.getType(), 0.0));

    // Extend comparison result to either bool (C++) or int (C).
    rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(castOp, kind, llvmSrcVal,
                                                    zeroFloat);

    return mlir::success();
  }
  case cir::CastKind::bool_to_int: {
    auto dstTy = mlir::cast<cir::IntType>(castOp.getType());
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmSrcTy = mlir::cast<mlir::IntegerType>(llvmSrcVal.getType());
    auto llvmDstTy =
        mlir::cast<mlir::IntegerType>(getTypeConverter()->convertType(dstTy));
    if (llvmSrcTy.getWidth() == llvmDstTy.getWidth())
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::bool_to_float: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::int_to_float: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (mlir::cast<cir::IntType>(elementTypeIfVector(castOp.getSrc().getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::float_to_int: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (mlir::cast<cir::IntType>(elementTypeIfVector(castOp.getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPToUIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::bitcast: {
    auto dstTy = castOp.getType();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);

    if (mlir::isa<cir::DataMemberType, cir::MethodType>(
            castOp.getSrc().getType())) {
      mlir::Value loweredResult;
      if (mlir::isa<cir::DataMemberType>(castOp.getSrc().getType()))
        loweredResult = lowerMod->getCXXABI().lowerDataMemberBitcast(
            castOp, llvmDstTy, src, rewriter);
      else
        loweredResult = lowerMod->getCXXABI().lowerMethodBitcast(
            castOp, llvmDstTy, src, rewriter);
      rewriter.replaceOp(castOp, loweredResult);
      return mlir::success();
    }

    auto llvmSrcVal = adaptor.getSrc();
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                       llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::ptr_to_bool: {
    auto zero = rewriter.getI64IntegerAttr(0);
    auto null = rewriter.create<cir::ConstantOp>(
        src.getLoc(), cir::ConstPtrAttr::get(castOp.getSrc().getType(), zero));
    rewriter.replaceOpWithNewOp<cir::CmpOp>(
        castOp, cir::BoolType::get(getContext()), cir::CmpOpKind::ne,
        castOp.getSrc(), null);
    break;
  }
  case cir::CastKind::address_space: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getSrc();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddrSpaceCastOp>(castOp, llvmDstTy,
                                                             llvmSrcVal);
    break;
  }
  case cir::CastKind::member_ptr_to_bool: {
    mlir::Value loweredResult;
    if (mlir::isa<cir::MethodType>(castOp.getSrc().getType()))
      loweredResult =
          lowerMod->getCXXABI().lowerMethodToBoolCast(castOp, src, rewriter);
    else
      loweredResult = lowerMod->getCXXABI().lowerDataMemberToBoolCast(
          castOp, src, rewriter);
    rewriter.replaceOp(castOp, loweredResult);
    break;
  }
  default: {
    return castOp.emitError("Unhandled cast kind: ")
           << castOp.getKindAttrName();
  }
  }

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMReturnOpLowering::matchAndRewrite(
    cir::ReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
  return mlir::LogicalResult::success();
}

struct ConvertCIRToLLVMPass
    : public mlir::PassWrapper<ConvertCIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::DLTIDialect,
                    mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();
  }
  void runOnOperation() final;

  void buildGlobalAnnotationsVar(
      llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
      llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
      llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap);

  void processCIRAttrs(mlir::ModuleOp moduleOp);

  StringRef getDescription() const override {
    return "Convert the prepared CIR dialect module to LLVM dialect";
  }

  StringRef getArgument() const override { return "cir-flat-to-llvm"; }
};

mlir::LogicalResult
rewriteToCallOrInvoke(mlir::Operation *op, mlir::ValueRange callOperands,
                      mlir::ConversionPatternRewriter &rewriter,
                      const mlir::TypeConverter *converter,
                      mlir::FlatSymbolRefAttr calleeAttr,
                      mlir::Block *continueBlock = nullptr,
                      mlir::Block *landingPadBlock = nullptr) {
  llvm::SmallVector<mlir::Type, 8> llvmResults;
  auto cirResults = op->getResultTypes();
  auto callIf = cast<cir::CIRCallOpInterface>(op);

  if (converter->convertTypes(cirResults, llvmResults).failed())
    return mlir::failure();

  auto cconv = convertCallingConv(callIf.getCallingConv());

  mlir::LLVM::MemoryEffectsAttr memoryEffects;
  bool noUnwind = false;
  bool willReturn = false;
  convertSideEffectForCall(op, callIf.getSideEffect(), memoryEffects, noUnwind,
                           willReturn);

  mlir::LLVM::LLVMFunctionType llvmFnTy;

  // Temporary to handle the case where we need to prepend an operand if the
  // callee is an alias.
  SmallVector<mlir::Value> adjustedCallOperands;

  if (calleeAttr) { // direct call
    mlir::Operation *callee =
        mlir::SymbolTable::lookupNearestSymbolFrom(op, calleeAttr);
    if (auto fn = dyn_cast<mlir::FunctionOpInterface>(callee)) {
      llvmFnTy = cast<mlir::LLVM::LLVMFunctionType>(
          converter->convertType(fn.getFunctionType()));
    } else if (auto alias = cast<mlir::LLVM::AliasOp>(callee)) {
      // If the callee wasan alias. In that case,
      // we need to prepend the address of the alias to the operands. The
      // way aliases work in the LLVM dialect is a little counter-intuitive.
      // The AliasOp itself is a pseudo-function that returns the address of
      // the global value being aliased, but when we generate the call we
      // need to insert an operation that gets the address of the AliasOp.
      // This all gets sorted out when the LLVM dialect is lowered to LLVM IR.
      auto symAttr = cast<mlir::FlatSymbolRefAttr>(calleeAttr);
      auto addrOfAlias =
          rewriter
              .create<mlir::LLVM::AddressOfOp>(
                  op->getLoc(),
                  mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
                  symAttr)
              .getResult();
      adjustedCallOperands.push_back(addrOfAlias);

      // Now add the regular operands and assign this to the range value.
      llvm::append_range(adjustedCallOperands, callOperands);
      callOperands = adjustedCallOperands;

      // Clear the callee attribute because we're calling an alias.
      calleeAttr = {};
      llvmFnTy = cast<mlir::LLVM::LLVMFunctionType>(alias.getType());
    } else {
      // Was this an ifunc?
      return op->emitError("Unexpected callee type!");
    }
  } else { // indirect call
    assert(op->getOperands().size() &&
           "operands list must no be empty for the indirect call");
    auto typ = op->getOperands().front().getType();
    assert(isa<cir::PointerType>(typ) && "expected pointer type");
    auto ptyp = dyn_cast<cir::PointerType>(typ);
    auto ftyp = dyn_cast<cir::FuncType>(ptyp.getPointee());
    assert(ftyp && "expected a pointer to a function as the first operand");
    llvm::append_range(adjustedCallOperands, callOperands);
    llvmFnTy = cast<mlir::LLVM::LLVMFunctionType>(converter->convertType(ftyp));
  }

  if (landingPadBlock) {
    auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
        op, llvmFnTy, calleeAttr, callOperands, continueBlock,
        mlir::ValueRange{}, landingPadBlock, mlir::ValueRange{});
    newOp.setCConv(cconv);
  } else {
    auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, llvmFnTy, calleeAttr, callOperands);
    newOp.setCConv(cconv);
    if (memoryEffects)
      newOp.setMemoryEffectsAttr(memoryEffects);
    newOp.setNoUnwind(noUnwind);
    newOp.setWillReturn(willReturn);
  }
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCallOpLowering::matchAndRewrite(
    cir::CallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  return rewriteToCallOrInvoke(op.getOperation(), adaptor.getOperands(),
                               rewriter, getTypeConverter(),
                               op.getCalleeAttr());
}

mlir::LogicalResult CIRToLLVMTryCallOpLowering::matchAndRewrite(
    cir::TryCallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (op.getCallingConv() != cir::CallingConv::C) {
    return op.emitError(
        "non-C calling convention is not implemented for try_call");
  }
  return rewriteToCallOrInvoke(op.getOperation(), adaptor.getOperands(),
                               rewriter, getTypeConverter(), op.getCalleeAttr(),
                               op.getCont(), op.getLandingPad());
}

static mlir::LLVM::LLVMStructType
getLLVMLandingPadStructTy(mlir::ConversionPatternRewriter &rewriter) {
  // Create the landing pad type: struct { ptr, i32 }
  mlir::MLIRContext *ctx = rewriter.getContext();
  auto llvmPtr = mlir::LLVM::LLVMPointerType::get(ctx);
  llvm::SmallVector<mlir::Type> structFields;
  structFields.push_back(llvmPtr);
  structFields.push_back(rewriter.getI32Type());

  return mlir::LLVM::LLVMStructType::getLiteral(ctx, structFields);
}

mlir::LogicalResult CIRToLLVMEhInflightOpLowering::matchAndRewrite(
    cir::EhInflightOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  auto llvmLandingPadStructTy = getLLVMLandingPadStructTy(rewriter);
  mlir::ArrayAttr symListAttr = op.getSymTypeListAttr();
  mlir::SmallVector<mlir::Value, 4> symAddrs;

  auto llvmFn = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  assert(llvmFn && "expected LLVM function parent");
  mlir::Block *entryBlock = &llvmFn.getRegion().front();
  assert(entryBlock->isEntryBlock());

  // %x = landingpad { ptr, i32 }
  // Note that since llvm.landingpad has to be the first operation on the
  // block, any needed value for its operands has to be added somewhere else.
  if (symListAttr) {
    //   catch ptr @_ZTIi
    //   catch ptr @_ZTIPKc
    for (mlir::Attribute attr : op.getSymTypeListAttr()) {
      auto symAttr = cast<mlir::FlatSymbolRefAttr>(attr);
      // Generate `llvm.mlir.addressof` for each symbol, and place those
      // operations in the LLVM function entry basic block.
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);
      mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
          loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
          symAttr.getValue());
      symAddrs.push_back(addrOp);
    }
  } else {
    if (!op.getCleanup()) {
      //   catch ptr null
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);
      mlir::Value nullOp = rewriter.create<mlir::LLVM::ZeroOp>(
          loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
      symAddrs.push_back(nullOp);
    }
  }

  // %slot = extractvalue { ptr, i32 } %x, 0
  // %selector = extractvalue { ptr, i32 } %x, 1
  auto padOp = rewriter.create<mlir::LLVM::LandingpadOp>(
      loc, llvmLandingPadStructTy, symAddrs);
  SmallVector<int64_t> slotIdx = {0};
  SmallVector<int64_t> selectorIdx = {1};

  if (op.getCleanup())
    padOp.setCleanup(true);

  mlir::Value slot =
      rewriter.create<mlir::LLVM::ExtractValueOp>(loc, padOp, slotIdx);
  mlir::Value selector =
      rewriter.create<mlir::LLVM::ExtractValueOp>(loc, padOp, selectorIdx);

  rewriter.replaceOp(op, mlir::ValueRange{slot, selector});

  // Landing pads are required to be in LLVM functions with personality
  // attribute. FIXME: for now hardcode personality creation in order to start
  // adding exception tests, once we annotate CIR with such information,
  // change it to be in FuncOp lowering instead.
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    // Insert personality decl before the current function.
    rewriter.setInsertionPoint(llvmFn);
    auto personalityFnTy =
        mlir::LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {},
                                          /*isVarArg=*/true);
    // Get or create `__gxx_personality_v0`
    StringRef fnName = "__gxx_personality_v0";
    getOrCreateLLVMFuncOp(rewriter, op, fnName, personalityFnTy);
    llvmFn.setPersonality(fnName);
  }
  return mlir::success();
}

void CIRToLLVMAllocaOpLowering::buildAllocaAnnotations(
    mlir::LLVM::AllocaOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter,
    mlir::ArrayAttr annotationValuesArray) const {
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  mlir::OpBuilder globalVarBuilder(module.getContext());

  mlir::OpBuilder::InsertPoint afterAlloca = rewriter.saveInsertionPoint();
  globalVarBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());

  mlir::Location loc = op.getLoc();
  mlir::OpBuilder varInitBuilder(module.getContext());
  varInitBuilder.restoreInsertionPoint(afterAlloca);

  auto intrinRetTy = mlir::LLVM::LLVMVoidType::get(getContext());
  constexpr const char *intrinNameAttr = "llvm.var.annotation.p0.p0";
  for (mlir::Attribute entry : annotationValuesArray) {
    SmallVector<mlir::Value, 4> intrinsicArgs;
    intrinsicArgs.push_back(op.getRes());
    auto annot = cast<cir::AnnotationAttr>(entry);
    lowerAnnotationValue(loc, loc, annot, module, varInitBuilder,
                         globalVarBuilder, stringGlobalsMap,
                         argStringGlobalsMap, argsVarMap, intrinsicArgs);
    rewriter.create<mlir::LLVM::CallIntrinsicOp>(
        loc, intrinRetTy, mlir::StringAttr::get(getContext(), intrinNameAttr),
        intrinsicArgs);
  }
}

mlir::LogicalResult CIRToLLVMAllocaOpLowering::matchAndRewrite(
    cir::AllocaOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value size =
      op.isDynamic() ? adaptor.getDynAllocSize()
                     : rewriter.create<mlir::LLVM::ConstantOp>(
                           op.getLoc(),
                           typeConverter->convertType(rewriter.getIndexType()),
                           rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  auto elementTy =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getAllocaType());
  auto resultTy = getTypeConverter()->convertType(op.getType());
  // Verification between the CIR alloca AS and the one from data layout.
  {
    auto resPtrTy = mlir::cast<mlir::LLVM::LLVMPointerType>(resultTy);
    auto dlAllocaASAttr = mlir::cast_if_present<mlir::IntegerAttr>(
        dataLayout.getAllocaMemorySpace());
    // Absence means 0
    // TODO: The query for the alloca AS should be done through CIRDataLayout
    // instead to reuse the logic of interpret null attr as 0.
    auto dlAllocaAS = dlAllocaASAttr ? dlAllocaASAttr.getInt() : 0;
    if (dlAllocaAS != resPtrTy.getAddressSpace()) {
      return op.emitError() << "alloca address space doesn't match the one "
                               "from the target data layout: "
                            << dlAllocaAS;
    }
  }

  // If there are annotations available, copy them out before we destroy the
  // original cir.alloca.
  mlir::ArrayAttr annotations;
  if (op.getAnnotations())
    annotations = op.getAnnotationsAttr();

  auto llvmAlloca = rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
      op, resultTy, elementTy, size, op.getAlignmentAttr().getInt());

  if (annotations && !annotations.empty())
    buildAllocaAnnotations(llvmAlloca, adaptor, rewriter, annotations);
  return mlir::success();
}

mlir::LLVM::AtomicOrdering
getLLVMMemOrder(std::optional<cir::MemOrder> &memorder) {
  if (!memorder)
    return mlir::LLVM::AtomicOrdering::not_atomic;
  switch (*memorder) {
  case cir::MemOrder::Relaxed:
    return mlir::LLVM::AtomicOrdering::monotonic;
  case cir::MemOrder::Consume:
  case cir::MemOrder::Acquire:
    return mlir::LLVM::AtomicOrdering::acquire;
  case cir::MemOrder::Release:
    return mlir::LLVM::AtomicOrdering::release;
  case cir::MemOrder::AcquireRelease:
    return mlir::LLVM::AtomicOrdering::acq_rel;
  case cir::MemOrder::SequentiallyConsistent:
    return mlir::LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown memory order");
}

static bool isLoadOrStoreInvariant(mlir::Value addr) {
  if (auto addrAllocaOp = addr.getDefiningOp<cir::AllocaOp>())
    return addrAllocaOp.getConstant();
  if (addr.getDefiningOp<cir::InvariantGroupOp>())
    return true;
  return false;
}

mlir::LogicalResult CIRToLLVMLoadOpLowering::matchAndRewrite(
    cir::LoadOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto llvmTy =
      convertTypeForMemory(*getTypeConverter(), dataLayout, op.getType());
  auto memorder = op.getMemOrder();
  auto ordering = getLLVMMemOrder(memorder);
  auto alignOpt = op.getAlignment();
  unsigned alignment = 0;
  if (!alignOpt) {
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    alignment = (unsigned)layout.getTypeABIAlignment(llvmTy);
  } else {
    alignment = *alignOpt;
  }

  auto invariant = false;
  // Under -O1 or higher optimization levels, add the invariant metadata if the
  // load operation loads from a constant object.
  if (lowerMod && lowerMod->getContext().getCodeGenOpts().OptimizationLevel > 0)
    invariant = isLoadOrStoreInvariant(op.getAddr());

  // TODO: nontemporal, syncscope.
  auto newLoad = rewriter.create<mlir::LLVM::LoadOp>(
      op->getLoc(), llvmTy, adaptor.getAddr(), /* alignment */ alignment,
      op.getIsVolatile(), /* nontemporal */ op.getIsNontemporal(),
      /* invariant */ false, /* invariantGroup */ invariant, ordering);

  // Convert adapted result to its original type if needed.
  mlir::Value result =
      emitFromMemory(rewriter, dataLayout, op, newLoad.getResult());
  rewriter.replaceOp(op, result);
  if (auto tbaa = op.getTbaaAttr()) {
    newLoad.setTBAATags(lowerCIRTBAAAttr(tbaa, rewriter, lowerMod));
  }
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMStoreOpLowering::matchAndRewrite(
    cir::StoreOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto memorder = op.getMemOrder();
  auto ordering = getLLVMMemOrder(memorder);
  auto alignOpt = op.getAlignment();
  unsigned alignment = 0;

  if (!alignOpt) {
    const auto llvmTy =
        getTypeConverter()->convertType(op.getValue().getType());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    alignment = (unsigned)layout.getTypeABIAlignment(llvmTy);
  } else {
    alignment = *alignOpt;
  }

  auto invariant = false;
  // Under -O1 or higher optimization levels, add the invariant metadata if the
  // store operation stores to a constant object.
  if (lowerMod && lowerMod->getContext().getCodeGenOpts().OptimizationLevel > 0)
    invariant = isLoadOrStoreInvariant(op.getAddr());

  // Convert adapted value to its memory type if needed.
  mlir::Value value = emitToMemory(rewriter, dataLayout,
                                   op.getValue().getType(), adaptor.getValue());
  // TODO: nontemporal, syncscope.
  auto storeOp = rewriter.create<mlir::LLVM::StoreOp>(
      op->getLoc(), value, adaptor.getAddr(), alignment, op.getIsVolatile(),
      /* nontemporal */ op.getIsNontemporal(), /* invariantGroup */ invariant,
      ordering);
  rewriter.replaceOp(op, storeOp);
  if (auto tbaa = op.getTbaaAttr()) {
    storeOp.setTBAATags(lowerCIRTBAAAttr(tbaa, rewriter, lowerMod));
  }
  return mlir::LogicalResult::success();
}

bool hasTrailingZeros(cir::ConstArrayAttr attr) {
  auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr.getElts());
  return attr.hasTrailingZeros() ||
         (array && std::count_if(array.begin(), array.end(), [](auto elt) {
            auto ar = dyn_cast<cir::ConstArrayAttr>(elt);
            return ar && hasTrailingZeros(ar);
          }));
}

mlir::LogicalResult CIRToLLVMConstantOpLowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Attribute attr = op.getValue();

  // Regardless of the type, we should lower the constant of poison value
  // into PoisonOp.
  if (auto poisonAttr = mlir::dyn_cast<cir::PoisonAttr>(attr)) {
    rewriter.replaceOp(op, lowerCirAttrAsValue(op, poisonAttr, rewriter,
                                               getTypeConverter(), dataLayout));
    return mlir::success();
  }

  if (mlir::isa<mlir::IntegerType>(op.getType())) {
    // Verified cir.const operations cannot actually be of these types, but the
    // lowering pass may generate temporary cir.const operations with these
    // types. This is OK since MLIR allows unverified operations to be alive
    // during a pass as long as they don't live past the end of the pass.
    attr = op.getValue();
  } else if (mlir::isa<cir::BoolType>(op.getType())) {
    int value = mlir::cast<cir::BoolAttr>(op.getValue()).getValue();
    attr = rewriter.getIntegerAttr(typeConverter->convertType(op.getType()),
                                   value);
  } else if (mlir::isa<cir::IntType>(op.getType())) {
    // Lower GlobalAddrAttr to llvm.mlir.addressof + llvm.mlir.ptrtoint
    if (auto ga = mlir::dyn_cast<cir::GlobalViewAttr>(op.getValue())) {
      auto newOp =
          lowerCirAttrAsValue(op, ga, rewriter, getTypeConverter(), dataLayout);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }

    attr = rewriter.getIntegerAttr(
        typeConverter->convertType(op.getType()),
        mlir::cast<cir::IntAttr>(op.getValue()).getValue());
  } else if (mlir::isa<cir::FPTypeInterface>(op.getType())) {
    attr = rewriter.getFloatAttr(
        typeConverter->convertType(op.getType()),
        mlir::cast<cir::FPAttr>(op.getValue()).getValue());
  } else if (auto complexTy = mlir::dyn_cast<cir::ComplexType>(op.getType())) {
    auto complexAttr = mlir::cast<cir::ComplexAttr>(op.getValue());
    auto complexElemTy = complexTy.getElementType();
    auto complexElemLLVMTy = typeConverter->convertType(complexElemTy);

    mlir::Attribute components[2];
    if (mlir::isa<cir::IntType>(complexElemTy)) {
      components[0] = rewriter.getIntegerAttr(
          complexElemLLVMTy,
          mlir::cast<cir::IntAttr>(complexAttr.getReal()).getValue());
      components[1] = rewriter.getIntegerAttr(
          complexElemLLVMTy,
          mlir::cast<cir::IntAttr>(complexAttr.getImag()).getValue());
    } else {
      components[0] = rewriter.getFloatAttr(
          complexElemLLVMTy,
          mlir::cast<cir::FPAttr>(complexAttr.getReal()).getValue());
      components[1] = rewriter.getFloatAttr(
          complexElemLLVMTy,
          mlir::cast<cir::FPAttr>(complexAttr.getImag()).getValue());
    }

    attr = rewriter.getArrayAttr(components);
  } else if (mlir::isa<cir::PointerType>(op.getType())) {
    // Optimize with dedicated LLVM op for null pointers.
    if (mlir::isa<cir::ConstPtrAttr>(op.getValue())) {
      if (mlir::cast<cir::ConstPtrAttr>(op.getValue()).isNullValue()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
            op, typeConverter->convertType(op.getType()));
        return mlir::success();
      }
    }
    // Lower GlobalViewAttr to llvm.mlir.addressof
    if (auto gv = mlir::dyn_cast<cir::GlobalViewAttr>(op.getValue())) {
      auto newOp =
          lowerCirAttrAsValue(op, gv, rewriter, getTypeConverter(), dataLayout);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    attr = op.getValue();
  } else if (mlir::isa<cir::DataMemberType>(op.getType())) {
    assert(lowerMod && "lower module is not available");
    auto dataMember = mlir::cast<cir::DataMemberAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerMod->getCXXABI().lowerDataMemberConstant(
        dataMember, layout, *typeConverter);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  } else if (mlir::isa<cir::MethodType>(op.getType())) {
    assert(lowerMod && "lower module is not available");
    auto method = mlir::cast<cir::MethodAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerMod->getCXXABI().lowerMethodConstant(
        method, layout, *typeConverter);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }
  // TODO(cir): constant arrays are currently just pushed into the stack using
  // the store instruction, instead of being stored as global variables and
  // then memcopyied into the stack (as done in Clang).
  else if (auto arrTy = mlir::dyn_cast<cir::ArrayType>(op.getType())) {
    // Fetch operation constant array initializer.

    auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(op.getValue());
    if (!constArr && !isa<cir::ZeroAttr, cir::UndefAttr>(op.getValue()))
      return op.emitError() << "array does not have a constant initializer";

    std::optional<mlir::Attribute> denseAttr;
    if (constArr && hasTrailingZeros(constArr)) {
      auto newOp = lowerCirAttrAsValue(op, constArr, rewriter,
                                       getTypeConverter(), dataLayout);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    } else if (constArr &&
               (denseAttr = lowerConstArrayAttr(constArr, typeConverter))) {
      attr = denseAttr.value();
    } else {
      auto initVal = lowerCirAttrAsValue(op, op.getValue(), rewriter,
                                         typeConverter, dataLayout);
      rewriter.replaceAllUsesWith(op, initVal);
      rewriter.eraseOp(op);
      return mlir::success();
    }
  }

  else if (const auto recordAttr =
               mlir::dyn_cast<cir::ConstRecordAttr>(op.getValue())) {
    // TODO(cir): this diverges from traditional lowering. Normally the
    // initializer would be a global constant that is memcopied. Here we just
    // define a local constant with llvm.undef that will be stored into the
    // stack.
    auto initVal = lowerCirAttrAsValue(op, recordAttr, rewriter, typeConverter,
                                       dataLayout);
    rewriter.replaceOp(op, initVal);
    return mlir::success();
  } else if (auto strTy = mlir::dyn_cast<cir::RecordType>(op.getType())) {
    auto attr = op.getValue();
    if (mlir::isa<cir::ZeroAttr, cir::UndefAttr>(attr)) {
      auto initVal =
          lowerCirAttrAsValue(op, attr, rewriter, typeConverter, dataLayout);
      rewriter.replaceOp(op, initVal);
      return mlir::success();
    }

    return op.emitError() << "unsupported lowering for record constant type "
                          << op.getType();
  } else if (const auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType())) {
    rewriter.replaceOp(op, lowerCirAttrAsValue(op, op.getValue(), rewriter,
                                               getTypeConverter(), dataLayout));
    return mlir::success();
  } else
    return op.emitError() << "unsupported constant type " << op.getType();

  rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
      op, getTypeConverter()->convertType(op.getType()), attr);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecCreateOpLowering::matchAndRewrite(
    cir::VecCreateOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Start with an 'undef' value for the vector.  Then 'insertelement' for
  // each of the vector elements.
  auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType());
  assert(vecTy && "result type of cir.vec.create op is not VectorType");
  auto llvmTy = typeConverter->convertType(vecTy);
  auto loc = op.getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::PoisonOp>(loc, llvmTy);
  assert(vecTy.getSize() == op.getElements().size() &&
         "cir.vec.create op count doesn't match vector type elements count");

  for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
    mlir::Value indexValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), i);
    result = rewriter.create<mlir::LLVM::InsertElementOp>(
        loc, result, adaptor.getElements()[i], indexValue);
  }
  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecCmpOpLowering::matchAndRewrite(
    cir::VecCmpOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // LLVM IR vector comparison returns a vector of i1.  This one-bit vector
  // must be sign-extended to the correct result type.
  auto elementType = elementTypeIfVector(op.getLhs().getType());
  mlir::Value bitResult;
  if (auto intType = mlir::dyn_cast<cir::IntType>(elementType)) {
    bitResult = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(),
        convertCmpKindToICmpPredicate(op.getKind(), intType.isSigned()),
        adaptor.getLhs(), adaptor.getRhs());
  } else if (mlir::isa<cir::FPTypeInterface>(elementType)) {
    bitResult = rewriter.create<mlir::LLVM::FCmpOp>(
        op.getLoc(), convertCmpKindToFCmpPredicate(op.getKind()),
        adaptor.getLhs(), adaptor.getRhs());
  } else {
    return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
  }

  // Check if the types are the same before generating SExtOp
  auto targetType = typeConverter->convertType(op.getType());
  if (bitResult.getType() == targetType)
    rewriter.replaceOp(op, bitResult);
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(op, targetType, bitResult);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecSplatOpLowering::matchAndRewrite(
    cir::VecSplatOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Vector splat can be implemented with an `insertelement` and a
  // `shufflevector`, which is better than an `insertelement` for each
  // element in the vector. Start with an undef vector. Insert the value into
  // the first element. Then use a `shufflevector` with a mask of all 0 to
  // fill out the entire vector with that value.
  auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType());
  assert(vecTy && "result type of cir.vec.splat op is not VectorType");
  auto llvmTy = typeConverter->convertType(vecTy);
  auto loc = op.getLoc();
  mlir::Value poison = rewriter.create<mlir::LLVM::PoisonOp>(loc, llvmTy);
  mlir::Value indexValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
  mlir::Value elementValue = adaptor.getValue();
  if (elementValue.getDefiningOp<mlir::LLVM::PoisonOp>()) {
    // If the splat value is poison, then we can just use poison value
    // for the entire vector.
    rewriter.replaceOp(op, poison);
    return mlir::success();
  }
  mlir::Value oneElement = rewriter.create<mlir::LLVM::InsertElementOp>(
      loc, poison, elementValue, indexValue);
  SmallVector<int32_t> zeroValues(vecTy.getSize(), 0);
  mlir::Value shuffled = rewriter.create<mlir::LLVM::ShuffleVectorOp>(
      loc, oneElement, poison, zeroValues);
  rewriter.replaceOp(op, shuffled);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecTernaryOpLowering::matchAndRewrite(
    cir::VecTernaryOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Convert `cond` into a vector of i1, then use that in a `select` op.
  mlir::Value bitVec = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(), mlir::LLVM::ICmpPredicate::ne, adaptor.getCond(),
      rewriter.create<mlir::LLVM::ZeroOp>(
          op.getCond().getLoc(),
          typeConverter->convertType(op.getCond().getType())));
  rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
      op, bitVec, adaptor.getLhs(), adaptor.getRhs());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecShuffleOpLowering::matchAndRewrite(
    cir::VecShuffleOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // LLVM::ShuffleVectorOp takes an ArrayRef of int for the list of indices.
  // Convert the ClangIR ArrayAttr of IntAttr constants into a
  // SmallVector<int>.
  SmallVector<int, 8> indices;
  std::transform(
      op.getIndices().begin(), op.getIndices().end(),
      std::back_inserter(indices), [](mlir::Attribute intAttr) {
        return mlir::cast<cir::IntAttr>(intAttr).getValue().getSExtValue();
      });
  rewriter.replaceOpWithNewOp<mlir::LLVM::ShuffleVectorOp>(
      op, adaptor.getVec1(), adaptor.getVec2(), indices);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecShuffleDynamicOpLowering::matchAndRewrite(
    cir::VecShuffleDynamicOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // LLVM IR does not have an operation that corresponds to this form of
  // the built-in.
  //     __builtin_shufflevector(V, I)
  // is implemented as this pseudocode, where the for loop is unrolled
  // and N is the number of elements:
  //
  // result = undef
  // maskbits = NextPowerOf2(N - 1)
  // masked = I & maskbits
  // for (i in 0 <= i < N)
  //    result[i] = V[masked[i]]
  auto loc = op.getLoc();
  mlir::Value input = adaptor.getVec();
  mlir::Type llvmIndexVecType =
      getTypeConverter()->convertType(op.getIndices().getType());
  mlir::Type llvmIndexType = getTypeConverter()->convertType(
      elementTypeIfVector(op.getIndices().getType()));
  uint64_t numElements =
      mlir::cast<cir::VectorType>(op.getVec().getType()).getSize();
  uint64_t maskBits = llvm::NextPowerOf2(numElements - 1) - 1;
  mlir::Value maskValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmIndexType, maskBits);
  mlir::Value maskVector =
      rewriter.create<mlir::LLVM::UndefOp>(loc, llvmIndexVecType);
  for (uint64_t i = 0; i < numElements; ++i) {
    mlir::Value iValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), i);
    maskVector = rewriter.create<mlir::LLVM::InsertElementOp>(
        loc, maskVector, maskValue, iValue);
  }
  mlir::Value maskedIndices = rewriter.create<mlir::LLVM::AndOp>(
      loc, llvmIndexVecType, adaptor.getIndices(), maskVector);
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(
      loc, getTypeConverter()->convertType(op.getVec().getType()));
  for (uint64_t i = 0; i < numElements; ++i) {
    mlir::Value iValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), i);
    mlir::Value indexValue = rewriter.create<mlir::LLVM::ExtractElementOp>(
        loc, maskedIndices, iValue);
    mlir::Value valueAtIndex =
        rewriter.create<mlir::LLVM::ExtractElementOp>(loc, input, indexValue);
    result = rewriter.create<mlir::LLVM::InsertElementOp>(loc, result,
                                                          valueAtIndex, iValue);
  }
  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVAStartOpLowering::matchAndRewrite(
    cir::VAStartOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
  auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), opaquePtr,
                                                       adaptor.getArgList());
  rewriter.replaceOpWithNewOp<mlir::LLVM::VaStartOp>(op, vaList);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVAEndOpLowering::matchAndRewrite(
    cir::VAEndOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
  auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), opaquePtr,
                                                       adaptor.getArgList());
  rewriter.replaceOpWithNewOp<mlir::LLVM::VaEndOp>(op, vaList);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVACopyOpLowering::matchAndRewrite(
    cir::VACopyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
  auto dstList = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), opaquePtr,
                                                        adaptor.getDstList());
  auto srcList = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), opaquePtr,
                                                        adaptor.getSrcList());
  rewriter.replaceOpWithNewOp<mlir::LLVM::VaCopyOp>(op, dstList, srcList);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVAArgOpLowering::matchAndRewrite(
    cir::VAArgOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  return op.emitError("cir.vaarg lowering is NYI");
}

/// Returns the name used for the linkage attribute. This *must* correspond
/// to the name of the attribute in ODS.
StringRef CIRToLLVMFuncOpLowering::getLinkageAttrNameString() {
  return "linkage";
}

/// Convert the `cir.func` attributes to `llvm.func` attributes.
/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out
/// argument attributes.
void CIRToLLVMFuncOpLowering::lowerFuncAttributes(
    cir::FuncOp func, bool filterArgAndResAttrs,
    SmallVectorImpl<mlir::NamedAttribute> &result) const {
  for (auto attr : func->getAttrs()) {
    StringRef name = attr.getName();
    if (name == mlir::SymbolTable::getSymbolAttrName() ||
        name == func.getFunctionTypeAttrName() ||
        name == getLinkageAttrNameString() ||
        name == func.getCallingConvAttrName() ||
        name == func.getDsoLocalAttrName() ||
        (filterArgAndResAttrs && (name == func.getArgAttrsAttrName() ||
                                  name == func.getResAttrsAttrName())))
      continue;

    // `CIRDialectLLVMIRTranslationInterface` requires "cir." prefix for
    // dialect specific attributes, rename them.
    if (name == func.getExtraAttrsAttrName()) {
      std::string cirName = "cir." + func.getExtraAttrsAttrName().str();
      attr.setName(mlir::StringAttr::get(getContext(), cirName));

      lowerFuncOpenCLKernelMetadata(attr);
    }
    result.push_back(attr);
  }
}

/// When do module translation, we can only translate LLVM-compatible types.
/// Here we lower possible OpenCLKernelMetadataAttr to use the converted type.
void CIRToLLVMFuncOpLowering::lowerFuncOpenCLKernelMetadata(
    mlir::NamedAttribute &extraAttrsEntry) const {
  const auto attrKey = cir::OpenCLKernelMetadataAttr::getMnemonic();
  auto oldExtraAttrs =
      cast<cir::ExtraFuncAttributesAttr>(extraAttrsEntry.getValue());
  if (!oldExtraAttrs.getElements().contains(attrKey))
    return;

  mlir::NamedAttrList newExtraAttrs;
  for (auto entry : oldExtraAttrs.getElements()) {
    if (entry.getName() == attrKey) {
      auto clKernelMetadata =
          cast<cir::OpenCLKernelMetadataAttr>(entry.getValue());
      if (auto vecTypeHint = clKernelMetadata.getVecTypeHint()) {
        auto newType = typeConverter->convertType(vecTypeHint.getValue());
        auto newTypeHint = mlir::TypeAttr::get(newType);
        auto newCLKMAttr = cir::OpenCLKernelMetadataAttr::get(
            getContext(), clKernelMetadata.getWorkGroupSizeHint(),
            clKernelMetadata.getReqdWorkGroupSize(), newTypeHint,
            clKernelMetadata.getVecTypeHintSignedness(),
            clKernelMetadata.getIntelReqdSubGroupSize());
        entry.setValue(newCLKMAttr);
      }
    }
    newExtraAttrs.push_back(entry);
  }
  extraAttrsEntry.setValue(cir::ExtraFuncAttributesAttr::get(
      newExtraAttrs.getDictionary(getContext())));
}

mlir::LogicalResult CIRToLLVMFuncOpLowering::matchAndRewriteAlias(
    cir::FuncOp op, mlir::FlatSymbolRefAttr aliasee, mlir::Type ty,
    OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const {
  SmallVector<mlir::NamedAttribute, 4> attributes;
  lowerFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

  auto loc = op.getLoc();
  auto aliasOp = rewriter.replaceOpWithNewOp<mlir::LLVM::AliasOp>(
      op, ty, convertLinkage(op.getLinkage()), op.getName(), op.getDsoLocal(),
      /*threadLocal=*/false, attributes);

  // Create the alias body
  mlir::OpBuilder builder(op.getContext());
  mlir::Block *block = builder.createBlock(&aliasOp.getInitializerRegion());
  builder.setInsertionPointToStart(block);
  // The type of AddressOfOp is always a pointer.
  assert(!cir::MissingFeatures::addressSpace());
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ty.getContext());
  auto addrOp =
      builder.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, aliasee.getValue());
  builder.create<mlir::LLVM::ReturnOp>(loc, addrOp);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMFuncOpLowering::matchAndRewrite(
    cir::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto fnType = op.getFunctionType();
  auto isDsoLocal = op.getDsoLocal();
  mlir::TypeConverter::SignatureConversion signatureConversion(
      fnType.getNumInputs());

  for (const auto &argType : enumerate(fnType.getInputs())) {
    auto convertedType = typeConverter->convertType(argType.value());
    if (!convertedType)
      return mlir::failure();
    signatureConversion.addInputs(argType.index(), convertedType);
  }

  mlir::Type resultType =
      getTypeConverter()->convertType(fnType.getReturnType());

  // Create the LLVM function operation.
  auto llvmFnTy = mlir::LLVM::LLVMFunctionType::get(
      resultType ? resultType : mlir::LLVM::LLVMVoidType::get(getContext()),
      signatureConversion.getConvertedTypes(),
      /*isVarArg=*/fnType.isVarArg());

  // If this is an alias, it needs to be lowered to llvm::AliasOp.
  std::optional<mlir::FlatSymbolRefAttr> aliasee = op.getAliaseeAttr();
  if (aliasee && *aliasee)
    return matchAndRewriteAlias(op, *aliasee, llvmFnTy, adaptor, rewriter);

  // LLVMFuncOp expects a single FileLine Location instead of a fused
  // location.
  auto Loc = op.getLoc();
  if (mlir::isa<mlir::FusedLoc>(Loc)) {
    auto FusedLoc = mlir::cast<mlir::FusedLoc>(Loc);
    Loc = FusedLoc.getLocations()[0];
  }
  assert((mlir::isa<mlir::FileLineColLoc>(Loc) ||
          mlir::isa<mlir::UnknownLoc>(Loc)) &&
         "expected single location or unknown location here");

  auto linkage = convertLinkage(op.getLinkage());
  auto cconv = convertCallingConv(op.getCallingConv());
  SmallVector<mlir::NamedAttribute, 4> attributes;
  lowerFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

  auto fn = rewriter.create<mlir::LLVM::LLVMFuncOp>(
      Loc, op.getName(), llvmFnTy, linkage, isDsoLocal, cconv,
      mlir::SymbolRefAttr(), attributes);

  // Lower CIR attributes for arguments.
  for (unsigned index = 0; index < fnType.getNumInputs(); index++) {
    mlir::DictionaryAttr cirAttrs = op.getArgAttrDict(index);
    if (cirAttrs) {
      if (cirAttrs.get(cir::CIRDialect::getZExtAttrName()))
        fn.setArgAttr(index, mlir::LLVM::LLVMDialect::getZExtAttrName(),
                      rewriter.getUnitAttr());
      if (cirAttrs.get(cir::CIRDialect::getSExtAttrName()))
        fn.setArgAttr(index, mlir::LLVM::LLVMDialect::getSExtAttrName(),
                      rewriter.getUnitAttr());
    }
  }

  // Lower CIR attributes for return value.
  // Each function has no more than one result in CIR, so no need for a loop.
  assert(op.getNumResults() <= 1);
  if (op.getNumResults() > 0) {
    mlir::DictionaryAttr resultAttrs = op.getResultAttrDict(0);
    if (resultAttrs) {
      if (resultAttrs.get(cir::CIRDialect::getZExtAttrName()))
        fn.setResultAttr(0, mlir::LLVM::LLVMDialect::getZExtAttrName(),
                         rewriter.getUnitAttr());
      if (resultAttrs.get(cir::CIRDialect::getSExtAttrName()))
        fn.setResultAttr(0, mlir::LLVM::LLVMDialect::getSExtAttrName(),
                         rewriter.getUnitAttr());
    }
  }

  fn.setVisibility_Attr(mlir::LLVM::VisibilityAttr::get(
      getContext(), lowerCIRVisibilityToLLVMVisibility(
                        op.getGlobalVisibilityAttr().getValue())));

  rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
  if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                         &signatureConversion)))
    return mlir::failure();

  rewriter.eraseOp(op);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMGetGlobalOpLowering::matchAndRewrite(
    cir::GetGlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
  // CIRGen should mitigate this and not emit the get_global.
  if (op->getUses().empty()) {
    rewriter.eraseOp(op);
    return mlir::success();
  }

  auto type = getTypeConverter()->convertType(op.getType());
  auto symbol = op.getName();
  mlir::Operation *newop =
      rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), type, symbol);

  if (op.getTls()) {
    // Handle access to TLS via intrinsic.
    newop = rewriter.create<mlir::LLVM::ThreadlocalAddressOp>(
        op.getLoc(), type, newop->getResult(0));
  }

  rewriter.replaceOp(op, newop);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexCreateOpLowering::matchAndRewrite(
    cir::ComplexCreateOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto complexLLVMTy = getTypeConverter()->convertType(op.getType());
  auto initialComplex =
      rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), complexLLVMTy);

  int64_t position[1]{0};
  auto realComplex = rewriter.create<mlir::LLVM::InsertValueOp>(
      op->getLoc(), initialComplex, adaptor.getReal(), position);

  position[0] = 1;
  auto complex = rewriter.create<mlir::LLVM::InsertValueOp>(
      op->getLoc(), realComplex, adaptor.getImag(), position);

  rewriter.replaceOp(op, complex);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexRealOpLowering::matchAndRewrite(
    cir::ComplexRealOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resultLLVMTy = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
      op, resultLLVMTy, adaptor.getOperand(), llvm::ArrayRef<std::int64_t>{0});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexImagOpLowering::matchAndRewrite(
    cir::ComplexImagOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resultLLVMTy = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
      op, resultLLVMTy, adaptor.getOperand(), llvm::ArrayRef<std::int64_t>{1});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexRealPtrOpLowering::matchAndRewrite(
    cir::ComplexRealPtrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  cir::PointerType operandTy = op.getOperand().getType();
  auto resultLLVMTy = getTypeConverter()->convertType(op.getType());
  auto elementLLVMTy = getTypeConverter()->convertType(operandTy.getPointee());

  mlir::LLVM::GEPArg gepIndices[2]{{0}, {0}};
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
      /*inbounds=*/true);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexImagPtrOpLowering::matchAndRewrite(
    cir::ComplexImagPtrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  cir::PointerType operandTy = op.getOperand().getType();
  auto resultLLVMTy = getTypeConverter()->convertType(op.getType());
  auto elementLLVMTy = getTypeConverter()->convertType(operandTy.getPointee());

  mlir::LLVM::GEPArg gepIndices[2]{{0}, {1}};
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
      /*inbounds=*/true);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMSwitchFlatOpLowering::matchAndRewrite(
    cir::SwitchFlatOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  llvm::SmallVector<mlir::APInt, 8> caseValues;
  if (op.getCaseValues()) {
    for (auto val : op.getCaseValues()) {
      auto intAttr = dyn_cast<cir::IntAttr>(val);
      caseValues.push_back(intAttr.getValue());
    }
  }

  llvm::SmallVector<mlir::Block *, 8> caseDestinations;
  llvm::SmallVector<mlir::ValueRange, 8> caseOperands;

  for (auto x : op.getCaseDestinations()) {
    caseDestinations.push_back(x);
  }

  for (auto x : op.getCaseOperands()) {
    caseOperands.push_back(x);
  }

  // Set switch op to branch to the newly created blocks.
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::LLVM::SwitchOp>(
      op, adaptor.getCondition(), op.getDefaultDestination(),
      op.getDefaultOperands(), caseValues, caseDestinations, caseOperands);
  return mlir::success();
}

llvm::SmallVector<mlir::NamedAttribute>
CIRToLLVMGlobalOpLowering::lowerGlobalAttributes(
    cir::GlobalOp op, mlir::ConversionPatternRewriter &rewriter) const {
  SmallVector<mlir::NamedAttribute> attributes;

  std::optional<mlir::StringRef> section = op.getSection();
  mlir::LLVM::VisibilityAttr visibility = mlir::LLVM::VisibilityAttr::get(
      getContext(), lowerCIRVisibilityToLLVMVisibility(
                        op.getGlobalVisibilityAttr().getValue()));

  if (section.has_value())
    attributes.push_back(rewriter.getNamedAttr(
        "section", rewriter.getStringAttr(section.value())));

  attributes.push_back(rewriter.getNamedAttr("visibility_", visibility));

  if (auto extInit =
          op->getAttr(CUDAExternallyInitializedAttr::getMnemonic())) {
    attributes.push_back(rewriter.getNamedAttr("externally_initialized",
                                               rewriter.getUnitAttr()));
  }
  return attributes;
}

mlir::LogicalResult CIRToLLVMGlobalOpLowering::lowerInitializer(
    mlir::ConversionPatternRewriter &rewriter, cir::GlobalOp op,
    mlir::Type llvmType, mlir::Attribute &init,
    bool &useInitializerRegion) const {
  if (!init)
    return mlir::success();

  if (mlir::isa<cir::FPAttr, cir::IntAttr, cir::BoolAttr>(init)) {
    // If a directly equivalent attribute is available, use it.
    return lowerInitializerDirect(rewriter, op, llvmType, init,
                                  useInitializerRegion);
  } else if (mlir::isa<cir::ZeroAttr, cir::ConstPtrAttr, cir::UndefAttr,
                       cir::ConstRecordAttr, cir::GlobalViewAttr,
                       cir::VTableAttr, cir::TypeInfoAttr>(init)) {
    // TODO(cir): once LLVM's dialect has proper equivalent attributes this
    // should be updated. For now, we use a custom op to initialize globals
    // to the appropriate value.
    useInitializerRegion = true;
    return mlir::success();
  } else if (mlir::isa<cir::ConstArrayAttr>(init)) {
    return lowerInitializerForConstArray(rewriter, op, init,
                                         useInitializerRegion);
  } else if (mlir::isa<cir::ConstVectorAttr>(init)) {
    return lowerInitializerForConstVector(rewriter, op, init,
                                          useInitializerRegion);
  } else if (mlir::isa<cir::ComplexAttr>(init)) {
    return lowerInitializerForConstComplex(rewriter, op, init,
                                           useInitializerRegion);
  } else if (auto dataMemberAttr = mlir::dyn_cast<cir::DataMemberAttr>(init)) {
    assert(lowerMod && "lower module is not available");
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerMod->getCXXABI().lowerDataMemberConstant(
        dataMemberAttr, layout, *typeConverter);
    init = abiValue;
    llvmType = convertTypeForMemory(*getTypeConverter(), dataLayout,
                                    abiValue.getType());
    // Recursively lower the CIR attribute produced by CXXABI.
    return lowerInitializer(rewriter, op, llvmType, init, useInitializerRegion);
  } else {
    op.emitError() << "unsupported initializer '" << init << "'";
    return mlir::failure();
  }
  llvm_unreachable("unreachable");
}

mlir::LogicalResult CIRToLLVMGlobalOpLowering::lowerInitializerForConstVector(
    mlir::ConversionPatternRewriter &rewriter, cir::GlobalOp op,
    mlir::Attribute &init, bool &useInitializerRegion) const {
  auto constVec = mlir::cast<cir::ConstVectorAttr>(init);
  if (const auto attr = mlir::dyn_cast<mlir::ArrayAttr>(constVec.getElts())) {
    if (auto val = lowerConstVectorAttr(constVec, getTypeConverter());
        val.has_value()) {
      init = val.value();
      useInitializerRegion = false;
    } else
      useInitializerRegion = true;
    return mlir::success();
  }

  op.emitError() << "unsupported lowering for #cir.const_vector with value "
                 << constVec.getElts();
  return mlir::failure();
}

mlir::LogicalResult CIRToLLVMGlobalOpLowering::lowerInitializerForConstComplex(
    mlir::ConversionPatternRewriter &rewriter, cir::GlobalOp op,
    mlir::Attribute &init, bool &useInitializerRegion) const {
  auto constVec = mlir::cast<cir::ComplexAttr>(init);
  if (auto val = lowerConstComplexAttr(constVec, getTypeConverter());
      val.has_value()) {
    init = val.value();
    useInitializerRegion = false;
  } else
    useInitializerRegion = true;
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGlobalOpLowering::lowerInitializerForConstArray(
    mlir::ConversionPatternRewriter &rewriter, cir::GlobalOp op,
    mlir::Attribute &init, bool &useInitializerRegion) const {
  auto constArr = mlir::cast<cir::ConstArrayAttr>(init);

  // Initializer is a constant array: convert it to a compatible llvm init.
  if (auto attr = mlir::dyn_cast<mlir::StringAttr>(constArr.getElts())) {
    llvm::SmallString<256> literal(attr.getValue());
    if (constArr.getTrailingZerosNum())
      literal.append(constArr.getTrailingZerosNum(), '\0');
    init = rewriter.getStringAttr(literal);
    useInitializerRegion = false;
    return mlir::success();
  } else if (auto attr = mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts())) {
    // If failed to use a compact attribute as an initializer, we initialize
    // elements individually.
    if (auto val = lowerConstArrayAttr(constArr, getTypeConverter());
        val.has_value()) {
      init = val.value();
      useInitializerRegion = false;
    } else
      useInitializerRegion = true;
    return mlir::success();
  } else {
    op.emitError() << "unsupported lowering for #cir.const_array with value "
                   << constArr.getElts();
    return mlir::failure();
  }
}
mlir::LogicalResult CIRToLLVMGlobalOpLowering::lowerInitializerDirect(
    mlir::ConversionPatternRewriter &rewriter, cir::GlobalOp op,
    mlir::Type llvmType, mlir::Attribute &init,
    bool &useInitializerRegion) const {
  init = llvm::TypeSwitch<mlir::Attribute, mlir::Attribute>(init)
             .Case<cir::FPAttr>([&](cir::FPAttr attr) {
               return rewriter.getFloatAttr(llvmType, attr.getValue());
             })
             .Case<cir::IntAttr>([&](cir::IntAttr attr) {
               return rewriter.getIntegerAttr(llvmType, attr.getValue());
             })
             .Case<cir::BoolAttr>([&](cir::BoolAttr attr) {
               return rewriter.getBoolAttr(attr.getValue());
             })
             .Default([&](mlir::Attribute attr) { return mlir::Attribute(); });
  useInitializerRegion = false;
  // If initRewriter returned a null attribute, init will have a value but
  // the value will be null. If that happens, initRewriter didn't handle the
  // attribute type. It probably needs to be added to
  // GlobalInitAttrRewriter.
  if (!init) {
    op.emitError() << "unsupported initializer '" << init << "'";
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGlobalOpLowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  // Fetch required values to create LLVM op.
  const auto cirSymType = op.getSymType();

  const auto llvmType =
      convertTypeForMemory(*getTypeConverter(), dataLayout, cirSymType);
  const auto isConst = op.getConstant();
  const auto isDsoLocal = op.getDsoLocal();
  const auto linkage = convertLinkage(op.getLinkage());
  const auto symbol = op.getSymName();
  mlir::Attribute init = op.getInitialValueAttr();

  SmallVector<mlir::NamedAttribute> attributes =
      lowerGlobalAttributes(op, rewriter);

  bool useInitializerRegion = false;
  if (lowerInitializer(rewriter, op, llvmType, init, useInitializerRegion)
          .failed())
    return mlir::failure();

  auto initValue = useInitializerRegion ? mlir::Attribute{} : init;
  // Rewrite op.
  auto llvmGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
      op, llvmType, isConst, linkage, symbol, initValue,
      /*alignment*/ op.getAlignment().value_or(0),
      /*addrSpace*/ getGlobalOpTargetAddrSpace(rewriter, typeConverter, op),
      /*dsoLocal*/ isDsoLocal, /*threadLocal*/ (bool)op.getTlsModelAttr(),
      /*comdat*/ mlir::SymbolRefAttr(), attributes);

  // Initialize the initializer region of LLVM global and update insertion point
  // to the end of the initializer block.
  if (useInitializerRegion) {
    assert(init && "Expected initializer to use initializer region");
    llvmGlobalOp.getInitializerRegion().push_back(new mlir::Block());
    rewriter.setInsertionPointToEnd(llvmGlobalOp.getInitializerBlock());

    rewriter.create<mlir::LLVM::ReturnOp>(
        op->getLoc(),
        lowerCirAttrAsValue(op, init, rewriter, typeConverter, dataLayout));
  }

  auto mod = op->getParentOfType<mlir::ModuleOp>();
  if (op.getComdat())
    addComdat(llvmGlobalOp, comdatOp, rewriter, mod);

  return mlir::success();
}

void CIRToLLVMGlobalOpLowering::addComdat(mlir::LLVM::GlobalOp &op,
                                          mlir::LLVM::ComdatOp &comdatOp,
                                          mlir::OpBuilder &builder,
                                          mlir::ModuleOp &module) {
  StringRef comdatName("__llvm_comdat_globals");
  if (!comdatOp) {
    builder.setInsertionPointToStart(module.getBody());
    comdatOp =
        builder.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
  }
  builder.setInsertionPointToStart(&comdatOp.getBody().back());
  auto selectorOp = builder.create<mlir::LLVM::ComdatSelectorOp>(
      comdatOp.getLoc(), op.getSymName(), mlir::LLVM::comdat::Comdat::Any);
  op.setComdatAttr(mlir::SymbolRefAttr::get(
      builder.getContext(), comdatName,
      mlir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
}

mlir::LogicalResult CIRToLLVMUnaryOpLowering::matchAndRewrite(
    cir::UnaryOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(op.getType() == op.getInput().getType() &&
         "Unary operation's operand type and result type are different");
  mlir::Type type = op.getType();
  mlir::Type elementType = elementTypeIfVector(type);
  bool IsVector = mlir::isa<cir::VectorType>(type);
  auto llvmType = getTypeConverter()->convertType(type);
  auto loc = op.getLoc();

  // Integer unary operations: + - ~ ++ --
  if (mlir::isa<cir::IntType>(elementType)) {
    auto overflowFlags = op.getNoSignedWrap()
                             ? mlir::LLVM::IntegerOverflowFlags::nsw
                             : mlir::LLVM::IntegerOverflowFlags::none;
    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      assert(!IsVector && "++ not allowed on vector types");
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, 1);
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(
          op, llvmType, adaptor.getInput(), One, overflowFlags);
      return mlir::success();
    }
    case cir::UnaryOpKind::Dec: {
      assert(!IsVector && "-- not allowed on vector types");
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, 1);
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(
          op, llvmType, adaptor.getInput(), One, overflowFlags);
      return mlir::success();
    }
    case cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Minus: {
      mlir::Value Zero;
      if (IsVector)
        Zero = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmType);
      else
        Zero = rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, 0);
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(
          op, Zero, adaptor.getInput(), overflowFlags);
      return mlir::success();
    }
    case cir::UnaryOpKind::Not: {
      // bit-wise compliment operator, implemented as an XOR with -1.
      mlir::Value minusOne;
      if (IsVector) {
        // Creating a vector object with all -1 values is easier said than
        // done. It requires a series of insertelement ops.
        mlir::Type llvmElementType =
            getTypeConverter()->convertType(elementType);
        auto MinusOneInt =
            rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmElementType, -1);
        minusOne = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmType);
        auto NumElements = mlir::dyn_cast<cir::VectorType>(type).getSize();
        for (uint64_t i = 0; i < NumElements; ++i) {
          mlir::Value indexValue = rewriter.create<mlir::LLVM::ConstantOp>(
              loc, rewriter.getI64Type(), i);
          minusOne = rewriter.create<mlir::LLVM::InsertElementOp>(
              loc, minusOne, MinusOneInt, indexValue);
        }
      } else {
        minusOne = rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, -1);
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, adaptor.getInput(),
                                                     minusOne);
      return mlir::success();
    }
    }
  }

  // Floating point unary operations: + - ++ --
  if (mlir::isa<cir::FPTypeInterface>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      assert(!IsVector && "++ not allowed on vector types");
      auto oneAttr = rewriter.getFloatAttr(llvmType, 1.0);
      auto oneConst =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, oneAttr);
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, oneConst,
                                                      adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Dec: {
      assert(!IsVector && "-- not allowed on vector types");
      auto negOneAttr = rewriter.getFloatAttr(llvmType, -1.0);
      auto negOneConst =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, negOneAttr);
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, negOneConst,
                                                      adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Plus:
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    case cir::UnaryOpKind::Minus: {
      rewriter.replaceOpWithNewOp<mlir::LLVM::FNegOp>(op, llvmType,
                                                      adaptor.getInput());
      return mlir::success();
    }
    default:
      return op.emitError()
             << "Unknown floating-point unary operation during CIR lowering";
    }
  }

  // Boolean unary operations: ! only. (For all others, the operand has
  // already been promoted to int.)
  if (mlir::isa<cir::BoolType>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Not:
      assert(!IsVector && "NYI: op! on vector mask");
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(
          op, adaptor.getInput(),
          rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, 1));
      return mlir::success();
    default:
      return op.emitError()
             << "Unknown boolean unary operation during CIR lowering";
    }
  }

  // Pointer unary operations: + only.  (++ and -- of pointers are implemented
  // with cir.ptr_stride, not cir.unary.)
  if (mlir::isa<cir::PointerType>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Plus:
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    default:
      op.emitError() << "Unknown pointer unary operation during CIR lowering";
      return mlir::failure();
    }
  }

  return op.emitError() << "Unary operation has unsupported type: "
                        << elementType;
}

mlir::LLVM::IntegerOverflowFlags
CIRToLLVMBinOpLowering::getIntOverflowFlag(cir::BinOp op) const {
  if (op.getNoUnsignedWrap())
    return mlir::LLVM::IntegerOverflowFlags::nuw;

  if (op.getNoSignedWrap())
    return mlir::LLVM::IntegerOverflowFlags::nsw;

  return mlir::LLVM::IntegerOverflowFlags::none;
}

static bool isIntTypeUnsigned(mlir::Type type) {
  // TODO: Ideally, we should only need to check cir::IntType here.
  return mlir::isa<cir::IntType>(type)
             ? mlir::cast<cir::IntType>(type).isUnsigned()
             : mlir::cast<mlir::IntegerType>(type).isUnsigned();
}

mlir::LogicalResult CIRToLLVMBinOpLowering::matchAndRewrite(
    cir::BinOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert((adaptor.getLhs().getType() == adaptor.getRhs().getType()) &&
         "inconsistent operands' types not supported yet");

  mlir::Type type = op.getRhs().getType();
  assert((mlir::isa<cir::IntType, cir::BoolType, cir::FPTypeInterface,
                    cir::VectorType, mlir::IntegerType>(type)) &&
         "operand type not supported yet");

  mlir::Type llvmTy = getTypeConverter()->convertType(op.getType());
  mlir::Type llvmEltTy = elementTypeIfVector(llvmTy);
  mlir::Value rhs = adaptor.getRhs();
  mlir::Value lhs = adaptor.getLhs();

  type = elementTypeIfVector(type);

  switch (op.getKind()) {
  case cir::BinOpKind::Add:
    if (mlir::isa<mlir::IntegerType>(llvmEltTy)) {
      if (op.getSaturated()) {
        if (isIntTypeUnsigned(type)) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::UAddSat>(op, lhs, rhs);
          break;
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::SAddSat>(op, lhs, rhs);
        break;
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmTy, lhs, rhs,
                                                     getIntOverflowFlag(op));
    } else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::Sub:
    if (mlir::isa<mlir::IntegerType>(llvmEltTy)) {
      if (op.getSaturated()) {
        if (isIntTypeUnsigned(type)) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::USubSat>(op, lhs, rhs);
          break;
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::SSubSat>(op, lhs, rhs);
        break;
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmTy, lhs, rhs,
                                                     getIntOverflowFlag(op));
    } else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::Mul:
    if (mlir::isa<mlir::IntegerType>(llvmEltTy))
      rewriter.replaceOpWithNewOp<mlir::LLVM::MulOp>(op, llvmTy, lhs, rhs,
                                                     getIntOverflowFlag(op));
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::Div:
    if (mlir::isa<mlir::IntegerType>(llvmEltTy)) {
      auto isUnsigned = isIntTypeUnsigned(type);
      if (isUnsigned)
        rewriter.replaceOpWithNewOp<mlir::LLVM::UDivOp>(op, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::SDivOp>(op, lhs, rhs);
    } else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FDivOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::Rem:
    if (mlir::isa<mlir::IntegerType>(llvmEltTy)) {
      auto isUnsigned = isIntTypeUnsigned(type);
      if (isUnsigned)
        rewriter.replaceOpWithNewOp<mlir::LLVM::URemOp>(op, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::SRemOp>(op, lhs, rhs);
    } else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FRemOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::And:
    rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::Or:
    rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::Xor:
    rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, lhs, rhs);
    break;
  case cir::BinOpKind::Max:
    if (mlir::isa<mlir::IntegerType>(llvmEltTy)) {
      auto isUnsigned = isIntTypeUnsigned(type);
      if (isUnsigned)
        rewriter.replaceOpWithNewOp<mlir::LLVM::UMaxOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::SMaxOp>(op, llvmTy, lhs, rhs);
    }
    break;
  }

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBinOpOverflowOpLowering::matchAndRewrite(
    cir::BinOpOverflowOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto arithKind = op.getKind();
  auto operandTy = op.getLhs().getType();
  auto resultTy = op.getResult().getType();

  auto encompassedTyInfo = computeEncompassedTypeWidth(operandTy, resultTy);
  auto encompassedLLVMTy = rewriter.getIntegerType(encompassedTyInfo.width);

  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (operandTy.getWidth() < encompassedTyInfo.width) {
    if (operandTy.isSigned()) {
      lhs = rewriter.create<mlir::LLVM::SExtOp>(loc, encompassedLLVMTy, lhs);
      rhs = rewriter.create<mlir::LLVM::SExtOp>(loc, encompassedLLVMTy, rhs);
    } else {
      lhs = rewriter.create<mlir::LLVM::ZExtOp>(loc, encompassedLLVMTy, lhs);
      rhs = rewriter.create<mlir::LLVM::ZExtOp>(loc, encompassedLLVMTy, rhs);
    }
  }

  auto intrinName = getLLVMIntrinName(arithKind, encompassedTyInfo.sign,
                                      encompassedTyInfo.width);
  auto intrinNameAttr = mlir::StringAttr::get(op.getContext(), intrinName);

  auto overflowLLVMTy = rewriter.getI1Type();
  auto intrinRetTy = mlir::LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {encompassedLLVMTy, overflowLLVMTy});

  auto callLLVMIntrinOp = rewriter.create<mlir::LLVM::CallIntrinsicOp>(
      loc, intrinRetTy, intrinNameAttr, mlir::ValueRange{lhs, rhs});
  auto intrinRet = callLLVMIntrinOp.getResult(0);

  auto result = rewriter
                    .create<mlir::LLVM::ExtractValueOp>(loc, intrinRet,
                                                        ArrayRef<int64_t>{0})
                    .getResult();
  auto overflow = rewriter
                      .create<mlir::LLVM::ExtractValueOp>(loc, intrinRet,
                                                          ArrayRef<int64_t>{1})
                      .getResult();

  if (resultTy.getWidth() < encompassedTyInfo.width) {
    auto resultLLVMTy = getTypeConverter()->convertType(resultTy);
    auto truncResult =
        rewriter.create<mlir::LLVM::TruncOp>(loc, resultLLVMTy, result);

    // Extend the truncated result back to the encompassing type to check for
    // any overflows during the truncation.
    mlir::Value truncResultExt;
    if (resultTy.isSigned())
      truncResultExt = rewriter.create<mlir::LLVM::SExtOp>(
          loc, encompassedLLVMTy, truncResult);
    else
      truncResultExt = rewriter.create<mlir::LLVM::ZExtOp>(
          loc, encompassedLLVMTy, truncResult);
    auto truncOverflow = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, truncResultExt, result);

    result = truncResult;
    overflow = rewriter.create<mlir::LLVM::OrOp>(loc, overflow, truncOverflow);
  }

  auto boolLLVMTy = getTypeConverter()->convertType(op.getOverflow().getType());
  if (boolLLVMTy != rewriter.getI1Type())
    overflow = rewriter.create<mlir::LLVM::ZExtOp>(loc, boolLLVMTy, overflow);

  rewriter.replaceOp(op, mlir::ValueRange{result, overflow});

  return mlir::success();
}

std::string CIRToLLVMBinOpOverflowOpLowering::getLLVMIntrinName(
    cir::BinOpOverflowKind opKind, bool isSigned, unsigned width) {
  // The intrinsic name is `@llvm.{s|u}{opKind}.with.overflow.i{width}`

  std::string name = "llvm.";

  if (isSigned)
    name.push_back('s');
  else
    name.push_back('u');

  switch (opKind) {
  case cir::BinOpOverflowKind::Add:
    name.append("add.");
    break;
  case cir::BinOpOverflowKind::Sub:
    name.append("sub.");
    break;
  case cir::BinOpOverflowKind::Mul:
    name.append("mul.");
    break;
  }

  name.append("with.overflow.i");
  name.append(std::to_string(width));

  return name;
}

CIRToLLVMBinOpOverflowOpLowering::EncompassedTypeInfo
CIRToLLVMBinOpOverflowOpLowering::computeEncompassedTypeWidth(
    cir::IntType operandTy, cir::IntType resultTy) {
  auto sign = operandTy.getIsSigned() || resultTy.getIsSigned();
  auto width = std::max(operandTy.getWidth() + (sign && operandTy.isUnsigned()),
                        resultTy.getWidth() + (sign && resultTy.isUnsigned()));
  return {sign, width};
}

mlir::LogicalResult CIRToLLVMShiftOpLowering::matchAndRewrite(
    cir::ShiftOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto cirAmtTy = mlir::dyn_cast<cir::IntType>(op.getAmount().getType());
  auto cirValTy = mlir::dyn_cast<cir::IntType>(op.getValue().getType());

  // Operands could also be vector type
  auto cirAmtVTy = mlir::dyn_cast<cir::VectorType>(op.getAmount().getType());
  auto cirValVTy = mlir::dyn_cast<cir::VectorType>(op.getValue().getType());
  auto llvmTy = getTypeConverter()->convertType(op.getType());
  mlir::Value amt = adaptor.getAmount();
  mlir::Value val = adaptor.getValue();

  assert(((cirValTy && cirAmtTy) || (cirAmtVTy && cirValVTy)) &&
         "shift input type must be integer or vector type, otherwise NYI");

  assert((cirValTy == op.getType() || cirValVTy == op.getType()) &&
         "inconsistent operands' types NYI");

  // Ensure shift amount is the same type as the value. Some undefined
  // behavior might occur in the casts below as per [C99 6.5.7.3].
  // Vector type shift amount needs no cast as type consistency is expected to
  // be already be enforced at CIRGen.
  // Negative shift amounts are undefined behavior so we can always zero extend
  // the integer here.
  if (cirAmtTy)
    amt = getLLVMIntCast(rewriter, amt, mlir::cast<mlir::IntegerType>(llvmTy),
                         true, cirAmtTy.getWidth(), cirValTy.getWidth());

  // Lower to the proper LLVM shift operation.
  if (op.getIsShiftleft())
    rewriter.replaceOpWithNewOp<mlir::LLVM::ShlOp>(op, llvmTy, val, amt);
  else {
    bool isUnSigned =
        cirValTy
            ? !cirValTy.isSigned()
            : !mlir::cast<cir::IntType>(cirValVTy.getElementType()).isSigned();
    if (isUnSigned)
      rewriter.replaceOpWithNewOp<mlir::LLVM::LShrOp>(op, llvmTy, val, amt);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::AShrOp>(op, llvmTy, val, amt);
  }

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCmpOpLowering::matchAndRewrite(
    cir::CmpOp cmpOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto type = cmpOp.getLhs().getType();

  if (mlir::isa<cir::DataMemberType, cir::MethodType>(type)) {
    assert(lowerMod && "lowering module is not available");

    mlir::Value loweredResult;
    if (mlir::isa<cir::DataMemberType>(type))
      loweredResult = lowerMod->getCXXABI().lowerDataMemberCmp(
          cmpOp, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    else
      loweredResult = lowerMod->getCXXABI().lowerMethodCmp(
          cmpOp, adaptor.getLhs(), adaptor.getRhs(), rewriter);

    rewriter.replaceOp(cmpOp, loweredResult);
    return mlir::success();
  }

  // Lower to LLVM comparison op.
  // if (auto intTy = mlir::dyn_cast<cir::IntType>(type)) {
  if (mlir::isa<cir::IntType, mlir::IntegerType>(type)) {
    auto isSigned = mlir::isa<cir::IntType>(type)
                        ? mlir::cast<cir::IntType>(type).isSigned()
                        : mlir::cast<mlir::IntegerType>(type).isSigned();
    auto kind = convertCmpKindToICmpPredicate(cmpOp.getKind(), isSigned);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        cmpOp, kind, adaptor.getLhs(), adaptor.getRhs());
  } else if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(type)) {
    auto kind = convertCmpKindToICmpPredicate(cmpOp.getKind(),
                                              /* isSigned=*/false);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        cmpOp, kind, adaptor.getLhs(), adaptor.getRhs());
  } else if (mlir::isa<cir::FPTypeInterface>(type)) {
    auto kind = convertCmpKindToFCmpPredicate(cmpOp.getKind());
    rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(
        cmpOp, kind, adaptor.getLhs(), adaptor.getRhs());
  } else {
    return cmpOp.emitError() << "unsupported type for CmpOp: " << type;
  }

  return mlir::success();
}

mlir::LLVM::CallIntrinsicOp
createCallLLVMIntrinsicOp(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Location loc, const llvm::Twine &intrinsicName,
                          mlir::Type resultTy, mlir::ValueRange operands) {
  auto intrinsicNameAttr =
      mlir::StringAttr::get(rewriter.getContext(), intrinsicName);
  return rewriter.create<mlir::LLVM::CallIntrinsicOp>(
      loc, resultTy, intrinsicNameAttr, operands);
}

mlir::LLVM::CallIntrinsicOp replaceOpWithCallLLVMIntrinsicOp(
    mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
    const llvm::Twine &intrinsicName, mlir::Type resultTy,
    mlir::ValueRange operands) {
  auto callIntrinOp = createCallLLVMIntrinsicOp(
      rewriter, op->getLoc(), intrinsicName, resultTy, operands);
  rewriter.replaceOp(op, callIntrinOp.getOperation());
  return callIntrinOp;
}

mlir::LogicalResult CIRToLLVMLLVMIntrinsicCallOpLowering::matchAndRewrite(
    cir::LLVMIntrinsicCallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type llvmResTy =
      getTypeConverter()->convertType(op->getResultTypes()[0]);
  if (!llvmResTy)
    return op.emitError("expected LLVM result type");
  StringRef name = op.getIntrinsicName();
  // Some llvm intrinsics require ElementType attribute to be attached to
  // the argument of pointer type. That prevents us from generating LLVM IR
  // because from LLVM dialect, we have LLVM IR like the below which fails
  // LLVM IR verification.
  // %3 = call i64 @llvm.aarch64.ldxr.p0(ptr %2)
  // The expected LLVM IR should be like
  // %3 = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i32) %2)
  // TODO(cir): MLIR LLVM dialect should handle this part as CIR has no way
  // to set LLVM IR attribute.
  assert(!cir::MissingFeatures::llvmIntrinsicElementTypeSupport());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm." + name, llvmResTy,
                                   adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAssumeOpLowering::matchAndRewrite(
    cir::AssumeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto cond = adaptor.getPredicate();
  rewriter.replaceOpWithNewOp<mlir::LLVM::AssumeOp>(op, cond);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAssumeAlignedOpLowering::matchAndRewrite(
    cir::AssumeAlignedOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  SmallVector<mlir::Value, 3> opBundleArgs{adaptor.getPointer()};

  auto alignment = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(), op.getAlignment());
  opBundleArgs.push_back(alignment);

  if (mlir::Value offset = adaptor.getOffset())
    opBundleArgs.push_back(offset);

  auto cond = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 1);
  rewriter.create<mlir::LLVM::AssumeOp>(op.getLoc(), cond, "align",
                                        opBundleArgs);
  rewriter.replaceAllUsesWith(op, op.getPointer());
  rewriter.eraseOp(op);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAssumeSepStorageOpLowering::matchAndRewrite(
    cir::AssumeSepStorageOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto cond = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 1);
  rewriter.replaceOpWithNewOp<mlir::LLVM::AssumeOp>(
      op, cond, "separate_storage",
      mlir::ValueRange{adaptor.getPtr1(), adaptor.getPtr2()});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBitClrsbOpLowering::matchAndRewrite(
    cir::BitClrsbOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), adaptor.getInput().getType(), 0);
  auto isNeg = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(),
      mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                         mlir::LLVM::ICmpPredicate::slt),
      adaptor.getInput(), zero);

  auto negOne = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), adaptor.getInput().getType(), -1);
  auto flipped = rewriter.create<mlir::LLVM::XOrOp>(op.getLoc(),
                                                    adaptor.getInput(), negOne);

  auto select = rewriter.create<mlir::LLVM::SelectOp>(
      op.getLoc(), isNeg, flipped, adaptor.getInput());

  auto resTy = getTypeConverter()->convertType(op.getType());
  auto clz = rewriter.create<mlir::LLVM::CountLeadingZerosOp>(
      op.getLoc(), resTy, select, false);

  auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
  auto res = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), clz, one);
  rewriter.replaceOp(op, res);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMObjSizeOpLowering::matchAndRewrite(
    cir::ObjSizeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmResTy = getTypeConverter()->convertType(op.getType());
  auto loc = op->getLoc();

  cir::SizeInfoType kindInfo = op.getKind();
  auto falseValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI1Type(), false);
  auto trueValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI1Type(), true);

  replaceOpWithCallLLVMIntrinsicOp(
      rewriter, op, "llvm.objectsize", llvmResTy,
      mlir::ValueRange{adaptor.getPtr(),
                       kindInfo == cir::SizeInfoType::Max ? falseValue
                                                          : trueValue,
                       trueValue, op.getDynamic() ? trueValue : falseValue});

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitClzOpLowering::matchAndRewrite(
    cir::BitClzOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp = rewriter.create<mlir::LLVM::CountLeadingZerosOp>(
      op.getLoc(), resTy, adaptor.getInput(), op.getIsZeroPoison());
  rewriter.replaceOp(op, llvmOp);
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitCtzOpLowering::matchAndRewrite(
    cir::BitCtzOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp = rewriter.create<mlir::LLVM::CountTrailingZerosOp>(
      op.getLoc(), resTy, adaptor.getInput(), op.getIsZeroPoison());
  rewriter.replaceOp(op, llvmOp);
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitFfsOpLowering::matchAndRewrite(
    cir::BitFfsOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto ctz = rewriter.create<mlir::LLVM::CountTrailingZerosOp>(
      op.getLoc(), resTy, adaptor.getInput(), false);

  auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
  auto ctzAddOne = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), ctz, one);

  auto zeroInputTy = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), adaptor.getInput().getType(), 0);
  auto isZero = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(),
      mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                         mlir::LLVM::ICmpPredicate::eq),
      adaptor.getInput(), zeroInputTy);

  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 0);
  auto res = rewriter.create<mlir::LLVM::SelectOp>(op.getLoc(), isZero, zero,
                                                   ctzAddOne);
  rewriter.replaceOp(op, res);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitParityOpLowering::matchAndRewrite(
    cir::BitParityOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto popcnt = rewriter.create<mlir::LLVM::CtPopOp>(op.getLoc(), resTy,
                                                     adaptor.getInput());

  auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
  auto popcntMod2 =
      rewriter.create<mlir::LLVM::AndOp>(op.getLoc(), popcnt, one);
  rewriter.replaceOp(op, popcntMod2);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitPopcountOpLowering::matchAndRewrite(
    cir::BitPopcountOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp = rewriter.create<mlir::LLVM::CtPopOp>(op.getLoc(), resTy,
                                                     adaptor.getInput());
  rewriter.replaceOp(op, llvmOp);
  return mlir::LogicalResult::success();
}

mlir::LLVM::AtomicOrdering getLLVMAtomicOrder(cir::MemOrder memo) {
  switch (memo) {
  case cir::MemOrder::Relaxed:
    return mlir::LLVM::AtomicOrdering::monotonic;
  case cir::MemOrder::Consume:
  case cir::MemOrder::Acquire:
    return mlir::LLVM::AtomicOrdering::acquire;
  case cir::MemOrder::Release:
    return mlir::LLVM::AtomicOrdering::release;
  case cir::MemOrder::AcquireRelease:
    return mlir::LLVM::AtomicOrdering::acq_rel;
  case cir::MemOrder::SequentiallyConsistent:
    return mlir::LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("shouldn't get here");
}

mlir::LogicalResult CIRToLLVMAtomicCmpXchgLowering::matchAndRewrite(
    cir::AtomicCmpXchg op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto expected = adaptor.getExpected();
  auto desired = adaptor.getDesired();

  auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
      op.getLoc(), adaptor.getPtr(), expected, desired,
      getLLVMAtomicOrder(adaptor.getSuccOrder()),
      getLLVMAtomicOrder(adaptor.getFailOrder()));
  cmpxchg.setSyncscope(getLLVMSyncScope(adaptor.getSyncscope()));
  cmpxchg.setAlignment(adaptor.getAlignment());
  cmpxchg.setWeak(adaptor.getWeak());
  cmpxchg.setVolatile_(adaptor.getIsVolatile());

  // Check result and apply stores accordingly.
  auto old = rewriter.create<mlir::LLVM::ExtractValueOp>(
      op.getLoc(), cmpxchg.getResult(), 0);
  auto cmp = rewriter.create<mlir::LLVM::ExtractValueOp>(
      op.getLoc(), cmpxchg.getResult(), 1);

  rewriter.replaceOp(op, {old, cmp});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAtomicXchgLowering::matchAndRewrite(
    cir::AtomicXchg op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // FIXME: add syncscope.
  auto llvmOrder = getLLVMAtomicOrder(adaptor.getMemOrder());
  rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
      op, mlir::LLVM::AtomicBinOp::xchg, adaptor.getPtr(), adaptor.getVal(),
      llvmOrder);
  return mlir::success();
}

mlir::Value CIRToLLVMAtomicFetchLowering::buildPostOp(
    cir::AtomicFetch op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter, mlir::Value rmwVal,
    bool isInt) const {
  SmallVector<mlir::Value> atomicOperands = {rmwVal, adaptor.getVal()};
  SmallVector<mlir::Type> atomicResTys = {rmwVal.getType()};
  return rewriter
      .create(op.getLoc(),
              rewriter.getStringAttr(getLLVMBinop(op.getBinop(), isInt)),
              atomicOperands, atomicResTys, {})
      ->getResult(0);
}

mlir::Value CIRToLLVMAtomicFetchLowering::buildMinMaxPostOp(
    cir::AtomicFetch op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter, mlir::Value rmwVal,
    bool isSigned) const {
  auto loc = op.getLoc();
  mlir::LLVM::ICmpPredicate pred;
  if (op.getBinop() == cir::AtomicFetchKind::Max) {
    pred = isSigned ? mlir::LLVM::ICmpPredicate::sgt
                    : mlir::LLVM::ICmpPredicate::ugt;
  } else { // Min
    pred = isSigned ? mlir::LLVM::ICmpPredicate::slt
                    : mlir::LLVM::ICmpPredicate::ult;
  }

  auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(), pred),
      rmwVal, adaptor.getVal());
  return rewriter.create<mlir::LLVM::SelectOp>(loc, cmp, rmwVal,
                                               adaptor.getVal());
}

llvm::StringLiteral
CIRToLLVMAtomicFetchLowering::getLLVMBinop(cir::AtomicFetchKind k,
                                           bool isInt) const {
  switch (k) {
  case cir::AtomicFetchKind::Add:
    return isInt ? mlir::LLVM::AddOp::getOperationName()
                 : mlir::LLVM::FAddOp::getOperationName();
  case cir::AtomicFetchKind::Sub:
    return isInt ? mlir::LLVM::SubOp::getOperationName()
                 : mlir::LLVM::FSubOp::getOperationName();
  case cir::AtomicFetchKind::And:
    return mlir::LLVM::AndOp::getOperationName();
  case cir::AtomicFetchKind::Xor:
    return mlir::LLVM::XOrOp::getOperationName();
  case cir::AtomicFetchKind::Or:
    return mlir::LLVM::OrOp::getOperationName();
  case cir::AtomicFetchKind::Nand:
    // There's no nand binop in LLVM, this is later fixed with a not.
    return mlir::LLVM::AndOp::getOperationName();
  case cir::AtomicFetchKind::Max:
  case cir::AtomicFetchKind::Min:
    llvm_unreachable("handled in buildMinMaxPostOp");
  }
  llvm_unreachable("Unknown atomic fetch opcode");
}

mlir::LLVM::AtomicBinOp CIRToLLVMAtomicFetchLowering::getLLVMAtomicBinOp(
    cir::AtomicFetchKind k, bool isInt, bool isSignedInt) const {
  switch (k) {
  case cir::AtomicFetchKind::Add:
    return isInt ? mlir::LLVM::AtomicBinOp::add : mlir::LLVM::AtomicBinOp::fadd;
  case cir::AtomicFetchKind::Sub:
    return isInt ? mlir::LLVM::AtomicBinOp::sub : mlir::LLVM::AtomicBinOp::fsub;
  case cir::AtomicFetchKind::And:
    return mlir::LLVM::AtomicBinOp::_and;
  case cir::AtomicFetchKind::Xor:
    return mlir::LLVM::AtomicBinOp::_xor;
  case cir::AtomicFetchKind::Or:
    return mlir::LLVM::AtomicBinOp::_or;
  case cir::AtomicFetchKind::Nand:
    return mlir::LLVM::AtomicBinOp::nand;
  case cir::AtomicFetchKind::Max: {
    if (!isInt)
      return mlir::LLVM::AtomicBinOp::fmax;
    return isSignedInt ? mlir::LLVM::AtomicBinOp::max
                       : mlir::LLVM::AtomicBinOp::umax;
  }
  case cir::AtomicFetchKind::Min: {
    if (!isInt)
      return mlir::LLVM::AtomicBinOp::fmin;
    return isSignedInt ? mlir::LLVM::AtomicBinOp::min
                       : mlir::LLVM::AtomicBinOp::umin;
  }
  }
  llvm_unreachable("Unknown atomic fetch opcode");
}

mlir::LogicalResult CIRToLLVMAtomicFetchLowering::matchAndRewrite(
    cir::AtomicFetch op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  bool isInt, isSignedInt = false; // otherwise it's float.
  if (auto intTy = mlir::dyn_cast<cir::IntType>(op.getVal().getType())) {
    isInt = true;
    isSignedInt = intTy.isSigned();
  } else if (mlir::isa<cir::SingleType, cir::DoubleType>(op.getVal().getType()))
    isInt = false;
  else {
    return op.emitError() << "Unsupported type: " << adaptor.getVal().getType();
  }

  // FIXME: add syncscope.
  auto llvmOrder = getLLVMAtomicOrder(adaptor.getMemOrder());
  auto llvmBinOpc = getLLVMAtomicBinOp(op.getBinop(), isInt, isSignedInt);
  auto rmwVal = rewriter.create<mlir::LLVM::AtomicRMWOp>(
      op.getLoc(), llvmBinOpc, adaptor.getPtr(), adaptor.getVal(), llvmOrder);

  mlir::Value result = rmwVal.getRes();
  if (!op.getFetchFirst()) {
    if (op.getBinop() == cir::AtomicFetchKind::Max ||
        op.getBinop() == cir::AtomicFetchKind::Min)
      result = buildMinMaxPostOp(op, adaptor, rewriter, rmwVal.getRes(),
                                 isSignedInt);
    else
      result = buildPostOp(op, adaptor, rewriter, rmwVal.getRes(), isInt);

    // Compensate lack of nand binop in LLVM IR.
    if (op.getBinop() == cir::AtomicFetchKind::Nand) {
      auto negOne = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), result.getType(), -1);
      result = rewriter.create<mlir::LLVM::XOrOp>(op.getLoc(), result, negOne);
    }
  }

  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAtomicFenceLowering::matchAndRewrite(
    cir::AtomicFence op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmOrder = getLLVMAtomicOrder(adaptor.getOrdering());

  auto fence = rewriter.create<mlir::LLVM::FenceOp>(op.getLoc(), llvmOrder);
  fence.setSyncscope(getLLVMSyncScope(adaptor.getSyncscope()));

  rewriter.replaceOp(op, fence);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMByteswapOpLowering::matchAndRewrite(
    cir::ByteswapOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Note that LLVM intrinsic calls to @llvm.bswap.i* have the same type as
  // the operand.

  auto resTy = mlir::cast<mlir::IntegerType>(
      getTypeConverter()->convertType(op.getType()));

  std::string llvmIntrinName = "llvm.bswap.i";
  llvmIntrinName.append(std::to_string(resTy.getWidth()));

  rewriter.replaceOpWithNewOp<mlir::LLVM::ByteSwapOp>(op, adaptor.getInput());

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMRotateOpLowering::matchAndRewrite(
    cir::RotateOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Note that LLVM intrinsic calls to @llvm.fsh{r,l}.i* have the same type as
  // the operand.
  auto src = adaptor.getSrc();
  if (op.getLeft())
    rewriter.replaceOpWithNewOp<mlir::LLVM::FshlOp>(op, src, src,
                                                    adaptor.getAmt());
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::FshrOp>(op, src, src,
                                                    adaptor.getAmt());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMSelectOpLowering::matchAndRewrite(
    cir::SelectOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto getConstantBool = [](mlir::Value value) -> std::optional<bool> {
    auto definingOp = value.getDefiningOp<cir::ConstantOp>();
    if (!definingOp)
      return std::nullopt;

    auto constValue = definingOp.getValueAttr<cir::BoolAttr>();
    if (!constValue)
      return std::nullopt;

    return constValue.getValue();
  };

  // Two special cases in the LLVMIR codegen of select op:
  // - select %0, %1, false => and %0, %1
  // - select %0, true, %1 => or %0, %1
  auto trueValue = op.getTrueValue();
  auto falseValue = op.getFalseValue();
  if (mlir::isa<cir::BoolType>(trueValue.getType())) {
    if (std::optional<bool> falseValueBool = getConstantBool(falseValue);
        falseValueBool.has_value() && !*falseValueBool) {
      // select %0, %1, false => and %0, %1
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(op, adaptor.getCondition(),
                                                     adaptor.getTrueValue());
      return mlir::success();
    }
    if (std::optional<bool> trueValueBool = getConstantBool(trueValue);
        trueValueBool.has_value() && *trueValueBool) {
      // select %0, true, %1 => or %0, %1
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, adaptor.getCondition(),
                                                    adaptor.getFalseValue());
      return mlir::success();
    }
  }

  auto llvmCondition = adaptor.getCondition();
  rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
      op, llvmCondition, adaptor.getTrueValue(), adaptor.getFalseValue());

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBrOpLowering::matchAndRewrite(
    cir::BrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, adaptor.getOperands(),
                                                op.getDest());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMGetMemberOpLowering::matchAndRewrite(
    cir::GetMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llResTy = getTypeConverter()->convertType(op.getType());
  const auto recordTy =
      mlir::cast<cir::RecordType>(op.getAddrTy().getPointee());
  assert(recordTy && "expected record type");

  switch (recordTy.getKind()) {
  case cir::RecordType::Struct:
  case cir::RecordType::Class: {
    // Since the base address is a pointer to an aggregate, the first offset
    // is always zero. The second offset tell us which member it will access.
    llvm::SmallVector<mlir::LLVM::GEPArg, 2> offset{0, op.getIndex()};
    const auto elementTy = getTypeConverter()->convertType(recordTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, llResTy, elementTy,
                                                   adaptor.getAddr(), offset);
    return mlir::success();
  }
  case cir::RecordType::Union:
    // Union members share the address space, so we just need a bitcast to
    // conform to type-checking.
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, llResTy,
                                                       adaptor.getAddr());
    return mlir::success();
  }
}

mlir::LogicalResult CIRToLLVMExtractMemberOpLowering::matchAndRewrite(
    cir::ExtractMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::int64_t indecies[1] = {static_cast<std::int64_t>(op.getIndex())};

  mlir::Type recordTy = op.getRecord().getType();
  if (auto llvmStructTy =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(recordTy)) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getRecord(), indecies);
    return mlir::success();
  }

  auto cirRecordTy = mlir::cast<cir::RecordType>(recordTy);
  switch (cirRecordTy.getKind()) {
  case cir::RecordType::Struct:
  case cir::RecordType::Class: {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getRecord(), indecies);
    return mlir::success();
  }

  case cir::RecordType::Union: {
    op.emitError("cir.extract_member cannot extract member from a union");
    return mlir::failure();
  }
  }
}

mlir::LogicalResult CIRToLLVMInsertMemberOpLowering::matchAndRewrite(
    cir::InsertMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::int64_t indecies[1] = {static_cast<std::int64_t>(op.getIndex())};
  mlir::Type recordTy = op.getRecord().getType();

  if (auto cirRecordTy = mlir::dyn_cast<cir::RecordType>(recordTy)) {
    if (cirRecordTy.getKind() == cir::RecordType::Union) {
      op.emitError("cir.update_member cannot update member of a union");
      return mlir::failure();
    }
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
      op, adaptor.getRecord(), adaptor.getValue(), indecies);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGetMethodOpLowering::matchAndRewrite(
    cir::GetMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(lowerMod && "lowering module is not available");
  mlir::Value loweredResults[2];
  lowerMod->getCXXABI().lowerGetMethod(op, loweredResults, adaptor.getMethod(),
                                       adaptor.getObject(), rewriter);
  rewriter.replaceOp(op, loweredResults);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGetRuntimeMemberOpLowering::matchAndRewrite(
    cir::GetRuntimeMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(lowerMod && "lowering module is not available");
  mlir::Type llvmResTy = getTypeConverter()->convertType(op.getType());
  mlir::Operation *llvmOp = lowerMod->getCXXABI().lowerGetRuntimeMember(
      op, llvmResTy, adaptor.getAddr(), adaptor.getMember(), rewriter);
  rewriter.replaceOp(op, llvmOp);
  return mlir::success();
}

uint64_t CIRToLLVMPtrDiffOpLowering::getTypeSize(mlir::Type type,
                                                 mlir::Operation &op) const {
  mlir::DataLayout layout(op.getParentOfType<mlir::ModuleOp>());
  // For LLVM purposes we treat void as u8.
  if (isa<cir::VoidType>(type))
    type = cir::IntType::get(type.getContext(), 8, /*isSigned=*/false);
  return llvm::divideCeil(layout.getTypeSizeInBits(type), 8);
}

mlir::LogicalResult CIRToLLVMPtrDiffOpLowering::matchAndRewrite(
    cir::PtrDiffOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto dstTy = mlir::cast<cir::IntType>(op.getType());
  auto llvmDstTy = getTypeConverter()->convertType(dstTy);

  auto lhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                     adaptor.getLhs());
  auto rhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                     adaptor.getRhs());

  auto diff =
      rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), llvmDstTy, lhs, rhs);

  cir::PointerType ptrTy = op.getLhs().getType();
  auto typeSize = getTypeSize(ptrTy.getPointee(), *op);

  // Avoid silly division by 1.
  auto resultVal = diff.getResult();
  if (typeSize != 1) {
    auto typeSizeVal = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), llvmDstTy, typeSize);

    if (dstTy.isUnsigned())
      resultVal =
          rewriter.create<mlir::LLVM::UDivOp>(op.getLoc(), diff, typeSizeVal);
    else
      resultVal =
          rewriter.create<mlir::LLVM::SDivOp>(op.getLoc(), diff, typeSizeVal);
  }
  rewriter.replaceOp(op, resultVal);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMExpectOpLowering::matchAndRewrite(
    cir::ExpectOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::optional<llvm::APFloat> prob = op.getProb();
  if (!prob)
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExpectOp>(op, adaptor.getVal(),
                                                      adaptor.getExpected());
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExpectWithProbabilityOp>(
        op, adaptor.getVal(), adaptor.getExpected(), prob.value());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVTableAddrPointOpLowering::matchAndRewrite(
    cir::VTableAddrPointOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto *converter = getTypeConverter();
  auto targetType = converter->convertType(op.getType());
  mlir::Value symAddr = op.getSymAddr();
  llvm::SmallVector<mlir::LLVM::GEPArg> offsets;
  mlir::Type eltType;
  if (!symAddr) {
    symAddr = getValueForVTableSymbol(op, rewriter, getTypeConverter(),
                                      op.getNameAttr(), eltType);
    offsets = llvm::SmallVector<mlir::LLVM::GEPArg>{
        0, op.getAddressPointAttr().getIndex(),
        op.getAddressPointAttr().getOffset()};
  } else {
    // Get indirect vtable address point retrieval
    symAddr = adaptor.getSymAddr();
    eltType = converter->convertType(symAddr.getType());
    offsets = llvm::SmallVector<mlir::LLVM::GEPArg>{
        op.getAddressPointAttr().getOffset()};
  }

  assert(eltType && "Shouldn't ever be missing an eltType here");
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, targetType, eltType,
                                                 symAddr, offsets, true);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMStackSaveOpLowering::matchAndRewrite(
    cir::StackSaveOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto ptrTy = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::StackSaveOp>(op, ptrTy);
  return mlir::success();
}

#define GET_BUILTIN_LOWERING_CLASSES_DEF
#include "clang/CIR/Dialect/IR/CIRBuiltinsLowering.inc"
#undef GET_BUILTIN_LOWERING_CLASSES_DEF

mlir::LogicalResult CIRToLLVMUnreachableOpLowering::matchAndRewrite(
    cir::UnreachableOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMTrapOpLowering::matchAndRewrite(
    cir::TrapOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  rewriter.eraseOp(op);

  rewriter.create<mlir::LLVM::Trap>(loc);

  // Note that the call to llvm.trap is not a terminator in LLVM dialect.
  // So we must emit an additional llvm.unreachable to terminate the current
  // block.
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMInlineAsmOpLowering::matchAndRewrite(
    cir::InlineAsmOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type llResTy;
  if (op.getNumResults())
    llResTy = getTypeConverter()->convertType(op.getType(0));

  auto dialect = op.getAsmFlavor();
  auto llDialect = dialect == cir::AsmFlavor::x86_att
                       ? mlir::LLVM::AsmDialect::AD_ATT
                       : mlir::LLVM::AsmDialect::AD_Intel;

  std::vector<mlir::Attribute> opAttrs;
  auto llvmAttrName = mlir::LLVM::InlineAsmOp::getElementTypeAttrName();

  // this is for the lowering to LLVM from LLVm dialect. Otherwise, if we
  // don't have the result (i.e. void type as a result of operation), the
  // element type attribute will be attached to the whole instruction, but not
  // to the operand
  if (!op.getNumResults())
    opAttrs.push_back(mlir::Attribute());

  llvm::SmallVector<mlir::Value> llvmOperands;
  llvm::SmallVector<mlir::Value> cirOperands;
  for (size_t i = 0; i < op.getAsmOperands().size(); ++i) {
    auto llvmOps = adaptor.getAsmOperands()[i];
    auto cirOps = op.getAsmOperands()[i];
    llvmOperands.insert(llvmOperands.end(), llvmOps.begin(), llvmOps.end());
    cirOperands.insert(cirOperands.end(), cirOps.begin(), cirOps.end());
  }

  // so far we infer the llvm dialect element type attr from
  // CIR operand type.
  for (std::size_t i = 0; i < op.getOperandAttrs().size(); ++i) {
    if (!op.getOperandAttrs()[i]) {
      opAttrs.push_back(mlir::Attribute());
      continue;
    }

    std::vector<mlir::NamedAttribute> attrs;
    auto typ = cast<cir::PointerType>(cirOperands[i].getType());
    auto typAttr = mlir::TypeAttr::get(convertTypeForMemory(
        *getTypeConverter(), dataLayout, typ.getPointee()));

    attrs.push_back(rewriter.getNamedAttr(llvmAttrName, typAttr));
    auto newDict = rewriter.getDictionaryAttr(attrs);
    opAttrs.push_back(newDict);
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::InlineAsmOp>(
      op, llResTy, llvmOperands, op.getAsmStringAttr(), op.getConstraintsAttr(),
      op.getSideEffectsAttr(),
      /*is_align_stack*/ mlir::UnitAttr(),
      mlir::LLVM::AsmDialectAttr::get(getContext(), llDialect),
      rewriter.getArrayAttr(opAttrs));

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMInvariantGroupOpLowering::matchAndRewrite(
    cir::InvariantGroupOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!lowerMod ||
      lowerMod->getContext().getCodeGenOpts().OptimizationLevel == 0) {
    rewriter.replaceOp(op, adaptor.getPtr());
    return mlir::success();
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::LaunderInvariantGroupOp>(
      op, adaptor.getPtr());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMPrefetchOpLowering::matchAndRewrite(
    cir::PrefetchOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::Prefetch>(
      op, adaptor.getAddr(), adaptor.getIsWrite(), adaptor.getLocality(),
      /*DataCache*/ 1);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMSetBitfieldOpLowering::matchAndRewrite(
    cir::SetBitfieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  auto info = op.getBitfieldInfo();
  auto size = info.getSize();
  auto offset = info.getOffset();
  auto storageType = info.getStorageType();
  auto context = storageType.getContext();

  unsigned storageSize = 0;

  if (auto arTy = mlir::dyn_cast<cir::ArrayType>(storageType))
    storageSize = arTy.getSize() * 8;
  else if (auto intTy = mlir::dyn_cast<cir::IntType>(storageType))
    storageSize = intTy.getWidth();
  else
    llvm_unreachable(
        "Either ArrayType or IntType expected for bitfields storage");

  auto intType = mlir::IntegerType::get(context, storageSize);
  auto srcVal = createIntCast(rewriter, adaptor.getSrc(), intType);
  auto srcWidth = storageSize;
  auto resultVal = srcVal;

  if (storageSize != size) {
    assert(storageSize > size && "Invalid bitfield size.");

    mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), intType, adaptor.getAddr(), /* alignment */ 0,
        op.getIsVolatile());

    srcVal =
        createAnd(rewriter, srcVal, llvm::APInt::getLowBitsSet(srcWidth, size));
    resultVal = srcVal;
    srcVal = createShL(rewriter, srcVal, offset);

    // Mask out the original value.
    val = createAnd(rewriter, val,
                    ~llvm::APInt::getBitsSet(srcWidth, offset, offset + size));

    // Or together the unchanged values and the source value.
    srcVal = rewriter.create<mlir::LLVM::OrOp>(op.getLoc(), val, srcVal);
  }

  rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), srcVal, adaptor.getAddr(),
                                       /* alignment */ 0, op.getIsVolatile());

  auto resultTy = getTypeConverter()->convertType(op.getType());

  if (info.getIsSigned()) {
    assert(size <= storageSize);
    unsigned highBits = storageSize - size;

    if (highBits) {
      resultVal = createShL(rewriter, resultVal, highBits);
      resultVal = createAShR(rewriter, resultVal, highBits);
    }
  }

  resultVal = createIntCast(rewriter, resultVal,
                            mlir::cast<mlir::IntegerType>(resultTy),
                            info.getIsSigned());

  rewriter.replaceOp(op, resultVal);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGetBitfieldOpLowering::matchAndRewrite(
    cir::GetBitfieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  auto info = op.getBitfieldInfo();
  auto size = info.getSize();
  auto offset = info.getOffset();
  auto storageType = info.getStorageType();
  auto context = storageType.getContext();
  unsigned storageSize = 0;

  if (auto arTy = mlir::dyn_cast<cir::ArrayType>(storageType))
    storageSize = arTy.getSize() * 8;
  else if (auto intTy = mlir::dyn_cast<cir::IntType>(storageType))
    storageSize = intTy.getWidth();
  else
    llvm_unreachable(
        "Either ArrayType or IntType expected for bitfields storage");

  auto intType = mlir::IntegerType::get(context, storageSize);

  mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
      op.getLoc(), intType, adaptor.getAddr(), 0, op.getIsVolatile());
  val = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), intType, val);

  if (info.getIsSigned()) {
    assert(static_cast<unsigned>(offset + size) <= storageSize);
    unsigned highBits = storageSize - offset - size;
    val = createShL(rewriter, val, highBits);
    val = createAShR(rewriter, val, offset + highBits);
  } else {
    val = createLShR(rewriter, val, offset);

    if (static_cast<unsigned>(offset) + size < storageSize)
      val = createAnd(rewriter, val,
                      llvm::APInt::getLowBitsSet(storageSize, size));
  }

  auto resTy = getTypeConverter()->convertType(op.getType());
  auto newOp = createIntCast(
      rewriter, val, mlir::cast<mlir::IntegerType>(resTy), info.getIsSigned());
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMDeleteArrayOpLowering::matchAndRewrite(
    cir::DeleteArrayOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  StringRef fnName = "_ZdaPv";

  auto voidTy = rewriter.getType<mlir::LLVM::LLVMVoidType>();
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

  auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {llvmPtrTy},
                                                /*isVarArg=*/false);

  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);

  // Replace the operation with a call to _ZdaPv with the pointer argument
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{}, fnName, mlir::ValueRange{adaptor.getAddress()});

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMIsConstantOpLowering::matchAndRewrite(
    cir::IsConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::IsConstantOp>(op, adaptor.getVal());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCmpThreeWayOpLowering::matchAndRewrite(
    cir::CmpThreeWayOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!op.isIntegralComparison() || !op.isStrongOrdering()) {
    op.emitError() << "unsupported three-way comparison type";
    return mlir::failure();
  }

  auto cmpInfo = op.getInfo();
  assert(cmpInfo.getLt() == -1 && cmpInfo.getEq() == 0 && cmpInfo.getGt() == 1);

  auto operandTy = mlir::cast<cir::IntType>(op.getLhs().getType());
  auto resultTy = op.getType();
  auto llvmIntrinsicName = getLLVMIntrinsicName(
      operandTy.isSigned(), operandTy.getWidth(), resultTy.getWidth());

  rewriter.setInsertionPoint(op);

  auto llvmLhs = adaptor.getLhs();
  auto llvmRhs = adaptor.getRhs();
  auto llvmResultTy = getTypeConverter()->convertType(resultTy);
  auto callIntrinsicOp =
      createCallLLVMIntrinsicOp(rewriter, op.getLoc(), llvmIntrinsicName,
                                llvmResultTy, {llvmLhs, llvmRhs});

  rewriter.replaceOp(op, callIntrinsicOp);
  return mlir::success();
}

std::string CIRToLLVMCmpThreeWayOpLowering::getLLVMIntrinsicName(
    bool signedCmp, unsigned operandWidth, unsigned resultWidth) {
  // The intrinsic's name takes the form:
  // `llvm.<scmp|ucmp>.i<resultWidth>.i<operandWidth>`

  std::string result = "llvm.";

  if (signedCmp)
    result.append("scmp.");
  else
    result.append("ucmp.");

  // Result type part.
  result.push_back('i');
  result.append(std::to_string(resultWidth));
  result.push_back('.');

  // Operand type part.
  result.push_back('i');
  result.append(std::to_string(operandWidth));

  return result;
}

mlir::LogicalResult CIRToLLVMReturnAddrOpLowering::matchAndRewrite(
    cir::ReturnAddrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.returnaddress",
                                   llvmPtrTy, adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMFrameAddrOpLowering::matchAndRewrite(
    cir::FrameAddrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.frameaddress", llvmPtrTy,
                                   adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMClearCacheOpLowering::matchAndRewrite(
    cir::ClearCacheOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto begin = adaptor.getBegin();
  auto end = adaptor.getEnd();
  auto intrinNameAttr =
      mlir::StringAttr::get(op.getContext(), "llvm.clear_cache");
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
      op, mlir::Type{}, intrinNameAttr, mlir::ValueRange{begin, end});

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMEhTypeIdOpLowering::matchAndRewrite(
    cir::EhTypeIdOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
      op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
      op.getTypeSymAttr());
  mlir::LLVM::CallIntrinsicOp newOp = createCallLLVMIntrinsicOp(
      rewriter, op.getLoc(), "llvm.eh.typeid.for.p0", rewriter.getI32Type(),
      mlir::ValueRange{addrOp});
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCatchParamOpLowering::matchAndRewrite(
    cir::CatchParamOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (op.isBegin()) {
    // Get or create `declare ptr @__cxa_begin_catch(ptr)`
    StringRef fnName = "__cxa_begin_catch";
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {llvmPtrTy},
                                                  /*isVarArg=*/false);
    getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, mlir::TypeRange{llvmPtrTy}, fnName,
        mlir::ValueRange{adaptor.getExceptionPtr()});
    return mlir::success();
  } else if (op.isEnd()) {
    StringRef fnName = "__cxa_end_catch";
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {},
        /*isVarArg=*/false);
    getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
    rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), mlir::TypeRange{}, fnName,
                                        mlir::ValueRange{});
    rewriter.eraseOp(op);
    return mlir::success();
  }
  llvm_unreachable("only begin/end supposed to make to lowering stage");
  return mlir::failure();
}

mlir::LogicalResult CIRToLLVMResumeOpLowering::matchAndRewrite(
    cir::ResumeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // %lpad.val = insertvalue { ptr, i32 } poison, ptr %exception_ptr, 0
  // %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %selector, 1
  // resume { ptr, i32 } %lpad.val2
  SmallVector<int64_t> slotIdx = {0};
  SmallVector<int64_t> selectorIdx = {1};
  auto llvmLandingPadStructTy = getLLVMLandingPadStructTy(rewriter);
  mlir::Value poison = rewriter.create<mlir::LLVM::PoisonOp>(
      op.getLoc(), llvmLandingPadStructTy);

  mlir::Value slot = rewriter.create<mlir::LLVM::InsertValueOp>(
      op.getLoc(), poison, adaptor.getExceptionPtr(), slotIdx);
  mlir::Value selector = rewriter.create<mlir::LLVM::InsertValueOp>(
      op.getLoc(), slot, adaptor.getTypeId(), selectorIdx);

  rewriter.replaceOpWithNewOp<mlir::LLVM::ResumeOp>(op, selector);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAllocExceptionOpLowering::matchAndRewrite(
    cir::AllocExceptionOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Get or create `declare ptr @__cxa_allocate_exception(i64)`
  StringRef fnName = "__cxa_allocate_exception";
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto int64Ty = mlir::IntegerType::get(rewriter.getContext(), 64);
  auto fnTy = mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {int64Ty},
                                                /*isVarArg=*/false);
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  auto size = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(),
                                                      adaptor.getSizeAttr());
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{llvmPtrTy}, fnName, mlir::ValueRange{size});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMFreeExceptionOpLowering::matchAndRewrite(
    cir::FreeExceptionOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Get or create `declare void @__cxa_free_exception(ptr)`
  StringRef fnName = "__cxa_free_exception";
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto voidTy = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
  auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {llvmPtrTy},
                                                /*isVarArg=*/false);
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{}, fnName, mlir::ValueRange{adaptor.getPtr()});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMThrowOpLowering::matchAndRewrite(
    cir::ThrowOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Get or create `declare void @__cxa_throw(ptr, ptr, ptr)`
  StringRef fnName = "__cxa_throw";
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto voidTy = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
  auto fnTy = mlir::LLVM::LLVMFunctionType::get(
      voidTy, {llvmPtrTy, llvmPtrTy, llvmPtrTy},
      /*isVarArg=*/false);
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  mlir::Value typeInfo = rewriter.create<mlir::LLVM::AddressOfOp>(
      op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
      adaptor.getTypeInfoAttr());

  mlir::Value dtor;
  if (op.getDtor()) {
    dtor = rewriter.create<mlir::LLVM::AddressOfOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        adaptor.getDtorAttr());
  } else {
    dtor = rewriter.create<mlir::LLVM::ZeroOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
  }
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{}, fnName,
      mlir::ValueRange{adaptor.getExceptionPtr(), typeInfo, dtor});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMIsFPClassOpLowering::matchAndRewrite(
    cir::IsFPClassOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto src = adaptor.getSrc();
  auto flags = adaptor.getFlags();
  auto retTy = rewriter.getI1Type();

  rewriter.replaceOpWithNewOp<mlir::LLVM::IsFPClass>(op, retTy, src, flags);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAbsOpLowering::matchAndRewrite(
    cir::AbsOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = this->getTypeConverter()->convertType(op.getType());
  auto absOp = rewriter.create<mlir::LLVM::AbsOp>(
      op.getLoc(), resTy, adaptor.getOperands()[0], adaptor.getPoison());
  rewriter.replaceOp(op, absOp);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMSignBitOpLowering::matchAndRewrite(
    cir::SignBitOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(!::cir::MissingFeatures::isPPC_FP128Ty());

  mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
  int width = layout.getTypeSizeInBits(op.getInput().getType());
  if (auto longDoubleType =
          mlir::dyn_cast<cir::LongDoubleType>(op.getInput().getType())) {
    if (mlir::isa<cir::FP80Type>(longDoubleType.getUnderlying())) {
      // If the underlying type of LongDouble is FP80Type,
      // DataLayout::getTypeSizeInBits returns 128.
      // See https://github.com/llvm/clangir/issues/1057.
      // Set the width to 80 manually.
      width = 80;
    }
  }
  auto intTy = mlir::IntegerType::get(rewriter.getContext(), width);
  auto bitcast = rewriter.create<mlir::LLVM::BitcastOp>(op->getLoc(), intTy,
                                                        adaptor.getInput());
  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), intTy, 0);
  auto cmpResult = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(), mlir::LLVM::ICmpPredicate::slt, bitcast.getResult(), zero);
  rewriter.replaceOp(op, cmpResult);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMLinkerOptionsOpLowering::matchAndRewrite(
    cir::LinkerOptionsOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto newOp = rewriter.create<mlir::LLVM::LinkerOptionsOp>(
      op.getLoc(), op.getOptionsAttr());
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

void populateCIRToLLVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &converter,
    mlir::DataLayout &dataLayout, cir::LowerModule *lowerModule,
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap) {
  patterns.add<CIRToLLVMReturnOpLowering>(patterns.getContext());
  patterns.add<CIRToLLVMAllocaOpLowering>(converter, dataLayout,
                                          stringGlobalsMap, argStringGlobalsMap,
                                          argsVarMap, patterns.getContext());
  patterns.add<
      // clang-format off
      CIRToLLVMCastOpLowering,
      CIRToLLVMLoadOpLowering,
      CIRToLLVMStoreOpLowering,
      CIRToLLVMGlobalOpLowering,
      CIRToLLVMConstantOpLowering
      // clang-format on
      >(converter, patterns.getContext(), lowerModule, dataLayout);
  patterns.add<
      // clang-format off
      CIRToLLVMBaseDataMemberOpLowering,
      CIRToLLVMBaseMethodOpLowering,
      CIRToLLVMCmpOpLowering,
      CIRToLLVMDerivedDataMemberOpLowering,
      CIRToLLVMDerivedMethodOpLowering,
      CIRToLLVMGetMethodOpLowering,
      CIRToLLVMGetRuntimeMemberOpLowering,
      CIRToLLVMInvariantGroupOpLowering
      // clang-format on
      >(converter, patterns.getContext(), lowerModule);
  patterns.add<
      // clang-format off
      CIRToLLVMPtrStrideOpLowering,
      CIRToLLVMGetElementOpLowering,
      CIRToLLVMInlineAsmOpLowering
      // clang-format on
      >(converter, patterns.getContext(), dataLayout);
  patterns.add<
      // clang-format off
      CIRToLLVMAbsOpLowering,
      CIRToLLVMAllocExceptionOpLowering,
      CIRToLLVMAssumeAlignedOpLowering,
      CIRToLLVMAssumeOpLowering,
      CIRToLLVMAssumeSepStorageOpLowering,
      CIRToLLVMAtomicCmpXchgLowering,
      CIRToLLVMAtomicFetchLowering,
      CIRToLLVMAtomicXchgLowering,
      CIRToLLVMAtomicFenceLowering,
      CIRToLLVMBaseClassAddrOpLowering,
      CIRToLLVMBinOpLowering,
      CIRToLLVMBinOpOverflowOpLowering,
      CIRToLLVMBitClrsbOpLowering,
      CIRToLLVMBitClzOpLowering,
      CIRToLLVMBitCtzOpLowering,
      CIRToLLVMBitFfsOpLowering,
      CIRToLLVMBitParityOpLowering,
      CIRToLLVMBitPopcountOpLowering,
      CIRToLLVMBrCondOpLowering,
      CIRToLLVMBrOpLowering,
      CIRToLLVMByteswapOpLowering,
      CIRToLLVMCallOpLowering,
      CIRToLLVMCatchParamOpLowering,
      CIRToLLVMClearCacheOpLowering,
      CIRToLLVMCmpThreeWayOpLowering,
      CIRToLLVMComplexCreateOpLowering,
      CIRToLLVMComplexImagOpLowering,
      CIRToLLVMComplexImagPtrOpLowering,
      CIRToLLVMComplexRealOpLowering,
      CIRToLLVMComplexRealPtrOpLowering,
      CIRToLLVMCopyOpLowering,
      CIRToLLVMDerivedClassAddrOpLowering,
      CIRToLLVMEhInflightOpLowering,
      CIRToLLVMEhTypeIdOpLowering,
      CIRToLLVMExpectOpLowering,
      CIRToLLVMExtractMemberOpLowering,
      CIRToLLVMFrameAddrOpLowering,
      CIRToLLVMFreeExceptionOpLowering,
      CIRToLLVMFuncOpLowering,
      CIRToLLVMGetBitfieldOpLowering,
      CIRToLLVMGetGlobalOpLowering,
      CIRToLLVMGetMemberOpLowering,
      CIRToLLVMInsertMemberOpLowering,
      CIRToLLVMDeleteArrayOpLowering,
      CIRToLLVMIsConstantOpLowering,
      CIRToLLVMIsFPClassOpLowering,
      CIRToLLVMLinkerOptionsOpLowering,
      CIRToLLVMLLVMIntrinsicCallOpLowering,
      CIRToLLVMMemChrOpLowering,
      CIRToLLVMMemCpyInlineOpLowering,
      CIRToLLVMMemCpyOpLowering,
      CIRToLLVMMemMoveOpLowering,
      CIRToLLVMMemSetInlineOpLowering,
      CIRToLLVMMemSetOpLowering,
      CIRToLLVMObjSizeOpLowering,
      CIRToLLVMPrefetchOpLowering,
      CIRToLLVMPtrDiffOpLowering,
      CIRToLLVMResumeOpLowering,
      CIRToLLVMReturnAddrOpLowering,
      CIRToLLVMRotateOpLowering,
      CIRToLLVMSelectOpLowering,
      CIRToLLVMSetBitfieldOpLowering,
      CIRToLLVMShiftOpLowering,
      CIRToLLVMSignBitOpLowering,
      CIRToLLVMStackSaveOpLowering,
      CIRToLLVMSwitchFlatOpLowering,
      CIRToLLVMThrowOpLowering,
      CIRToLLVMTrapOpLowering,
      CIRToLLVMTryCallOpLowering,
      CIRToLLVMUnaryOpLowering,
      CIRToLLVMUnreachableOpLowering,
      CIRToLLVMVAArgOpLowering,
      CIRToLLVMVACopyOpLowering,
      CIRToLLVMVAEndOpLowering,
      CIRToLLVMVAStartOpLowering,
      CIRToLLVMVecCmpOpLowering,
      CIRToLLVMVecCreateOpLowering,
      CIRToLLVMVecShuffleDynamicOpLowering,
      CIRToLLVMVecShuffleOpLowering,
      CIRToLLVMVecSplatOpLowering,
      CIRToLLVMVecTernaryOpLowering,
      CIRToLLVMVTableAddrPointOpLowering,
      CIRToLLVMVTTAddrPointOpLowering
#define GET_BUILTIN_LOWERING_LIST
#include "clang/CIR/Dialect/IR/CIRBuiltinsLowering.inc"

#undef GET_BUILTIN_LOWERING_LIST
      // clang-format on
      >(converter, patterns.getContext());
}

std::unique_ptr<cir::LowerModule> prepareLowerModule(mlir::ModuleOp module) {
  mlir::PatternRewriter rewriter{module->getContext()};
  // If the triple is not present, e.g. CIR modules parsed from text, we
  // cannot init LowerModule properly.
  assert(!cir::MissingFeatures::makeTripleAlwaysPresent());
  if (!module->hasAttr(cir::CIRDialect::getTripleAttrName()))
    return {};
  return cir::createLowerModule(module, rewriter);
}

static unsigned
getTargetAddrSpaceFromCIRAddrSpace(cir::AddressSpace addrSpace,
                                   cir::LowerModule *lowerModule) {
  if (addrSpace == cir::AddressSpace::Default)
    return 0; // Default address space is always 0 in LLVM.

  if (cir::isTargetAddressSpace(addrSpace))
    return cir::getTargetAddressSpaceValue(addrSpace);

  assert(lowerModule && "CIR AS map is not available");
  return lowerModule->getTargetLoweringInfo()
      .getTargetAddrSpaceFromCIRAddrSpace(addrSpace);
}

// FIXME: change the type of lowerModule to `LowerModule &` to have better
// lambda capturing experience. Also blocked by makeTripleAlwaysPresent.
void prepareTypeConverter(mlir::LLVMTypeConverter &converter,
                          mlir::DataLayout &dataLayout,
                          cir::LowerModule *lowerModule) {
  converter.addConversion([&,
                           lowerModule](cir::PointerType type) -> mlir::Type {
    unsigned addrSpace =
        getTargetAddrSpaceFromCIRAddrSpace(type.getAddrSpace(), lowerModule);
    return mlir::LLVM::LLVMPointerType::get(type.getContext(), addrSpace);
  });
  converter.addConversion(
      [&, lowerModule](cir::DataMemberType type) -> mlir::Type {
        assert(lowerModule && "CXXABI is not available");
        mlir::Type abiType =
            lowerModule->getCXXABI().lowerDataMemberType(type, converter);
        return converter.convertType(abiType);
      });
  converter.addConversion([&, lowerModule](cir::MethodType type) -> mlir::Type {
    assert(lowerModule && "CXXABI is not available");
    mlir::Type abiType =
        lowerModule->getCXXABI().lowerMethodType(type, converter);
    return converter.convertType(abiType);
  });
  converter.addConversion([&](cir::ArrayType type) -> mlir::Type {
    auto ty =
        convertTypeForMemory(converter, dataLayout, type.getElementType());
    return mlir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](cir::VectorType type) -> mlir::Type {
    auto ty = converter.convertType(type.getElementType());
    return mlir::VectorType::get(type.getSize(), ty);
  });
  converter.addConversion([&](cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 1,
                                  mlir::IntegerType::Signless);
  });
  converter.addConversion([&](cir::IntType type) -> mlir::Type {
    // LLVM doesn't work with signed types, so we drop the CIR signs here.
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::Float128Type::get(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::Float16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::BFloat16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::ComplexType type) -> mlir::Type {
    // A complex type is lowered to an LLVM struct that contains the real and
    // imaginary part as data fields.
    mlir::Type elementTy = converter.convertType(type.getElementType());
    mlir::Type structFields[2] = {elementTy, elementTy};
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                  structFields);
  });
  converter.addConversion([&](cir::FuncType type) -> mlir::Type {
    auto result = converter.convertType(type.getReturnType());
    llvm::SmallVector<mlir::Type> arguments;
    if (converter.convertTypes(type.getInputs(), arguments).failed())
      llvm_unreachable("Failed to convert function type parameters");
    auto varArg = type.isVarArg();
    return mlir::LLVM::LLVMFunctionType::get(result, arguments, varArg);
  });
  converter.addConversion([&](cir::RecordType type) -> mlir::Type {
    // FIXME(cir): create separate unions, struct, and classes types.
    // Convert struct members.
    llvm::SmallVector<mlir::Type> llvmMembers;
    switch (type.getKind()) {
    case cir::RecordType::Class:
      // TODO(cir): This should be properly validated.
    case cir::RecordType::Struct:
      for (auto ty : type.getMembers())
        llvmMembers.push_back(convertTypeForMemory(converter, dataLayout, ty));
      break;
    // Unions are lowered as only the largest member.
    case cir::RecordType::Union: {
      auto largestMember = type.getLargestMember(dataLayout);
      if (largestMember)
        llvmMembers.push_back(
            convertTypeForMemory(converter, dataLayout, largestMember));
      if (type.getPadded()) {
        auto last = *type.getMembers().rbegin();
        llvmMembers.push_back(
            convertTypeForMemory(converter, dataLayout, last));
      }
      break;
    }
    }

    // Record has a name: lower as an identified record.
    mlir::LLVM::LLVMStructType llvmStruct;
    if (type.getName()) {
      llvmStruct = mlir::LLVM::LLVMStructType::getIdentified(
          type.getContext(), type.getPrefixedName());
      if (llvmStruct.setBody(llvmMembers, /*isPacked=*/type.getPacked())
              .failed())
        llvm_unreachable("Failed to set body of record");
    } else { // Record has no name: lower as literal record.
      llvmStruct = mlir::LLVM::LLVMStructType::getLiteral(
          type.getContext(), llvmMembers, /*isPacked=*/type.getPacked());
    }

    return llvmStruct;
  });
  converter.addConversion([&](cir::VoidType type) -> mlir::Type {
    return mlir::LLVM::LLVMVoidType::get(type.getContext());
  });
}

void buildCtorDtorList(
    mlir::ModuleOp module, StringRef globalXtorName, StringRef llvmXtorName,
    llvm::function_ref<std::pair<StringRef, int>(mlir::Attribute)> createXtor) {
  llvm::SmallVector<std::pair<StringRef, int>, 2> globalXtors;
  for (auto namedAttr : module->getAttrs()) {
    if (namedAttr.getName() == globalXtorName) {
      for (auto attr : mlir::cast<mlir::ArrayAttr>(namedAttr.getValue()))
        globalXtors.emplace_back(createXtor(attr));
      break;
    }
  }

  if (globalXtors.empty())
    return;

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Create a global array llvm.global_ctors with element type of
  // struct { i32, ptr, ptr }
  auto CtorPFTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  llvm::SmallVector<mlir::Type> CtorStructFields;
  CtorStructFields.push_back(builder.getI32Type());
  CtorStructFields.push_back(CtorPFTy);
  CtorStructFields.push_back(CtorPFTy);

  auto CtorStructTy = mlir::LLVM::LLVMStructType::getLiteral(
      builder.getContext(), CtorStructFields);
  auto CtorStructArrayTy =
      mlir::LLVM::LLVMArrayType::get(CtorStructTy, globalXtors.size());

  auto loc = module.getLoc();
  auto newGlobalOp = builder.create<mlir::LLVM::GlobalOp>(
      loc, CtorStructArrayTy, true, mlir::LLVM::Linkage::Appending,
      llvmXtorName, mlir::Attribute());

  newGlobalOp.getRegion().push_back(new mlir::Block());
  builder.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());

  mlir::Value result =
      builder.create<mlir::LLVM::UndefOp>(loc, CtorStructArrayTy);

  for (uint64_t I = 0; I < globalXtors.size(); I++) {
    auto fn = globalXtors[I];
    mlir::Value structInit =
        builder.create<mlir::LLVM::UndefOp>(loc, CtorStructTy);
    mlir::Value initPriority = builder.create<mlir::LLVM::ConstantOp>(
        loc, CtorStructFields[0], fn.second);
    mlir::Value initFuncAddr = builder.create<mlir::LLVM::AddressOfOp>(
        loc, CtorStructFields[1], fn.first);
    mlir::Value initAssociate =
        builder.create<mlir::LLVM::ZeroOp>(loc, CtorStructFields[2]);
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initPriority, 0);
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initFuncAddr, 1);
    // TODO: handle associated data for initializers.
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initAssociate, 2);
    result =
        builder.create<mlir::LLVM::InsertValueOp>(loc, result, structInit, I);
  }

  builder.create<mlir::LLVM::ReturnOp>(loc, result);
}

// The unreachable code is not lowered by applyPartialConversion function
// since it traverses blocks in the dominance order. At the same time we
// do need to lower such code - otherwise verification errors occur.
// For instance, the next CIR code:
//
//    cir.func @foo(%arg0: !s32i) -> !s32i {
//      %4 = cir.cast(int_to_bool, %arg0 : !s32i), !cir.bool
//      cir.if %4 {
//        %5 = cir.const #cir.int<1> : !s32i
//        cir.return %5 : !s32i
//      } else {
//        %5 = cir.const #cir.int<0> : !s32i
//       cir.return %5 : !s32i
//      }
//     cir.return %arg0 : !s32i
//    }
//
// contains an unreachable return operation (the last one). After the flattening
// pass it will be placed into the unreachable block. And the possible error
// after the lowering pass is: error: 'cir.return' op expects parent op to be
// one of 'cir.func, cir.scope, cir.if ... The reason that this operation was
// not lowered and the new parent is llvm.func.
//
// In the future we may want to get rid of this function and use DCE pass or
// something similar. But now we need to guarantee the absence of the dialect
// verification errors.
void collect_unreachable(mlir::Operation *parent,
                         llvm::SmallVector<mlir::Operation *> &ops) {

  llvm::SmallVector<mlir::Block *> unreachable_blocks;
  parent->walk([&](mlir::Block *blk) { // check
    if (blk->hasNoPredecessors() && !blk->isEntryBlock())
      unreachable_blocks.push_back(blk);
  });

  std::set<mlir::Block *> visited;
  for (auto *root : unreachable_blocks) {
    // We create a work list for each unreachable block.
    // Thus we traverse operations in some order.
    std::deque<mlir::Block *> workList;
    workList.push_back(root);

    while (!workList.empty()) {
      auto *blk = workList.back();
      workList.pop_back();
      if (visited.count(blk))
        continue;
      visited.emplace(blk);

      for (auto &op : *blk)
        ops.push_back(&op);

      for (auto it = blk->succ_begin(); it != blk->succ_end(); ++it)
        workList.push_back(*it);
    }
  }
}

void ConvertCIRToLLVMPass::buildGlobalAnnotationsVar(
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap) {
  mlir::ModuleOp module = getOperation();
  mlir::Attribute attr =
      module->getAttr(cir::CIRDialect::getGlobalAnnotationsAttrName());
  if (!attr)
    return;
  if (auto globalAnnotValues =
          mlir::dyn_cast<cir::GlobalAnnotationValuesAttr>(attr)) {
    auto annotationValuesArray =
        mlir::dyn_cast<mlir::ArrayAttr>(globalAnnotValues.getAnnotations());
    if (!annotationValuesArray || annotationValuesArray.empty())
      return;
    mlir::OpBuilder globalVarBuilder(module.getContext());
    globalVarBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());

    // Create a global array for annotation values with element type of
    // struct { ptr, ptr, ptr, i32, ptr }
    mlir::LLVM::LLVMPointerType annoPtrTy =
        mlir::LLVM::LLVMPointerType::get(globalVarBuilder.getContext());
    llvm::SmallVector<mlir::Type> annoStructFields;
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(globalVarBuilder.getI32Type());
    annoStructFields.push_back(annoPtrTy);

    mlir::LLVM::LLVMStructType annoStructTy =
        mlir::LLVM::LLVMStructType::getLiteral(globalVarBuilder.getContext(),
                                               annoStructFields);
    mlir::LLVM::LLVMArrayType annoStructArrayTy =
        mlir::LLVM::LLVMArrayType::get(annoStructTy,
                                       annotationValuesArray.size());
    mlir::Location moduleLoc = module.getLoc();
    auto annotationGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
        moduleLoc, annoStructArrayTy, false, mlir::LLVM::Linkage::Appending,
        "llvm.global.annotations", mlir::Attribute());
    annotationGlobalOp.setSection(llvmMetadataSectionName);
    annotationGlobalOp.getRegion().push_back(new mlir::Block());
    mlir::OpBuilder varInitBuilder(module.getContext());
    varInitBuilder.setInsertionPointToEnd(
        annotationGlobalOp.getInitializerBlock());
    // Globals created for annotation strings and args to be
    // placed before the var llvm.global.annotations.
    // This is consistent with clang code gen.
    globalVarBuilder.setInsertionPoint(annotationGlobalOp);

    mlir::Value result = varInitBuilder.create<mlir::LLVM::UndefOp>(
        moduleLoc, annoStructArrayTy);

    int idx = 0;
    for (mlir::Attribute entry : annotationValuesArray) {
      auto annotValue = cast<mlir::ArrayAttr>(entry);
      mlir::Value valueEntry =
          varInitBuilder.create<mlir::LLVM::UndefOp>(moduleLoc, annoStructTy);
      SmallVector<mlir::Value, 4> vals;

      auto globalValueName = mlir::cast<mlir::StringAttr>(annotValue[0]);
      mlir::Operation *globalValue =
          mlir::SymbolTable::lookupSymbolIn(module, globalValueName);
      // The first field is ptr to the global value
      auto globalValueFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
          moduleLoc, annoPtrTy, globalValueName);
      vals.push_back(globalValueFld->getResult(0));

      cir::AnnotationAttr annot =
          mlir::cast<cir::AnnotationAttr>(annotValue[1]);
      lowerAnnotationValue(moduleLoc, globalValue->getLoc(), annot, module,
                           varInitBuilder, globalVarBuilder, stringGlobalsMap,
                           argStringGlobalsMap, argsVarMap, vals);
      for (unsigned valIdx = 0, endIdx = vals.size(); valIdx != endIdx;
           ++valIdx) {
        valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
            moduleLoc, valueEntry, vals[valIdx], valIdx);
      }
      result = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
          moduleLoc, result, valueEntry, idx++);
    }
    varInitBuilder.create<mlir::LLVM::ReturnOp>(moduleLoc, result);
  }
}

void ConvertCIRToLLVMPass::processCIRAttrs(mlir::ModuleOp module) {
  // Lower the module attributes to LLVM equivalents.
  if (auto tripleAttr = module->getAttr(cir::CIRDialect::getTripleAttrName()))
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    tripleAttr);

  // Strip the CIR attributes.
  module->removeAttr(cir::CIRDialect::getSOBAttrName());
  module->removeAttr(cir::CIRDialect::getSourceLanguageAttrName());
  module->removeAttr(cir::CIRDialect::getTripleAttrName());
}

void ConvertCIRToLLVMPass::runOnOperation() {
  llvm::TimeTraceScope scope("Convert CIR to LLVM Pass");

  auto module = getOperation();
  mlir::DataLayout dataLayout(module);
  mlir::LLVMTypeConverter converter(&getContext());
  std::unique_ptr<cir::LowerModule> lowerModule = prepareLowerModule(module);
  prepareTypeConverter(converter, dataLayout, lowerModule.get());

  mlir::RewritePatternSet patterns(&getContext());

  // Track globals created for annotation related strings
  llvm::StringMap<mlir::LLVM::GlobalOp> stringGlobalsMap;
  // Track globals created for annotation arg related strings.
  // They are different from annotation strings, as strings used in args
  // are not in llvmMetadataSectionName, and also has aligment 1.
  llvm::StringMap<mlir::LLVM::GlobalOp> argStringGlobalsMap;
  // Track globals created for annotation args.
  llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> argsVarMap;

  populateCIRToLLVMConversionPatterns(patterns, converter, dataLayout,
                                      lowerModule.get(), stringGlobalsMap,
                                      argStringGlobalsMap, argsVarMap);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(getContext());
  using namespace cir;
  // clang-format off
  target.addLegalOp<mlir::ModuleOp
                    // ,AllocaOp
                    // ,BrCondOp
                    // ,BrOp
                    // ,CallOp
                    // ,CastOp
                    // ,CmpOp
                    // ,ConstantOp
                    // ,FuncOp
                    // ,LoadOp
                    // ,ReturnOp
                    // ,StoreOp
                    // ,YieldOp
                    >();
  // clang-format on
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::BuiltinDialect, cir::CIRDialect,
                           mlir::func::FuncDialect>();

  // Allow operations that will be lowered directly to LLVM IR.
  target.addLegalOp<mlir::LLVM::ZeroOp>();

  llvm::SmallVector<mlir::Operation *> ops;
  ops.push_back(module);
  collect_unreachable(module, ops);

  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();

  // Emit the llvm.global_ctors array.
  buildCtorDtorList(module, cir::CIRDialect::getGlobalCtorsAttrName(),
                    "llvm.global_ctors", [](mlir::Attribute attr) {
                      assert(mlir::isa<cir::GlobalCtorAttr>(attr) &&
                             "must be a GlobalCtorAttr");
                      auto ctorAttr = mlir::cast<cir::GlobalCtorAttr>(attr);
                      return std::make_pair(ctorAttr.getName(),
                                            ctorAttr.getPriority());
                    });
  // Emit the llvm.global_dtors array.
  buildCtorDtorList(module, cir::CIRDialect::getGlobalDtorsAttrName(),
                    "llvm.global_dtors", [](mlir::Attribute attr) {
                      assert(mlir::isa<cir::GlobalDtorAttr>(attr) &&
                             "must be a GlobalDtorAttr");
                      auto dtorAttr = mlir::cast<cir::GlobalDtorAttr>(attr);
                      return std::make_pair(dtorAttr.getName(),
                                            dtorAttr.getPriority());
                    });
  buildGlobalAnnotationsVar(stringGlobalsMap, argStringGlobalsMap, argsVarMap);

  processCIRAttrs(module);
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

void populateCIRToLLVMPasses(mlir::OpPassManager &pm, bool useCCLowering) {
  populateCIRPreLoweringPasses(pm, useCCLowering);
  pm.addPass(createConvertCIRToLLVMPass());
}

extern void registerCIRDialectTranslation(mlir::MLIRContext &context);

mlir::ModuleOp lowerDirectlyFromCIRToLLVMDialect(mlir::ModuleOp theModule,
                                                 bool disableVerifier,
                                                 bool disableCCLowering,
                                                 bool disableDebugInfo) {
  llvm::TimeTraceScope scope("lower from CIR to LLVM Dialect");

  mlir::MLIRContext *mlirCtx = theModule.getContext();
  mlir::PassManager pm(mlirCtx);
  populateCIRToLLVMPasses(pm, !disableCCLowering);

  // This is necessary to have line tables emitted and basic
  // debugger working. In the future we will add proper debug information
  // emission directly from our frontend.
  if (!disableDebugInfo) {
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }
  // FIXME(cir): this shouldn't be necessary. It's meant to be a temporary
  // workaround until we understand why some unrealized casts are being
  // emmited and how to properly avoid them.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.enableVerifier(!disableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  return theModule;
}

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule, LLVMContext &llvmCtx,
                             bool disableVerifier, bool disableCCLowering,
                             bool disableDebugInfo) {
  llvm::TimeTraceScope scope("lower from CIR to LLVM directly");

  lowerDirectlyFromCIRToLLVMDialect(theModule, disableVerifier,
                                    disableCCLowering, disableDebugInfo);

  mlir::MLIRContext *mlirCtx = theModule.getContext();

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  mlir::registerOpenMPDialectTranslation(*mlirCtx);
  registerCIRDialectTranslation(*mlirCtx);

  llvm::TimeTraceScope __scope("translateModuleToLLVMIR");

  auto ModuleName = theModule.getName();
  auto llvmModule = mlir::translateModuleToLLVMIR(
      theModule, llvmCtx, ModuleName ? *ModuleName : "CIRToLLVMModule");

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}
} // namespace direct
} // namespace cir
