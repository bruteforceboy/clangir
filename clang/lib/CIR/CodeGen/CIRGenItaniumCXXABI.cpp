//===----- CIRGenItaniumCXXABI.cpp - Emit CIR from ASTs for a Module ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Itanium C++ ABI.  The class
// in this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCleanup.h"
#include "CIRGenFunctionInfo.h"
#include "ConstantInitBuilder.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
class CIRGenItaniumCXXABI : public CIRGenCXXABI {
  /// All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, cir::GlobalOp> VTables;

protected:
  bool UseARMMethodPtrABI;
  bool UseARMGuardVarABI;
  bool Use32BitVTableOffsetABI;

  ItaniumMangleContext &getMangleContext() {
    return cast<ItaniumMangleContext>(CIRGenCXXABI::getMangleContext());
  }

  bool isVTableHidden(const CXXRecordDecl *RD) const {
    const auto &VtableLayout =
        CGM.getItaniumVTableContext().getVTableLayout(RD);

    for (const auto &VtableComponent : VtableLayout.vtable_components()) {
      if (VtableComponent.isRTTIKind()) {
        const CXXRecordDecl *RTTIDecl = VtableComponent.getRTTIDecl();
        if (RTTIDecl->getVisibility() == Visibility::HiddenVisibility)
          return true;
      } else if (VtableComponent.isUsedFunctionPointerKind()) {
        const CXXMethodDecl *Method = VtableComponent.getFunctionDecl();
        if (Method->getVisibility() == Visibility::HiddenVisibility &&
            !Method->isDefined())
          return true;
      }
    }
    return false;
  }

  bool hasAnyUnusedVirtualInlineFunction(const CXXRecordDecl *RD) const {
    const auto &VtableLayout =
        CGM.getItaniumVTableContext().getVTableLayout(RD);

    for (const auto &VtableComponent : VtableLayout.vtable_components()) {
      // Skip empty slot.
      if (!VtableComponent.isUsedFunctionPointerKind())
        continue;

      const CXXMethodDecl *Method = VtableComponent.getFunctionDecl();
      if (!Method->getCanonicalDecl()->isInlined())
        continue;

      StringRef Name = CGM.getMangledName(VtableComponent.getGlobalDecl());
      auto *op = CGM.getGlobalValue(Name);
      if (auto globalOp = dyn_cast_or_null<cir::GlobalOp>(op))
        llvm_unreachable("NYI");

      if (auto funcOp = dyn_cast_or_null<cir::FuncOp>(op)) {
        // This checks if virtual inline function has already been emitted.
        // Note that it is possible that this inline function would be emitted
        // after trying to emit vtable speculatively. Because of this we do
        // an extra pass after emitting all deferred vtables to find and emit
        // these vtables opportunistically.
        if (!funcOp || funcOp.isDeclaration())
          return true;
      }
    }
    return false;
  }

public:
  CIRGenItaniumCXXABI(CIRGenModule &CGM, bool UseARMMethodPtrABI = false,
                      bool UseARMGuardVarABI = false)
      : CIRGenCXXABI(CGM), UseARMMethodPtrABI{UseARMMethodPtrABI},
        UseARMGuardVarABI{UseARMGuardVarABI}, Use32BitVTableOffsetABI{false} {
    assert(!UseARMMethodPtrABI && "NYI");
    assert(!UseARMGuardVarABI && "NYI");
  }
  AddedStructorArgs getImplicitConstructorArgs(CIRGenFunction &CGF,
                                               const CXXConstructorDecl *D,
                                               CXXCtorType Type,
                                               bool ForVirtualBase,
                                               bool Delegating) override;

  bool NeedsVTTParameter(GlobalDecl GD) override;

  RecordArgABI getRecordArgABI(const clang::CXXRecordDecl *RD) const override {
    // If C++ prohibits us from making a copy, pass by address.
    if (!RD->canPassInRegisters())
      return RecordArgABI::Indirect;
    else
      return RecordArgABI::Default;
  }

  bool classifyReturnType(CIRGenFunctionInfo &FI) const override;
  bool isZeroInitializable(const MemberPointerType *MPT) override;

  AddedStructorArgCounts
  buildStructorSignature(GlobalDecl GD,
                         llvm::SmallVectorImpl<CanQualType> &ArgTys) override;

  bool isThisCompleteObject(GlobalDecl GD) const override {
    // The Itanium ABI has separate complete-object vs. base-object variants of
    // both constructors and destructors.
    if (isa<CXXDestructorDecl>(GD.getDecl())) {
      switch (GD.getDtorType()) {
      case Dtor_Complete:
      case Dtor_Deleting:
        return true;

      case Dtor_Base:
        return false;

      case Dtor_Comdat:
        llvm_unreachable("emitting dtor comdat as function?");
      }
      llvm_unreachable("bad dtor kind");
    }
    if (isa<CXXConstructorDecl>(GD.getDecl())) {
      switch (GD.getCtorType()) {
      case Ctor_Complete:
        return true;

      case Ctor_Base:
        return false;

      case Ctor_CopyingClosure:
      case Ctor_DefaultClosure:
        llvm_unreachable("closure ctors in Itanium ABI?");

      case Ctor_Comdat:
        llvm_unreachable("emitting ctor comdat as function?");
      }
      llvm_unreachable("bad dtor kind");
    }

    // No other kinds.
    return false;
  }

  void emitInstanceFunctionProlog(SourceLocation Loc,
                                  CIRGenFunction &CGF) override;

  void addImplicitStructorParams(CIRGenFunction &CGF, QualType &ResTy,
                                 FunctionArgList &Params) override;

  mlir::Value getCXXDestructorImplicitParam(CIRGenFunction &CGF,
                                            const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            bool ForVirtualBase,
                                            bool Delegating) override;
  void emitCXXConstructors(const clang::CXXConstructorDecl *D) override;
  void emitCXXDestructors(const clang::CXXDestructorDecl *D) override;
  void emitCXXStructor(clang::GlobalDecl GD) override;
  void emitDestructorCall(CIRGenFunction &CGF, const CXXDestructorDecl *DD,
                          CXXDtorType Type, bool ForVirtualBase,
                          bool Delegating, Address This,
                          QualType ThisTy) override;
  void registerGlobalDtor(CIRGenFunction &CGF, const VarDecl *D,
                          cir::FuncOp dtor, mlir::Value Addr) override;
  void emitVirtualObjectDelete(CIRGenFunction &CGF, const CXXDeleteExpr *DE,
                               Address Ptr, QualType ElementType,
                               const CXXDestructorDecl *Dtor) override;
  virtual void emitRethrow(CIRGenFunction &CGF, bool isNoReturn) override;
  virtual void emitThrow(CIRGenFunction &CGF, const CXXThrowExpr *E) override;
  CatchTypeInfo
  getAddrOfCXXCatchHandlerType(mlir::Location loc, QualType Ty,
                               QualType CatchHandlerType) override {
    auto rtti = dyn_cast<cir::GlobalViewAttr>(getAddrOfRTTIDescriptor(loc, Ty));
    assert(rtti && "expected GlobalViewAttr");
    return CatchTypeInfo{rtti, 0};
  }

  void emitBeginCatch(CIRGenFunction &CGF, const CXXCatchStmt *C) override;

  bool canSpeculativelyEmitVTable(const CXXRecordDecl *RD) const override;
  cir::GlobalOp getAddrOfVTable(const CXXRecordDecl *RD,
                                CharUnits VPtrOffset) override;
  CIRGenCallee getVirtualFunctionPointer(CIRGenFunction &CGF, GlobalDecl GD,
                                         Address This, mlir::Type Ty,
                                         SourceLocation Loc) override;
  mlir::Value emitVirtualDestructorCall(CIRGenFunction &CGF,
                                        const CXXDestructorDecl *Dtor,
                                        CXXDtorType DtorType, Address This,
                                        DeleteOrMemberCallExpr E) override;
  mlir::Value getVTableAddressPoint(BaseSubobject Base,
                                    const CXXRecordDecl *VTableClass) override;
  mlir::Value getVTableAddressPointInStructorWithVTT(
      CIRGenFunction &CGF, const CXXRecordDecl *VTableClass, BaseSubobject Base,
      const CXXRecordDecl *NearestVBase);
  bool isVirtualOffsetNeededForVTableField(CIRGenFunction &CGF,
                                           CIRGenFunction::VPtr Vptr) override;
  bool canSpeculativelyEmitVTableAsBaseClass(const CXXRecordDecl *RD) const;
  mlir::Value getVTableAddressPointInStructor(
      CIRGenFunction &CGF, const CXXRecordDecl *VTableClass, BaseSubobject Base,
      const CXXRecordDecl *NearestVBase) override;
  void emitVTableDefinitions(CIRGenVTables &CGVT,
                             const CXXRecordDecl *RD) override;
  void emitVirtualInheritanceTables(const CXXRecordDecl *RD) override;
  mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc,
                                          QualType Ty) override;
  bool useThunkForDtorVariant(const CXXDestructorDecl *Dtor,
                              CXXDtorType DT) const override {
    // Itanium does not emit any destructor variant as an inline thunk.
    // Delegating may occur as an optimization, but all variants are either
    // emitted with external linkage or as linkonce if they are inline and used.
    return false;
  }

  StringRef getPureVirtualCallName() override { return "__cxa_pure_virtual"; }
  StringRef getDeletedVirtualCallName() override {
    return "__cxa_deleted_virtual";
  }

  /// TODO(cir): seems like could be shared between LLVM IR and CIR codegen.
  bool mayNeedDestruction(const VarDecl *VD) const {
    if (VD->needsDestruction(getContext()))
      return true;

    // If the variable has an incomplete class type (or array thereof), it
    // might need destruction.
    const Type *T = VD->getType()->getBaseElementTypeUnsafe();
    if (T->getAs<RecordType>() && T->isIncompleteType())
      return true;

    return false;
  }

  /// Determine whether we will definitely emit this variable with a constant
  /// initializer, either because the language semantics demand it or because
  /// we know that the initializer is a constant.
  /// For weak definitions, any initializer available in the current translation
  /// is not necessarily reflective of the initializer used; such initializers
  /// are ignored unless if InspectInitForWeakDef is true.
  /// TODO(cir): seems like could be shared between LLVM IR and CIR codegen.
  bool
  isEmittedWithConstantInitializer(const VarDecl *VD,
                                   bool InspectInitForWeakDef = false) const {
    VD = VD->getMostRecentDecl();
    if (VD->hasAttr<ConstInitAttr>())
      return true;

    // All later checks examine the initializer specified on the variable. If
    // the variable is weak, such examination would not be correct.
    if (!InspectInitForWeakDef &&
        (VD->isWeak() || VD->hasAttr<SelectAnyAttr>()))
      return false;

    const VarDecl *InitDecl = VD->getInitializingDeclaration();
    if (!InitDecl)
      return false;

    // If there's no initializer to run, this is constant initialization.
    if (!InitDecl->hasInit())
      return true;

    // If we have the only definition, we don't need a thread wrapper if we
    // will emit the value as a constant.
    if (isUniqueGVALinkage(getContext().GetGVALinkageForVariable(VD)))
      return !mayNeedDestruction(VD) && InitDecl->evaluateValue();

    // Otherwise, we need a thread wrapper unless we know that every
    // translation unit will emit the value as a constant. We rely on the
    // variable being constant-initialized in every translation unit if it's
    // constant-initialized in any translation unit, which isn't actually
    // guaranteed by the standard but is necessary for sanity.
    return InitDecl->hasConstantInitialization();
  }

  // TODO(cir): seems like could be shared between LLVM IR and CIR codegen.
  bool usesThreadWrapperFunction(const VarDecl *VD) const override {
    return !isEmittedWithConstantInitializer(VD) || mayNeedDestruction(VD);
  }

  bool doStructorsInitializeVPtrs(const CXXRecordDecl *VTableClass) override {
    return true;
  }

  size_t getSrcArgforCopyCtor(const CXXConstructorDecl *,
                              FunctionArgList &Args) const override {
    assert(!Args.empty() && "expected the arglist to not be empty!");
    return Args.size() - 1;
  }

  void emitBadCastCall(CIRGenFunction &CGF, mlir::Location loc) override;

  mlir::Value
  getVirtualBaseClassOffset(mlir::Location loc, CIRGenFunction &CGF,
                            Address This, const CXXRecordDecl *ClassDecl,
                            const CXXRecordDecl *BaseClassDecl) override;

  // The traditional clang CodeGen emits calls to `__dynamic_cast` directly into
  // LLVM in the `emitDynamicCastCall` function. In CIR, `dynamic_cast`
  // expressions are lowered to `cir.dyn_cast` ops instead of calls to runtime
  // functions. So during CIRGen we don't need the `emitDynamicCastCall`
  // function that clang CodeGen has.
  mlir::Value emitDynamicCast(CIRGenFunction &CGF, mlir::Location Loc,
                              QualType SrcRecordTy, QualType DestRecordTy,
                              cir::PointerType DestCIRTy, bool isRefCast,
                              Address Src) override;

  cir::MethodAttr buildVirtualMethodAttr(cir::MethodType MethodTy,
                                         const CXXMethodDecl *MD) override;

  Address initializeArrayCookie(CIRGenFunction &CGF, Address NewPtr,
                                mlir::Value NumElements, const CXXNewExpr *E,
                                QualType ElementType) override;

protected:
  CharUnits getArrayCookieSizeImpl(QualType ElementType) override;

  /**************************** RTTI Uniqueness ******************************/
protected:
  /// Returns true if the ABI requires RTTI type_info objects to be unique
  /// across a program.
  virtual bool shouldRTTIBeUnique() const { return true; }

public:
  /// What sort of unique-RTTI behavior should we use?
  enum RTTIUniquenessKind {
    /// We are guaranteeing, or need to guarantee, that the RTTI string
    /// is unique.
    RUK_Unique,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// can demote to hidden visibility but must use string comparisons.
    RUK_NonUniqueHidden,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// have to use string comparisons, but we also have to emit it with
    /// non-hidden visibility.
    RUK_NonUniqueVisible
  };

  /// Return the required visibility status for the given type and linkage in
  /// the current ABI.
  RTTIUniquenessKind
  classifyRTTIUniqueness(QualType CanTy, cir::GlobalLinkageKind Linkage) const;
  friend class CIRGenItaniumRTTIBuilder;
};

class CIRGenARMCXXABI : public CIRGenItaniumCXXABI {
public:
  CIRGenARMCXXABI(CIRGenModule &CGM) : CIRGenItaniumCXXABI(CGM) {
    // TODO(cir): When implemented, /*UseARMMethodPtrABI=*/true,
    //                              /*UseARMGuardVarABI=*/true) {}
    assert(!cir::MissingFeatures::appleArm64CXXABI());
  }
  CharUnits getArrayCookieSizeImpl(QualType elementType) override;
  Address initializeArrayCookie(CIRGenFunction &CGF, Address NewPtr,
                                mlir::Value NumElements, const CXXNewExpr *E,
                                QualType ElementType) override;
};

class CIRGenAppleARM64CXXABI : public CIRGenARMCXXABI {
public:
  CIRGenAppleARM64CXXABI(CIRGenModule &CGM) : CIRGenARMCXXABI(CGM) {
    Use32BitVTableOffsetABI = true;
  }

  // ARM64 libraries are prepared for non-unique RTTI.
  bool shouldRTTIBeUnique() const override { return false; }
};

} // namespace

CIRGenCXXABI::AddedStructorArgs CIRGenItaniumCXXABI::getImplicitConstructorArgs(
    CIRGenFunction &CGF, const CXXConstructorDecl *D, CXXCtorType Type,
    bool ForVirtualBase, bool Delegating) {
  if (!NeedsVTTParameter(GlobalDecl(D, Type)))
    return AddedStructorArgs{};

  // Insert the implicit 'vtt' argument as the second argument. Make sure to
  // correctly reflect its address space, which can differ from generic on
  // some targets.
  mlir::Value VTT =
      CGF.GetVTTParameter(GlobalDecl(D, Type), ForVirtualBase, Delegating);
  // LangAS AS = CGM.getGlobalConstantAddressSpace(nullptr);
  LangAS AS = LangAS::Default;
  QualType Q = getContext().getAddrSpaceQualType(getContext().VoidPtrTy, AS);
  QualType VTTTy = getContext().getPointerType(Q);
  return AddedStructorArgs::prefix({{VTT, VTTTy}});
}

/// Return whether the given global decl needs a VTT parameter, which it does if
/// it's a base constructor or destructor with virtual bases.
bool CIRGenItaniumCXXABI::NeedsVTTParameter(GlobalDecl GD) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());

  // We don't have any virtual bases, just return early.
  if (!MD->getParent()->getNumVBases())
    return false;

  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(MD) && GD.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    return true;

  return false;
}

CIRGenCXXABI *clang::CIRGen::CreateCIRGenItaniumCXXABI(CIRGenModule &CGM) {
  switch (CGM.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
    return new CIRGenItaniumCXXABI(CGM);

  case TargetCXXABI::AppleARM64:
    return new CIRGenAppleARM64CXXABI(CGM);

  default:
    llvm_unreachable("bad or NYI ABI kind");
  }
}

bool CIRGenItaniumCXXABI::classifyReturnType(CIRGenFunctionInfo &FI) const {
  auto *RD = FI.getReturnType()->getAsCXXRecordDecl();
  assert(!RD && "RecordDecl return types NYI");
  return false;
}

CIRGenCXXABI::AddedStructorArgCounts
CIRGenItaniumCXXABI::buildStructorSignature(
    GlobalDecl GD, llvm::SmallVectorImpl<CanQualType> &ArgTys) {
  clang::ASTContext &astContext = getContext();

  // All parameters are already in place except VTT, which goes after 'this'.
  // These are clang types, so we don't need to worry about sret yet.

  // Check if we need to add a VTT parameter (which has type void **).
  if ((isa<CXXConstructorDecl>(GD.getDecl()) ? GD.getCtorType() == Ctor_Base
                                             : GD.getDtorType() == Dtor_Base) &&
      cast<CXXMethodDecl>(GD.getDecl())->getParent()->getNumVBases() != 0) {
    LangAS AS = CGM.getGlobalVarAddressSpace(nullptr);
    QualType Q = astContext.getAddrSpaceQualType(astContext.VoidPtrTy, AS);
    ArgTys.insert(ArgTys.begin() + 1,
                  astContext.getPointerType(CanQualType::CreateUnsafe(Q)));
    return AddedStructorArgCounts::prefix(1);
  }

  return AddedStructorArgCounts{};
}

// Find out how to cirgen the complete destructor and constructor
namespace {
enum class StructorCIRGen { Emit, RAUW, Alias, COMDAT };
}

static StructorCIRGen getCIRGenToUse(CIRGenModule &CGM,
                                     const CXXMethodDecl *MD) {
  if (!CGM.getCodeGenOpts().CXXCtorDtorAliases)
    return StructorCIRGen::Emit;

  // The complete and base structors are not equivalent if there are any virtual
  // bases, so emit separate functions.
  if (MD->getParent()->getNumVBases())
    return StructorCIRGen::Emit;

  GlobalDecl AliasDecl;
  if (const auto *DD = dyn_cast<CXXDestructorDecl>(MD)) {
    AliasDecl = GlobalDecl(DD, Dtor_Complete);
  } else {
    const auto *CD = cast<CXXConstructorDecl>(MD);
    AliasDecl = GlobalDecl(CD, Ctor_Complete);
  }
  auto Linkage = CGM.getFunctionLinkage(AliasDecl);
  (void)Linkage;

  if (cir::isDiscardableIfUnused(Linkage))
    return StructorCIRGen::RAUW;

  // FIXME: Should we allow available_externally aliases?
  if (!cir::isValidLinkage(Linkage))
    return StructorCIRGen::RAUW;

  if (cir::isWeakForLinker(Linkage)) {
    // Only ELF and wasm support COMDATs with arbitrary names (C5/D5).
    if (CGM.getTarget().getTriple().isOSBinFormatELF() ||
        CGM.getTarget().getTriple().isOSBinFormatWasm())
      return StructorCIRGen::COMDAT;
    return StructorCIRGen::Emit;
  }

  return StructorCIRGen::Alias;
}

static void emitConstructorDestructorAlias(CIRGenModule &CGM,
                                           GlobalDecl AliasDecl,
                                           GlobalDecl TargetDecl) {
  auto Linkage = CGM.getFunctionLinkage(AliasDecl);

  // Does this function alias already exists?
  StringRef MangledName = CGM.getMangledName(AliasDecl);
  auto globalValue = dyn_cast_or_null<cir::CIRGlobalValueInterface>(
      CGM.getGlobalValue(MangledName));
  if (globalValue && !globalValue.isDeclaration()) {
    return;
  }

  auto Entry = dyn_cast_or_null<cir::FuncOp>(CGM.getGlobalValue(MangledName));

  // Retrieve aliasee info.
  auto Aliasee = dyn_cast_or_null<cir::FuncOp>(CGM.GetAddrOfGlobal(TargetDecl));
  assert(Aliasee && "expected cir.func");

  // Populate actual alias.
  CGM.emitAliasForGlobal(MangledName, Entry, AliasDecl, Aliasee, Linkage);
}

void CIRGenItaniumCXXABI::emitCXXStructor(GlobalDecl GD) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());
  auto *CD = dyn_cast<CXXConstructorDecl>(MD);
  const CXXDestructorDecl *DD = CD ? nullptr : cast<CXXDestructorDecl>(MD);

  StructorCIRGen CIRGenType = getCIRGenToUse(CGM, MD);

  if (CD ? GD.getCtorType() == Ctor_Complete
         : GD.getDtorType() == Dtor_Complete) {
    GlobalDecl BaseDecl;
    if (CD)
      BaseDecl = GD.getWithCtorType(Ctor_Base);
    else
      BaseDecl = GD.getWithDtorType(Dtor_Base);

    if (CIRGenType == StructorCIRGen::Alias ||
        CIRGenType == StructorCIRGen::COMDAT) {
      emitConstructorDestructorAlias(CGM, GD, BaseDecl);
      return;
    }

    if (CIRGenType == StructorCIRGen::RAUW) {
      StringRef MangledName = CGM.getMangledName(GD);
      auto *Aliasee = CGM.GetAddrOfGlobal(BaseDecl);
      CGM.addReplacement(MangledName, Aliasee);
      return;
    }
  }

  // The base destructor is equivalent to the base destructor of its base class
  // if there is exactly one non-virtual base class with a non-trivial
  // destructor, there are no fields with a non-trivial destructor, and the body
  // of the destructor is trivial.
  if (DD && GD.getDtorType() == Dtor_Base &&
      CIRGenType != StructorCIRGen::COMDAT &&
      !CGM.tryEmitBaseDestructorAsAlias(DD))
    return;

  // FIXME: The deleting destructor is equivalent to the selected operator
  // delete if:
  //  * either the delete is a destroying operator delete or the destructor
  //    would be trivial if it weren't virtual.
  //  * the conversion from the 'this' parameter to the first parameter of the
  //    destructor is equivalent to a bitcast,
  //  * the destructor does not have an implicit "this" return, and
  //  * the operator delete has the same calling convention and CIR function
  //    type as the destructor.
  // In such cases we should try to emit the deleting dtor as an alias to the
  // selected 'operator delete'.

  auto Fn = CGM.codegenCXXStructor(GD);

  if (CIRGenType == StructorCIRGen::COMDAT) {
    llvm_unreachable("NYI");
  } else {
    CGM.maybeSetTrivialComdat(*MD, Fn);
  }
}

void CIRGenItaniumCXXABI::addImplicitStructorParams(CIRGenFunction &CGF,
                                                    QualType &ResTY,
                                                    FunctionArgList &Params) {
  const auto *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  assert(isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD));

  // Check if we need a VTT parameter as well.
  if (NeedsVTTParameter(CGF.CurGD)) {
    ASTContext &astContext = getContext();

    // FIXME: avoid the fake decl
    LangAS AS = CGM.getGlobalVarAddressSpace(nullptr);
    QualType Q = astContext.getAddrSpaceQualType(astContext.VoidPtrTy, AS);
    QualType T = astContext.getPointerType(Q);
    auto *VTTDecl = ImplicitParamDecl::Create(
        astContext, /*DC=*/nullptr, MD->getLocation(),
        &astContext.Idents.get("vtt"), T, ImplicitParamKind::CXXVTT);
    Params.insert(Params.begin() + 1, VTTDecl);
    getStructorImplicitParamDecl(CGF) = VTTDecl;
  }
}

mlir::Value CIRGenCXXABI::loadIncomingCXXThis(CIRGenFunction &CGF) {
  return CGF.createLoad(getThisDecl(CGF), "this");
}

void CIRGenCXXABI::setCXXABIThisValue(CIRGenFunction &CGF,
                                      mlir::Value ThisPtr) {
  /// Initialize the 'this' slot.
  assert(getThisDecl(CGF) && "no 'this' variable for function");
  CGF.CXXABIThisValue = ThisPtr;
}

void CIRGenItaniumCXXABI::emitInstanceFunctionProlog(SourceLocation Loc,
                                                     CIRGenFunction &CGF) {
  // Naked functions have no prolog.
  if (CGF.CurFuncDecl && CGF.CurFuncDecl->hasAttr<NakedAttr>())
    llvm_unreachable("NYI");

  /// Initialize the 'this' slot. In the Itanium C++ ABI, no prologue
  /// adjustments are required, because they are all handled by thunks.
  setCXXABIThisValue(CGF, loadIncomingCXXThis(CGF));

  /// Initialize the 'vtt' slot if needed.
  if (getStructorImplicitParamDecl(CGF)) {
    auto Val = CGF.getBuilder().createLoad(
        CGF.getLoc(Loc),
        CGF.GetAddrOfLocalVar(getStructorImplicitParamDecl(CGF)));
    setStructorImplicitParamValue(CGF, Val);
  }

  /// If this is a function that the ABI specifies returns 'this', initialize
  /// the return slot to this' at the start of the function.
  ///
  /// Unlike the setting of return types, this is done within the ABI
  /// implementation instead of by clients of CIRGenCXXBI because:
  /// 1) getThisValue is currently protected
  /// 2) in theory, an ABI could implement 'this' returns some other way;
  ///    HasThisReturn only specifies a contract, not the implementation
  if (HasThisReturn(CGF.CurGD))
    llvm_unreachable("NYI");
}

void CIRGenItaniumCXXABI::emitCXXConstructors(const CXXConstructorDecl *D) {
  // Just make sure we're in sync with TargetCXXABI.
  assert(CGM.getTarget().getCXXABI().hasConstructorVariants());

  // The constructor used for constructing this as a base class;
  // ignores virtual bases.
  CGM.emitGlobal(GlobalDecl(D, Ctor_Base));

  // The constructor used for constructing this as a complete class;
  // constructs the virtual bases, then calls the base constructor.
  if (!D->getParent()->isAbstract()) {
    // We don't need to emit the complete ctro if the class is abstract.
    CGM.emitGlobal(GlobalDecl(D, Ctor_Complete));
  }
}

void CIRGenItaniumCXXABI::emitCXXDestructors(const CXXDestructorDecl *D) {
  // The destructor used for destructing this as a base class; ignores
  // virtual bases.
  CGM.emitGlobal(GlobalDecl(D, Dtor_Base));

  // The destructor used for destructing this as a most-derived class;
  // call the base destructor and then destructs any virtual bases.
  CGM.emitGlobal(GlobalDecl(D, Dtor_Complete));

  // The destructor in a virtual table is always a 'deleting'
  // destructor, which calls the complete destructor and then uses the
  // appropriate operator delete.
  if (D->isVirtual())
    CGM.emitGlobal(GlobalDecl(D, Dtor_Deleting));
}

namespace {
/// From traditional LLVM, useful info for LLVM lowering support:
/// A cleanup to call __cxa_end_catch.  In many cases, the caught
/// exception type lets us state definitively that the thrown exception
/// type does not have a destructor.  In particular:
///   - Catch-alls tell us nothing, so we have to conservatively
///     assume that the thrown exception might have a destructor.
///   - Catches by reference behave according to their base types.
///   - Catches of non-record types will only trigger for exceptions
///     of non-record types, which never have destructors.
///   - Catches of record types can trigger for arbitrary subclasses
///     of the caught type, so we have to assume the actual thrown
///     exception type might have a throwing destructor, even if the
///     caught type's destructor is trivial or nothrow.
struct CallEndCatch final : EHScopeStack::Cleanup {
  CallEndCatch(bool MightThrow) : MightThrow(MightThrow) {}
  bool MightThrow;

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    if (!MightThrow) {
      // Traditional LLVM codegen would emit a call to __cxa_end_catch
      // here. For CIR, just let it pass since the cleanup is going
      // to be emitted on a later pass when lowering the catch region.
      // CGF.EmitNounwindRuntimeCall(getEndCatchFn(CGF.CGM));
      CGF.getBuilder().create<cir::YieldOp>(*CGF.currSrcLoc);
      return;
    }

    // Traditional LLVM codegen would emit a call to __cxa_end_catch
    // here. For CIR, just let it pass since the cleanup is going
    // to be emitted on a later pass when lowering the catch region.
    // CGF.EmitRuntimeCallOrTryCall(getEndCatchFn(CGF.CGM));
    if (!CGF.getBuilder().getBlock()->mightHaveTerminator())
      CGF.getBuilder().create<cir::YieldOp>(*CGF.currSrcLoc);
  }
};
} // namespace

/// From traditional LLVM codegen, useful info for LLVM lowering support:
/// Emits a call to __cxa_begin_catch and enters a cleanup to call
/// __cxa_end_catch. If -fassume-nothrow-exception-dtor is specified, we assume
/// that the exception object's dtor is nothrow, therefore the __cxa_end_catch
/// call can be marked as nounwind even if EndMightThrow is true.
///
/// \param EndMightThrow - true if __cxa_end_catch might throw
static mlir::Value CallBeginCatch(CIRGenFunction &CGF, mlir::Type ParamTy,
                                  bool EndMightThrow) {
  auto catchParam = CGF.getBuilder().create<cir::CatchParamOp>(
      CGF.getBuilder().getUnknownLoc(), ParamTy, nullptr, nullptr);

  CGF.EHStack.pushCleanup<CallEndCatch>(
      NormalAndEHCleanup,
      EndMightThrow && !CGF.CGM.getLangOpts().AssumeNothrowExceptionDtor);

  return catchParam.getParam();
}

/// A "special initializer" callback for initializing a catch
/// parameter during catch initialization.
static void InitCatchParam(CIRGenFunction &CGF, const VarDecl &CatchParam,
                           Address ParamAddr, SourceLocation Loc) {
  CanQualType CatchType =
      CGF.CGM.getASTContext().getCanonicalType(CatchParam.getType());
  auto CIRCatchTy = CGF.convertTypeForMem(CatchType);

  // If we're catching by reference, we can just cast the object
  // pointer to the appropriate pointer.
  if (isa<ReferenceType>(CatchType)) {
    llvm_unreachable("NYI");
    return;
  }

  // Scalars and complexes.
  cir::TypeEvaluationKind TEK = CGF.getEvaluationKind(CatchType);
  if (TEK != cir::TEK_Aggregate) {
    // Notes for LLVM lowering:
    // If the catch type is a pointer type, __cxa_begin_catch returns
    // the pointer by value.
    if (CatchType->hasPointerRepresentation()) {
      auto catchParam = CallBeginCatch(CGF, CIRCatchTy, false);

      switch (CatchType.getQualifiers().getObjCLifetime()) {
      case Qualifiers::OCL_Strong:
        llvm_unreachable("NYI");
        // arc retain non block:
        assert(!cir::MissingFeatures::ARC());
        [[fallthrough]];

      case Qualifiers::OCL_None:
      case Qualifiers::OCL_ExplicitNone:
      case Qualifiers::OCL_Autoreleasing:
        CGF.getBuilder().createStore(CGF.getBuilder().getUnknownLoc(),
                                     catchParam, ParamAddr);
        return;

      case Qualifiers::OCL_Weak:
        llvm_unreachable("NYI");
        // arc init weak:
        assert(!cir::MissingFeatures::ARC());
        return;
      }
      llvm_unreachable("bad ownership qualifier!");
    }

    // Otherwise, it returns a pointer into the exception object.
    auto catchParam =
        CallBeginCatch(CGF, CGF.getBuilder().getPointerTo(CIRCatchTy), false);
    LValue srcLV = CGF.MakeNaturalAlignAddrLValue(catchParam, CatchType);
    LValue destLV = CGF.makeAddrLValue(ParamAddr, CatchType);
    switch (TEK) {
    case cir::TEK_Complex:
      llvm_unreachable("NYI");
      return;
    case cir::TEK_Scalar: {
      auto exnLoad = CGF.emitLoadOfScalar(srcLV, catchParam.getLoc());
      CGF.emitStoreOfScalar(exnLoad, destLV, /*init*/ true);
      return;
    }
    case cir::TEK_Aggregate:
      llvm_unreachable("evaluation kind filtered out!");
    }
    llvm_unreachable("bad evaluation kind");
  }

  // Check for a copy expression.  If we don't have a copy expression,
  // that means a trivial copy is okay.
  const Expr *copyExpr = CatchParam.getInit();
  if (!copyExpr) {
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
}

/// Begins a catch statement by initializing the catch variable and
/// calling __cxa_begin_catch.
void CIRGenItaniumCXXABI::emitBeginCatch(CIRGenFunction &CGF,
                                         const CXXCatchStmt *S) {
  // Notes for LLVM lowering:
  // We have to be very careful with the ordering of cleanups here:
  //   C++ [except.throw]p4:
  //     The destruction [of the exception temporary] occurs
  //     immediately after the destruction of the object declared in
  //     the exception-declaration in the handler.
  //
  // So the precise ordering is:
  //   1.  Construct catch variable.
  //   2.  __cxa_begin_catch
  //   3.  Enter __cxa_end_catch cleanup
  //   4.  Enter dtor cleanup
  //
  // We do this by using a slightly abnormal initialization process.
  // Delegation sequence:
  //   - ExitCXXTryStmt opens a RunCleanupsScope
  //     - EmitAutoVarAlloca creates the variable and debug info
  //       - InitCatchParam initializes the variable from the exception
  //       - CallBeginCatch calls __cxa_begin_catch
  //       - CallBeginCatch enters the __cxa_end_catch cleanup
  //     - EmitAutoVarCleanups enters the variable destructor cleanup
  //   - EmitCXXTryStmt emits the code for the catch body
  //   - EmitCXXTryStmt close the RunCleanupsScope

  VarDecl *CatchParam = S->getExceptionDecl();
  if (!CatchParam) {
    CallBeginCatch(CGF, CGF.getBuilder().getVoidPtrTy(), true);
    return;
  }

  auto getCatchParamAllocaIP = [&]() {
    auto currIns = CGF.getBuilder().saveInsertionPoint();
    auto currParent = currIns.getBlock()->getParentOp();

    mlir::Block *insertBlock = nullptr;
    if (auto scopeOp = currParent->getParentOfType<cir::ScopeOp>()) {
      insertBlock = &scopeOp.getScopeRegion().getBlocks().back();
    } else if (auto fnOp = currParent->getParentOfType<cir::FuncOp>()) {
      insertBlock = &fnOp.getRegion().getBlocks().back();
    } else {
      llvm_unreachable("unknown outermost scope-like parent");
    }

    return CGF.getBuilder().getBestAllocaInsertPoint(insertBlock);
  };

  // Emit the local. Make sure the alloca's superseed the current scope, since
  // these are going to be consumed by `cir.catch`, which is not within the
  // current scope.
  auto var = CGF.emitAutoVarAlloca(*CatchParam, getCatchParamAllocaIP());
  InitCatchParam(CGF, *CatchParam, var.getObjectAddress(CGF), S->getBeginLoc());
  // FIXME(cir): double check cleanups here are happening in the right blocks.
  CGF.emitAutoVarCleanups(var);
}

cir::GlobalOp CIRGenItaniumCXXABI::getAddrOfVTable(const CXXRecordDecl *RD,
                                                   CharUnits VPtrOffset) {
  assert(VPtrOffset.isZero() && "Itanium ABI only supports zero vptr offsets");
  cir::GlobalOp &vtable = VTables[RD];
  if (vtable)
    return vtable;

  // Queue up this vtable for possible deferred emission.
  CGM.addDeferredVTable(RD);

  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  getMangleContext().mangleCXXVTable(RD, Out);

  const VTableLayout &VTLayout =
      CGM.getItaniumVTableContext().getVTableLayout(RD);
  auto VTableType = CGM.getVTables().getVTableType(VTLayout);

  // Use pointer alignment for the vtable. Otherwise we would align them based
  // on the size of the initializer which doesn't make sense as only single
  // values are read.
  unsigned PAlign = CGM.getItaniumVTableContext().isRelativeLayout()
                        ? 32
                        : CGM.getTarget().getPointerAlign(LangAS::Default);

  vtable = CGM.createOrReplaceCXXRuntimeVariable(
      CGM.getLoc(RD->getSourceRange()), Name, VTableType,
      cir::GlobalLinkageKind::ExternalLinkage,
      getContext().toCharUnitsFromBits(PAlign));
  // LLVM codegen handles unnamedAddr
  assert(!cir::MissingFeatures::unnamedAddr());

  // In MS C++ if you have a class with virtual functions in which you are using
  // selective member import/export, then all virtual functions must be exported
  // unless they are inline, otherwise a link error will result. To match this
  // behavior, for such classes, we dllimport the vtable if it is defined
  // externally and all the non-inline virtual methods are marked dllimport, and
  // we dllexport the vtable if it is defined in this TU and all the non-inline
  // virtual methods are marked dllexport.
  if (CGM.getTarget().hasPS4DLLImportExport())
    llvm_unreachable("NYI");

  CGM.setGVProperties(vtable, RD);
  return vtable;
}

CIRGenCallee CIRGenItaniumCXXABI::getVirtualFunctionPointer(
    CIRGenFunction &CGF, GlobalDecl GD, Address This, mlir::Type Ty,
    SourceLocation Loc) {
  auto loc = CGF.getLoc(Loc);
  auto TyPtr = CGF.getBuilder().getPointerTo(Ty);
  auto *MethodDecl = cast<CXXMethodDecl>(GD.getDecl());
  auto VTable = CGF.getVTablePtr(
      loc, This, CGF.getBuilder().getPointerTo(TyPtr), MethodDecl->getParent());

  uint64_t VTableIndex = CGM.getItaniumVTableContext().getMethodVTableIndex(GD);
  mlir::Value VFunc{};
  if (CGF.shouldEmitVTableTypeCheckedLoad(MethodDecl->getParent())) {
    llvm_unreachable("NYI");
  } else {
    CGF.emitTypeMetadataCodeForVCall(MethodDecl->getParent(), VTable, Loc);

    mlir::Value VFuncLoad;
    if (CGM.getItaniumVTableContext().isRelativeLayout()) {
      llvm_unreachable("NYI");
    } else {
      VTable = CGF.getBuilder().createBitcast(
          loc, VTable, CGF.getBuilder().getPointerTo(TyPtr));
      auto VTableSlotPtr = CGF.getBuilder().create<cir::VTableAddrPointOp>(
          loc, CGF.getBuilder().getPointerTo(TyPtr),
          ::mlir::FlatSymbolRefAttr{}, VTable,
          cir::AddressPointAttr::get(CGF.getBuilder().getContext(), 0,
                                     VTableIndex));
      VFuncLoad = CGF.getBuilder().createAlignedLoad(loc, TyPtr, VTableSlotPtr,
                                                     CGF.getPointerAlign());
    }

    // Add !invariant.load md to virtual function load to indicate that
    // function didn't change inside vtable.
    // It's safe to add it without -fstrict-vtable-pointers, but it would not
    // help in devirtualization because it will only matter if we will have 2
    // the same virtual function loads from the same vtable load, which won't
    // happen without enabled devirtualization with -fstrict-vtable-pointers.
    if (CGM.getCodeGenOpts().OptimizationLevel > 0 &&
        CGM.getCodeGenOpts().StrictVTablePointers) {
      llvm_unreachable("NYI");
    }
    VFunc = VFuncLoad;
  }

  CIRGenCallee Callee(GD, VFunc.getDefiningOp());
  return Callee;
}

mlir::Value CIRGenItaniumCXXABI::getVTableAddressPointInStructorWithVTT(
    CIRGenFunction &CGF, const CXXRecordDecl *VTableClass, BaseSubobject Base,
    const CXXRecordDecl *NearestVBase) {
  assert((Base.getBase()->getNumVBases() || NearestVBase != nullptr) &&
         NeedsVTTParameter(CGF.CurGD) && "This class doesn't have VTT");

  // Get the secondary vpointer index.
  uint64_t VirtualPointerIndex =
      CGM.getVTables().getSecondaryVirtualPointerIndex(VTableClass, Base);

  /// Load the VTT.
  auto VTTPtr = CGF.LoadCXXVTT();
  auto Loc = CGF.getLoc(VTableClass->getSourceRange());
  // Calculate the address point from the VTT, and the offset may be zero.
  VTTPtr = CGF.getBuilder().createVTTAddrPoint(Loc, VTTPtr.getType(), VTTPtr,
                                               VirtualPointerIndex);
  // And load the address point from the VTT.
  return CGF.getBuilder().createAlignedLoad(Loc, CGF.VoidPtrTy, VTTPtr,
                                            CGF.getPointerAlign());
}

mlir::Value
CIRGenItaniumCXXABI::getVTableAddressPoint(BaseSubobject Base,
                                           const CXXRecordDecl *VTableClass) {
  auto vtable = getAddrOfVTable(VTableClass, CharUnits());

  // Find the appropriate vtable within the vtable group, and the address point
  // within that vtable.
  VTableLayout::AddressPointLocation AddressPoint =
      CGM.getItaniumVTableContext()
          .getVTableLayout(VTableClass)
          .getAddressPoint(Base);

  auto &builder = CGM.getBuilder();
  auto vtablePtrTy = builder.getVirtualFnPtrType(/*isVarArg=*/false);

  return builder.create<cir::VTableAddrPointOp>(
      CGM.getLoc(VTableClass->getSourceRange()), vtablePtrTy,
      mlir::FlatSymbolRefAttr::get(vtable.getSymNameAttr()), mlir::Value{},
      cir::AddressPointAttr::get(CGM.getBuilder().getContext(),
                                 AddressPoint.VTableIndex,
                                 AddressPoint.AddressPointIndex));
}

mlir::Value CIRGenItaniumCXXABI::getVTableAddressPointInStructor(
    CIRGenFunction &CGF, const CXXRecordDecl *VTableClass, BaseSubobject Base,
    const CXXRecordDecl *NearestVBase) {

  if ((Base.getBase()->getNumVBases() || NearestVBase != nullptr) &&
      NeedsVTTParameter(CGF.CurGD)) {
    return getVTableAddressPointInStructorWithVTT(CGF, VTableClass, Base,
                                                  NearestVBase);
  }
  return getVTableAddressPoint(Base, VTableClass);
}

bool CIRGenItaniumCXXABI::isVirtualOffsetNeededForVTableField(
    CIRGenFunction &CGF, CIRGenFunction::VPtr Vptr) {
  if (Vptr.NearestVBase == nullptr)
    return false;
  return NeedsVTTParameter(CGF.CurGD);
}

bool CIRGenItaniumCXXABI::canSpeculativelyEmitVTableAsBaseClass(
    const CXXRecordDecl *RD) const {
  // We don't emit available_externally vtables if we are in -fapple-kext mode
  // because kext mode does not permit devirtualization.
  if (CGM.getLangOpts().AppleKext)
    return false;

  // If the vtable is hidden then it is not safe to emit an available_externally
  // copy of vtable.
  if (isVTableHidden(RD))
    return false;

  if (CGM.getCodeGenOpts().ForceEmitVTables)
    return true;

  // If we don't have any not emitted inline virtual function then we are safe
  // to emit an available_externally copy of vtable.
  // FIXME we can still emit a copy of the vtable if we
  // can emit definition of the inline functions.
  if (hasAnyUnusedVirtualInlineFunction(RD))
    return false;

  // For a class with virtual bases, we must also be able to speculatively
  // emit the VTT, because CodeGen doesn't have separate notions of "can emit
  // the vtable" and "can emit the VTT". For a base subobject, this means we
  // need to be able to emit non-virtual base vtables.
  if (RD->getNumVBases()) {
    for (const auto &B : RD->bases()) {
      auto *BRD = B.getType()->getAsCXXRecordDecl();
      assert(BRD && "no class for base specifier");
      if (B.isVirtual() || !BRD->isDynamicClass())
        continue;
      if (!canSpeculativelyEmitVTableAsBaseClass(BRD))
        return false;
    }
  }

  return true;
}

bool CIRGenItaniumCXXABI::canSpeculativelyEmitVTable(
    const CXXRecordDecl *RD) const {
  if (!canSpeculativelyEmitVTableAsBaseClass(RD))
    return false;

  // For a complete-object vtable (or more specifically, for the VTT), we need
  // to be able to speculatively emit the vtables of all dynamic virtual bases.
  for (const auto &B : RD->vbases()) {
    auto *BRD = B.getType()->getAsCXXRecordDecl();
    assert(BRD && "no class for base specifier");
    if (!BRD->isDynamicClass())
      continue;
    if (!canSpeculativelyEmitVTableAsBaseClass(BRD))
      return false;
  }

  return true;
}

namespace {
class CIRGenItaniumRTTIBuilder {
  CIRGenModule &CGM;                 // Per-module state.
  const CIRGenItaniumCXXABI &CXXABI; // Per-module state.

  /// The fields of the RTTI descriptor currently being built.
  SmallVector<mlir::Attribute, 16> Fields;

  // Returns the mangled type name of the given type.
  cir::GlobalOp GetAddrOfTypeName(mlir::Location loc, QualType Ty,
                                  cir::GlobalLinkageKind Linkage);

  // /// Returns the constant for the RTTI
  // /// descriptor of the given type.
  mlir::Attribute GetAddrOfExternalRTTIDescriptor(mlir::Location loc,
                                                  QualType Ty);

  /// Build the vtable pointer for the given type.
  void BuildVTablePointer(mlir::Location loc, const Type *Ty);

  /// Build an abi::__si_class_type_info, used for single inheritance, according
  /// to the Itanium C++ ABI, 2.9.5p6b.
  void BuildSIClassTypeInfo(mlir::Location loc, const CXXRecordDecl *RD);

  /// Build an abi::__vmi_class_type_info, used for
  /// classes with bases that do not satisfy the abi::__si_class_type_info
  /// constraints, according ti the Itanium C++ ABI, 2.9.5p5c.
  void BuildVMIClassTypeInfo(mlir::Location loc, const CXXRecordDecl *RD);

  // /// Build an abi::__pointer_type_info struct, used
  // /// for pointer types.
  // void BuildPointerTypeInfo(QualType PointeeTy);

  // /// Build the appropriate kind of
  // /// type_info for an object type.
  // void BuildObjCObjectTypeInfo(const ObjCObjectType *Ty);

  // /// Build an
  // abi::__pointer_to_member_type_info
  // /// struct, used for member pointer types.
  // void BuildPointerToMemberTypeInfo(const MemberPointerType *Ty);

public:
  CIRGenItaniumRTTIBuilder(const CIRGenItaniumCXXABI &ABI, CIRGenModule &_CGM)
      : CGM(_CGM), CXXABI(ABI) {}

  // Pointer type info flags.
  enum {
    /// PTI_Const - Type has const qualifier.
    PTI_Const = 0x1,

    /// PTI_Volatile - Type has volatile qualifier.
    PTI_Volatile = 0x2,

    /// PTI_Restrict - Type has restrict qualifier.
    PTI_Restrict = 0x4,

    /// PTI_Incomplete - Type is incomplete.
    PTI_Incomplete = 0x8,

    /// PTI_ContainingClassIncomplete - Containing class is incomplete.
    /// (in pointer to member).
    PTI_ContainingClassIncomplete = 0x10,

    /// PTI_TransactionSafe - Pointee is transaction_safe function (C++ TM TS).
    // PTI_TransactionSafe = 0x20,

    /// PTI_Noexcept - Pointee is noexcept function (C++1z).
    PTI_Noexcept = 0x40,
  };

  // VMI type info flags.
  enum {
    /// VMI_NonDiamondRepeat - Class has non-diamond repeated inheritance.
    VMI_NonDiamondRepeat = 0x1,

    /// VMI_DiamondShaped - Class is diamond shaped.
    VMI_DiamondShaped = 0x2
  };

  // Base class type info flags.
  enum {
    /// BCTI_Virtual - Base class is virtual.
    BCTI_Virtual = 0x1,

    /// BCTI_Public - Base class is public.
    BCTI_Public = 0x2
  };

  /// Build the RTTI type info struct for the given type, or
  /// link to an existing RTTI descriptor if one already exists.
  mlir::Attribute BuildTypeInfo(mlir::Location loc, QualType Ty);

  /// Build the RTTI type info struct for the given type.
  mlir::Attribute BuildTypeInfo(mlir::Location loc, QualType Ty,
                                cir::GlobalLinkageKind Linkage,
                                mlir::SymbolTable::Visibility Visibility);
};
} // namespace

/// Given a builtin type, returns whether the type
/// info for that type is defined in the standard library.
/// TODO(cir): this can unified with LLVM codegen
static bool TypeInfoIsInStandardLibrary(const BuiltinType *Ty) {
  // Itanium C++ ABI 2.9.2:
  //   Basic type information (e.g. for "int", "bool", etc.) will be kept in
  //   the run-time support library. Specifically, the run-time support
  //   library should contain type_info objects for the types X, X* and
  //   X const*, for every X in: void, std::nullptr_t, bool, wchar_t, char,
  //   unsigned char, signed char, short, unsigned short, int, unsigned int,
  //   long, unsigned long, long long, unsigned long long, float, double,
  //   long double, char16_t, char32_t, and the IEEE 754r decimal and
  //   half-precision floating point types.
  //
  // GCC also emits RTTI for __int128.
  // FIXME: We do not emit RTTI information for decimal types here.

  // Types added here must also be added to EmitFundamentalRTTIDescriptors.
  switch (Ty->getKind()) {
  case BuiltinType::WasmExternRef:
  case BuiltinType::HLSLResource:
    llvm_unreachable("NYI");
  case BuiltinType::Void:
  case BuiltinType::NullPtr:
  case BuiltinType::Bool:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char_U:
  case BuiltinType::Char_S:
  case BuiltinType::UChar:
  case BuiltinType::SChar:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
  case BuiltinType::Half:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::Float16:
  case BuiltinType::Float128:
  case BuiltinType::Ibm128:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return true;

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
  case BuiltinType::OCLSampler:
  case BuiltinType::OCLEvent:
  case BuiltinType::OCLClkEvent:
  case BuiltinType::OCLQueue:
  case BuiltinType::OCLReserveID:
#define SVE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/AArch64SVEACLETypes.def"
#define PPC_VECTOR_TYPE(Name, Id, Size) case BuiltinType::Id:
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
#define AMDGPU_TYPE(Name, Id, SingletonId, Width, Align) case BuiltinType::Id:
#include "clang/Basic/AMDGPUTypes.def"
  case BuiltinType::ShortAccum:
  case BuiltinType::Accum:
  case BuiltinType::LongAccum:
  case BuiltinType::UShortAccum:
  case BuiltinType::UAccum:
  case BuiltinType::ULongAccum:
  case BuiltinType::ShortFract:
  case BuiltinType::Fract:
  case BuiltinType::LongFract:
  case BuiltinType::UShortFract:
  case BuiltinType::UFract:
  case BuiltinType::ULongFract:
  case BuiltinType::SatShortAccum:
  case BuiltinType::SatAccum:
  case BuiltinType::SatLongAccum:
  case BuiltinType::SatUShortAccum:
  case BuiltinType::SatUAccum:
  case BuiltinType::SatULongAccum:
  case BuiltinType::SatShortFract:
  case BuiltinType::SatFract:
  case BuiltinType::SatLongFract:
  case BuiltinType::SatUShortFract:
  case BuiltinType::SatUFract:
  case BuiltinType::SatULongFract:
  case BuiltinType::BFloat16:
    return false;

  case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
    llvm_unreachable("asking for RRTI for a placeholder type!");

  case BuiltinType::ObjCId:
  case BuiltinType::ObjCClass:
  case BuiltinType::ObjCSel:
    llvm_unreachable("FIXME: Objective-C types are unsupported!");
  }

  llvm_unreachable("Invalid BuiltinType Kind!");
}

static bool TypeInfoIsInStandardLibrary(const PointerType *PointerTy) {
  QualType PointeeTy = PointerTy->getPointeeType();
  const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(PointeeTy);
  if (!BuiltinTy)
    return false;

  // Check the qualifiers.
  Qualifiers Quals = PointeeTy.getQualifiers();
  Quals.removeConst();

  if (!Quals.empty())
    return false;

  return TypeInfoIsInStandardLibrary(BuiltinTy);
}

/// Returns whether the type
/// information for the given type exists in the standard library.
/// TODO(cir): this can unified with LLVM codegen
static bool IsStandardLibraryRTTIDescriptor(QualType Ty) {
  // Type info for builtin types is defined in the standard library.
  if (const BuiltinType *BuiltinTy = dyn_cast<BuiltinType>(Ty))
    return TypeInfoIsInStandardLibrary(BuiltinTy);

  // Type info for some pointer types to builtin types is defined in the
  // standard library.
  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return TypeInfoIsInStandardLibrary(PointerTy);

  return false;
}

/// Returns whether the type information for
/// the given type exists somewhere else, and that we should not emit the type
/// information in this translation unit.  Assumes that it is not a
/// standard-library type.
/// TODO(cir): this can unified with LLVM codegen
static bool ShouldUseExternalRTTIDescriptor(CIRGenModule &CGM, QualType Ty) {
  ASTContext &astContext = CGM.getASTContext();

  // If RTTI is disabled, assume it might be disabled in the
  // translation unit that defines any potential key function, too.
  if (!astContext.getLangOpts().RTTI)
    return false;

  if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RecordTy->getDecl());
    if (!RD->hasDefinition())
      return false;

    if (!RD->isDynamicClass())
      return false;

    // FIXME: this may need to be reconsidered if the key function
    // changes.
    // N.B. We must always emit the RTTI data ourselves if there exists a key
    // function.
    bool IsDLLImport = RD->hasAttr<DLLImportAttr>();

    // Don't import the RTTI but emit it locally.
    if (CGM.getTriple().isWindowsGNUEnvironment())
      return false;

    if (CGM.getVTables().isVTableExternal(RD)) {
      if (CGM.getTarget().hasPS4DLLImportExport())
        return true;

      return IsDLLImport && !CGM.getTriple().isWindowsItaniumEnvironment()
                 ? false
                 : true;
    }
    if (IsDLLImport)
      return true;
  }

  return false;
}

/// Returns whether the given record type is incomplete.
/// TODO(cir): this can unified with LLVM codegen
static bool IsIncompleteClassType(const RecordType *RecordTy) {
  return !RecordTy->getDecl()->isCompleteDefinition();
}

/// Returns whether the given type contains an
/// incomplete class type. This is true if
///
///   * The given type is an incomplete class type.
///   * The given type is a pointer type whose pointee type contains an
///     incomplete class type.
///   * The given type is a member pointer type whose class is an incomplete
///     class type.
///   * The given type is a member pointer type whoise pointee type contains an
///     incomplete class type.
/// is an indirect or direct pointer to an incomplete class type.
/// TODO(cir): this can unified with LLVM codegen
static bool ContainsIncompleteClassType(QualType Ty) {
  if (const RecordType *RecordTy = dyn_cast<RecordType>(Ty)) {
    if (IsIncompleteClassType(RecordTy))
      return true;
  }

  if (const PointerType *PointerTy = dyn_cast<PointerType>(Ty))
    return ContainsIncompleteClassType(PointerTy->getPointeeType());

  if (const MemberPointerType *MemberPointerTy =
          dyn_cast<MemberPointerType>(Ty)) {
    // Check if the class type is incomplete.
    const RecordType *ClassType = cast<RecordType>(MemberPointerTy->getClass());
    if (IsIncompleteClassType(ClassType))
      return true;

    return ContainsIncompleteClassType(MemberPointerTy->getPointeeType());
  }

  return false;
}

// Return whether the given record decl has a "single,
// public, non-virtual base at offset zero (i.e. the derived class is dynamic
// iff the base is)", according to Itanium C++ ABI, 2.95p6b.
// TODO(cir): this can unified with LLVM codegen
static bool CanUseSingleInheritance(const CXXRecordDecl *RD) {
  // Check the number of bases.
  if (RD->getNumBases() != 1)
    return false;

  // Get the base.
  CXXRecordDecl::base_class_const_iterator Base = RD->bases_begin();

  // Check that the base is not virtual.
  if (Base->isVirtual())
    return false;

  // Check that the base is public.
  if (Base->getAccessSpecifier() != AS_public)
    return false;

  // Check that the class is dynamic iff the base is.
  auto *BaseDecl =
      cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
  if (!BaseDecl->isEmpty() &&
      BaseDecl->isDynamicClass() != RD->isDynamicClass())
    return false;

  return true;
}

/// Return the linkage that the type info and type info name constants
/// should have for the given type.
static cir::GlobalLinkageKind getTypeInfoLinkage(CIRGenModule &CGM,
                                                 QualType Ty) {
  // Itanium C++ ABI 2.9.5p7:
  //   In addition, it and all of the intermediate abi::__pointer_type_info
  //   structs in the chain down to the abi::__class_type_info for the
  //   incomplete class type must be prevented from resolving to the
  //   corresponding type_info structs for the complete class type, possibly
  //   by making them local static objects. Finally, a dummy class RTTI is
  //   generated for the incomplete type that will not resolve to the final
  //   complete class RTTI (because the latter need not exist), possibly by
  //   making it a local static object.
  if (ContainsIncompleteClassType(Ty))
    return cir::GlobalLinkageKind::InternalLinkage;

  switch (Ty->getLinkage()) {
  case Linkage::None:
  case Linkage::Internal:
  case Linkage::UniqueExternal:
    return cir::GlobalLinkageKind::InternalLinkage;

  case Linkage::VisibleNone:
  case Linkage::Module:
  case Linkage::External:
    // RTTI is not enabled, which means that this type info struct is going
    // to be used for exception handling. Give it linkonce_odr linkage.
    if (!CGM.getLangOpts().RTTI)
      return cir::GlobalLinkageKind::LinkOnceODRLinkage;

    if (const RecordType *Record = dyn_cast<RecordType>(Ty)) {
      const CXXRecordDecl *RD = cast<CXXRecordDecl>(Record->getDecl());
      if (RD->hasAttr<WeakAttr>())
        return cir::GlobalLinkageKind::WeakODRLinkage;
      if (CGM.getTriple().isWindowsItaniumEnvironment())
        if (RD->hasAttr<DLLImportAttr>() &&
            ShouldUseExternalRTTIDescriptor(CGM, Ty))
          return cir::GlobalLinkageKind::ExternalLinkage;
      // MinGW always uses LinkOnceODRLinkage for type info.
      if (RD->isDynamicClass() && !CGM.getASTContext()
                                       .getTargetInfo()
                                       .getTriple()
                                       .isWindowsGNUEnvironment())
        return CGM.getVTableLinkage(RD);
    }

    return cir::GlobalLinkageKind::LinkOnceODRLinkage;
  case Linkage::Invalid:
    llvm_unreachable("Invalid linkage!");
  }

  llvm_unreachable("Invalid linkage!");
}

mlir::Attribute CIRGenItaniumRTTIBuilder::BuildTypeInfo(mlir::Location loc,
                                                        QualType Ty) {
  // We want to operate on the canonical type.
  Ty = Ty.getCanonicalType();

  // Check if we've already emitted an RTTI descriptor for this type.
  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTI(Ty, Out);

  auto OldGV = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(CGM.getModule(), Name));

  if (OldGV && !OldGV.isDeclaration()) {
    assert(!OldGV.hasAvailableExternallyLinkage() &&
           "available_externally typeinfos not yet implemented");
    return CGM.getBuilder().getGlobalViewAttr(CGM.getBuilder().getUInt8PtrTy(),
                                              OldGV);
  }

  // Check if there is already an external RTTI descriptor for this type.
  if (IsStandardLibraryRTTIDescriptor(Ty) ||
      ShouldUseExternalRTTIDescriptor(CGM, Ty))
    return GetAddrOfExternalRTTIDescriptor(loc, Ty);

  // Emit the standard library with external linkage.
  auto Linkage = getTypeInfoLinkage(CGM, Ty);

  // Give the type_info object and name the formal visibility of the
  // type itself.
  assert(!cir::MissingFeatures::hiddenVisibility());
  assert(!cir::MissingFeatures::protectedVisibility());
  mlir::SymbolTable::Visibility symVisibility;
  if (cir::isLocalLinkage(Linkage))
    // If the linkage is local, only default visibility makes sense.
    symVisibility = mlir::SymbolTable::Visibility::Public;
  else if (CXXABI.classifyRTTIUniqueness(Ty, Linkage) ==
           CIRGenItaniumCXXABI::RUK_NonUniqueHidden)
    llvm_unreachable("NYI");
  else
    symVisibility = CIRGenModule::getCIRVisibility(Ty->getVisibility());

  assert(!cir::MissingFeatures::setDLLStorageClass());
  return BuildTypeInfo(loc, Ty, Linkage, symVisibility);
}

void CIRGenItaniumRTTIBuilder::BuildVTablePointer(mlir::Location loc,
                                                  const Type *Ty) {
  auto &builder = CGM.getBuilder();

  // abi::__class_type_info.
  static const char *const ClassTypeInfo =
      "_ZTVN10__cxxabiv117__class_type_infoE";
  // abi::__si_class_type_info.
  static const char *const SIClassTypeInfo =
      "_ZTVN10__cxxabiv120__si_class_type_infoE";
  // abi::__vmi_class_type_info.
  static const char *const VMIClassTypeInfo =
      "_ZTVN10__cxxabiv121__vmi_class_type_infoE";

  const char *VTableName = nullptr;

  switch (Ty->getTypeClass()) {
  case Type::ArrayParameter:
  case Type::HLSLAttributedResource:
    llvm_unreachable("NYI");
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Undeduced type shouldn't get here");

  case Type::Pipe:
    llvm_unreachable("Pipe types shouldn't get here");

  case Type::Builtin:
  case Type::BitInt:
  // GCC treats vector and complex types as fundamental types.
  case Type::Vector:
  case Type::ExtVector:
  case Type::ConstantMatrix:
  case Type::Complex:
  case Type::Atomic:
  // FIXME: GCC treats block pointers as fundamental types?!
  case Type::BlockPointer:
    // abi::__fundamental_type_info.
    VTableName = "_ZTVN10__cxxabiv123__fundamental_type_infoE";
    break;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    // abi::__array_type_info.
    VTableName = "_ZTVN10__cxxabiv117__array_type_infoE";
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // abi::__function_type_info.
    VTableName = "_ZTVN10__cxxabiv120__function_type_infoE";
    break;

  case Type::Enum:
    // abi::__enum_type_info.
    VTableName = "_ZTVN10__cxxabiv116__enum_type_infoE";
    break;

  case Type::Record: {
    const CXXRecordDecl *RD =
        cast<CXXRecordDecl>(cast<RecordType>(Ty)->getDecl());

    if (!RD->hasDefinition() || !RD->getNumBases()) {
      VTableName = ClassTypeInfo;
    } else if (CanUseSingleInheritance(RD)) {
      VTableName = SIClassTypeInfo;
    } else {
      VTableName = VMIClassTypeInfo;
    }

    break;
  }

  case Type::ObjCObject:
    // Ignore protocol qualifiers.
    Ty = cast<ObjCObjectType>(Ty)->getBaseType().getTypePtr();

    // Handle id and Class.
    if (isa<BuiltinType>(Ty)) {
      VTableName = ClassTypeInfo;
      break;
    }

    assert(isa<ObjCInterfaceType>(Ty));
    [[fallthrough]];

  case Type::ObjCInterface:
    if (cast<ObjCInterfaceType>(Ty)->getDecl()->getSuperClass()) {
      VTableName = SIClassTypeInfo;
    } else {
      VTableName = ClassTypeInfo;
    }
    break;

  case Type::ObjCObjectPointer:
  case Type::Pointer:
    // abi::__pointer_type_info.
    VTableName = "_ZTVN10__cxxabiv119__pointer_type_infoE";
    break;

  case Type::MemberPointer:
    // abi::__pointer_to_member_type_info.
    VTableName = "_ZTVN10__cxxabiv129__pointer_to_member_type_infoE";
    break;
  }

  cir::GlobalOp VTable{};

  // Check if the alias exists. If it doesn't, then get or create the global.
  if (CGM.getItaniumVTableContext().isRelativeLayout())
    llvm_unreachable("NYI");
  if (!VTable) {
    VTable = CGM.getOrInsertGlobal(loc, VTableName,
                                   CGM.getBuilder().getUInt8PtrTy());
  }

  if (cir::MissingFeatures::setDSOLocal())
    llvm_unreachable("NYI");

  // The vtable address point is 2.
  mlir::Attribute field{};
  if (CGM.getItaniumVTableContext().isRelativeLayout()) {
    llvm_unreachable("NYI");
  } else {
    SmallVector<mlir::Attribute, 4> offsets{
        CGM.getBuilder().getI32IntegerAttr(2)};
    auto indices = mlir::ArrayAttr::get(builder.getContext(), offsets);
    field = CGM.getBuilder().getGlobalViewAttr(CGM.getBuilder().getUInt8PtrTy(),
                                               VTable, indices);
  }

  assert(field && "expected attribute");
  Fields.push_back(field);
}

cir::GlobalOp
CIRGenItaniumRTTIBuilder::GetAddrOfTypeName(mlir::Location loc, QualType Ty,
                                            cir::GlobalLinkageKind Linkage) {
  auto &builder = CGM.getBuilder();
  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTIName(Ty, Out);

  // We know that the mangled name of the type starts at index 4 of the
  // mangled name of the typename, so we can just index into it in order to
  // get the mangled name of the type.
  auto Init = builder.getString(Name.substr(4),
                                CGM.convertType(CGM.getASTContext().CharTy));
  auto Align =
      CGM.getASTContext().getTypeAlignInChars(CGM.getASTContext().CharTy);

  // builder.getString can return a #cir.zero if the string given to it only
  // contains null bytes. However, type names cannot be full of null bytes.
  // So cast Init to a ConstArrayAttr should be safe.
  auto InitStr = cast<cir::ConstArrayAttr>(Init);

  auto GV = CGM.createOrReplaceCXXRuntimeVariable(loc, Name, InitStr.getType(),
                                                  Linkage, Align);
  CIRGenModule::setInitializer(GV, Init);
  return GV;
}

/// Build an abi::__si_class_type_info, used for single inheritance, according
/// to the Itanium C++ ABI, 2.95p6b.
void CIRGenItaniumRTTIBuilder::BuildSIClassTypeInfo(mlir::Location loc,
                                                    const CXXRecordDecl *RD) {
  // Itanium C++ ABI 2.9.5p6b:
  // It adds to abi::__class_type_info a single member pointing to the
  // type_info structure for the base type,
  auto BaseTypeInfo = CIRGenItaniumRTTIBuilder(CXXABI, CGM)
                          .BuildTypeInfo(loc, RD->bases_begin()->getType());
  Fields.push_back(BaseTypeInfo);
}

namespace {
/// Contains virtual and non-virtual bases seen when traversing a class
/// hierarchy.
struct SeenBases {
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> NonVirtualBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> VirtualBases;
};
} // namespace

/// Compute the value of the flags member in abi::__vmi_class_type_info.
///
static unsigned ComputeVMIClassTypeInfoFlags(const CXXBaseSpecifier *Base,
                                             SeenBases &Bases) {

  unsigned Flags = 0;

  auto *BaseDecl =
      cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());

  if (Base->isVirtual()) {
    // Mark the virtual base as seen.
    if (!Bases.VirtualBases.insert(BaseDecl).second) {
      // If this virtual base has been seen before, then the class is diamond
      // shaped.
      Flags |= CIRGenItaniumRTTIBuilder::VMI_DiamondShaped;
    } else {
      if (Bases.NonVirtualBases.count(BaseDecl))
        Flags |= CIRGenItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    }
  } else {
    // Mark the non-virtual base as seen.
    if (!Bases.NonVirtualBases.insert(BaseDecl).second) {
      // If this non-virtual base has been seen before, then the class has non-
      // diamond shaped repeated inheritance.
      Flags |= CIRGenItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    } else {
      if (Bases.VirtualBases.count(BaseDecl))
        Flags |= CIRGenItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    }
  }

  // Walk all bases.
  for (const auto &I : BaseDecl->bases())
    Flags |= ComputeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}

static unsigned ComputeVMIClassTypeInfoFlags(const CXXRecordDecl *RD) {
  unsigned Flags = 0;
  SeenBases Bases;

  // Walk all bases.
  for (const auto &I : RD->bases())
    Flags |= ComputeVMIClassTypeInfoFlags(&I, Bases);

  return Flags;
}

/// Build an abi::__vmi_class_type_info, used for
/// classes with bases that do not satisfy the abi::__si_class_type_info
/// constraints, according to the Itanium C++ ABI, 2.9.5p5c.
void CIRGenItaniumRTTIBuilder::BuildVMIClassTypeInfo(mlir::Location loc,
                                                     const CXXRecordDecl *RD) {
  auto UnsignedIntLTy = CGM.convertType(CGM.getASTContext().UnsignedIntTy);
  // Itanium C++ ABI 2.9.5p6c:
  //   __flags is a word with flags describing details about the class
  //   structure, which may be referenced by using the __flags_masks
  //   enumeration. These flags refer to both direct and indirect bases.
  unsigned Flags = ComputeVMIClassTypeInfoFlags(RD);
  Fields.push_back(cir::IntAttr::get(UnsignedIntLTy, Flags));

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_count is a word with the number of direct proper base class
  //   descriptions that follow.
  Fields.push_back(cir::IntAttr::get(UnsignedIntLTy, RD->getNumBases()));

  if (!RD->getNumBases())
    return;

  // Now add the base class descriptions.

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_info[] is an array of base class descriptions -- one for every
  //   direct proper base. Each description is of the type:
  //
  //   struct abi::__base_class_type_info {
  //   public:
  //     const __class_type_info *__base_type;
  //     long __offset_flags;
  //
  //     enum __offset_flags_masks {
  //       __virtual_mask = 0x1,
  //       __public_mask = 0x2,
  //       __offset_shift = 8
  //     };
  //   };

  // If we're in mingw and 'long' isn't wide enough for a pointer, use 'long
  // long' instead of 'long' for __offset_flags. libstdc++abi uses long long on
  // LLP64 platforms.
  // FIXME: Consider updating libc++abi to match, and extend this logic to all
  // LLP64 platforms.
  QualType OffsetFlagsTy = CGM.getASTContext().LongTy;
  const TargetInfo &TI = CGM.getASTContext().getTargetInfo();
  if (TI.getTriple().isOSCygMing() &&
      TI.getPointerWidth(LangAS::Default) > TI.getLongWidth())
    OffsetFlagsTy = CGM.getASTContext().LongLongTy;
  auto OffsetFlagsLTy = CGM.convertType(OffsetFlagsTy);

  for (const auto &Base : RD->bases()) {
    // The __base_type member points to the RTTI for the base type.
    Fields.push_back(CIRGenItaniumRTTIBuilder(CXXABI, CGM)
                         .BuildTypeInfo(loc, Base.getType()));

    auto *BaseDecl =
        cast<CXXRecordDecl>(Base.getType()->castAs<RecordType>()->getDecl());

    int64_t OffsetFlags = 0;

    // All but the lower 8 bits of __offset_flags are a signed offset.
    // For a non-virtual base, this is the offset in the object of the base
    // subobject. For a virtual base, this is the offset in the virtual table of
    // the virtual base offset for the virtual base referenced (negative).
    CharUnits Offset;
    if (Base.isVirtual())
      Offset = CGM.getItaniumVTableContext().getVirtualBaseOffsetOffset(
          RD, BaseDecl);
    else {
      const ASTRecordLayout &Layout =
          CGM.getASTContext().getASTRecordLayout(RD);
      Offset = Layout.getBaseClassOffset(BaseDecl);
    }
    OffsetFlags = uint64_t(Offset.getQuantity()) << 8;

    // The low-order byte of __offset_flags contains flags, as given by the
    // masks from the enumeration __offset_flags_masks.
    if (Base.isVirtual())
      OffsetFlags |= BCTI_Virtual;
    if (Base.getAccessSpecifier() == AS_public)
      OffsetFlags |= BCTI_Public;

    Fields.push_back(cir::IntAttr::get(OffsetFlagsLTy, OffsetFlags));
  }
}

mlir::Attribute
CIRGenItaniumRTTIBuilder::GetAddrOfExternalRTTIDescriptor(mlir::Location loc,
                                                          QualType Ty) {
  // Mangle the RTTI name.
  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTI(Ty, Out);
  auto &builder = CGM.getBuilder();

  // Look for an existing global.
  auto GV = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(CGM.getModule(), Name));

  if (!GV) {
    // Create a new global variable.
    // From LLVM codegen => Note for the future: If we would ever like to do
    // deferred emission of RTTI, check if emitting vtables opportunistically
    // need any adjustment.
    GV = CIRGenModule::createGlobalOp(CGM, loc, Name, builder.getUInt8PtrTy(),
                                      /*isConstant=*/true);
    const CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    CGM.setGVProperties(GV, RD);

    // Import the typeinfo symbol when all non-inline virtual methods are
    // imported.
    if (CGM.getTarget().hasPS4DLLImportExport())
      llvm_unreachable("NYI");
  }

  return builder.getGlobalViewAttr(builder.getUInt8PtrTy(), GV);
}

mlir::Attribute CIRGenItaniumRTTIBuilder::BuildTypeInfo(
    mlir::Location loc, QualType Ty, cir::GlobalLinkageKind Linkage,
    mlir::SymbolTable::Visibility Visibility) {
  auto &builder = CGM.getBuilder();
  assert(!cir::MissingFeatures::setDLLStorageClass());

  // Add the vtable pointer.
  BuildVTablePointer(loc, cast<Type>(Ty));

  // And the name.
  auto TypeName = GetAddrOfTypeName(loc, Ty, Linkage);
  mlir::Attribute TypeNameField;

  // If we're supposed to demote the visibility, be sure to set a flag
  // to use a string comparison for type_info comparisons.
  CIRGenItaniumCXXABI::RTTIUniquenessKind RTTIUniqueness =
      CXXABI.classifyRTTIUniqueness(Ty, Linkage);
  if (RTTIUniqueness != CIRGenItaniumCXXABI::RUK_Unique) {
    // The flag is the sign bit, which on ARM64 is defined to be clear
    // for global pointers.  This is very ARM64-specific.
    llvm_unreachable("NYI");
  } else {
    TypeNameField =
        builder.getGlobalViewAttr(builder.getUInt8PtrTy(), TypeName);
  }
  Fields.push_back(TypeNameField);

  switch (Ty->getTypeClass()) {
  case Type::ArrayParameter:
  case Type::HLSLAttributedResource:
    llvm_unreachable("NYI");
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  // GCC treats vector types as fundamental types.
  case Type::Builtin:
  case Type::Vector:
  case Type::ExtVector:
  case Type::ConstantMatrix:
  case Type::Complex:
  case Type::BlockPointer:
    // Itanium C++ ABI 2.9.5p4:
    // abi::__fundamental_type_info adds no data members to std::type_info.
    break;

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Undeduced type shouldn't get here");

  case Type::Pipe:
    break;

  case Type::BitInt:
    break;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__array_type_info adds no data members to std::type_info.
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__function_type_info adds no data members to std::type_info.
    break;

  case Type::Enum:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__enum_type_info adds no data members to std::type_info.
    break;

  case Type::Record: {
    const CXXRecordDecl *RD =
        cast<CXXRecordDecl>(cast<RecordType>(Ty)->getDecl());
    if (!RD->hasDefinition() || !RD->getNumBases()) {
      // We don't need to emit any fields.
      break;
    }

    if (CanUseSingleInheritance(RD)) {
      BuildSIClassTypeInfo(loc, RD);
    } else {
      BuildVMIClassTypeInfo(loc, RD);
    }

    break;
  }

  case Type::ObjCObject:
  case Type::ObjCInterface:
    llvm_unreachable("NYI");
    break;

  case Type::ObjCObjectPointer:
    llvm_unreachable("NYI");
    break;

  case Type::Pointer:
    llvm_unreachable("NYI");
    break;

  case Type::MemberPointer:
    llvm_unreachable("NYI");
    break;

  case Type::Atomic:
    // No fields, at least for the moment.
    break;
  }

  assert(!cir::MissingFeatures::setDLLImportDLLExport());
  auto init = builder.getTypeInfo(builder.getArrayAttr(Fields));

  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  CGM.getCXXABI().getMangleContext().mangleCXXRTTI(Ty, Out);

  // Create new global and search for an existing global.
  auto OldGV = dyn_cast_or_null<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(CGM.getModule(), Name));
  cir::GlobalOp GV =
      CIRGenModule::createGlobalOp(CGM, loc, Name, init.getType(),
                                   /*isConstant=*/true);

  // Export the typeinfo in the same circumstances as the vtable is
  // exported.
  if (CGM.getTarget().hasPS4DLLImportExport())
    llvm_unreachable("NYI");

  // If there's already an old global variable, replace it with the new one.
  if (OldGV) {
    // Replace occurrences of the old variable if needed.
    GV.setName(OldGV.getName());
    if (!OldGV->use_empty()) {
      // TODO: replaceAllUsesWith
      llvm_unreachable("NYI");
    }
    OldGV->erase();
  }

  if (CGM.supportsCOMDAT() && cir::isWeakForLinker(GV.getLinkage())) {
    assert(!cir::MissingFeatures::setComdat());
    llvm_unreachable("NYI");
  }

  CharUnits Align = CGM.getASTContext().toCharUnitsFromBits(
      CGM.getTarget().getPointerAlign(LangAS::Default));
  GV.setAlignmentAttr(CGM.getSize(Align));

  // The Itanium ABI specifies that type_info objects must be globally
  // unique, with one exception: if the type is an incomplete class
  // type or a (possibly indirect) pointer to one.  That exception
  // affects the general case of comparing type_info objects produced
  // by the typeid operator, which is why the comparison operators on
  // std::type_info generally use the type_info name pointers instead
  // of the object addresses.  However, the language's built-in uses
  // of RTTI generally require class types to be complete, even when
  // manipulating pointers to those class types.  This allows the
  // implementation of dynamic_cast to rely on address equality tests,
  // which is much faster.
  //
  // All of this is to say that it's important that both the type_info
  // object and the type_info name be uniqued when weakly emitted.

  // TODO(cir): setup other bits for TypeName
  assert(!cir::MissingFeatures::setDLLStorageClass());
  assert(!cir::MissingFeatures::setPartition());
  assert(!cir::MissingFeatures::setDSOLocal());
  mlir::SymbolTable::setSymbolVisibility(
      TypeName, CIRGenModule::getMLIRVisibility(TypeName));

  // TODO(cir): setup other bits for GV
  assert(!cir::MissingFeatures::setDLLStorageClass());
  assert(!cir::MissingFeatures::setPartition());
  assert(!cir::MissingFeatures::setDSOLocal());
  CIRGenModule::setInitializer(GV, init);

  return builder.getGlobalViewAttr(builder.getUInt8PtrTy(), GV);
  ;
}

mlir::Attribute CIRGenItaniumCXXABI::getAddrOfRTTIDescriptor(mlir::Location loc,
                                                             QualType Ty) {
  return CIRGenItaniumRTTIBuilder(*this, CGM).BuildTypeInfo(loc, Ty);
}

void CIRGenItaniumCXXABI::emitVTableDefinitions(CIRGenVTables &CGVT,
                                                const CXXRecordDecl *RD) {
  auto VTable = getAddrOfVTable(RD, CharUnits());
  if (VTable.hasInitializer())
    return;

  ItaniumVTableContext &VTContext = CGM.getItaniumVTableContext();
  const VTableLayout &VTLayout = VTContext.getVTableLayout(RD);
  auto Linkage = CGM.getVTableLinkage(RD);
  auto RTTI = CGM.getAddrOfRTTIDescriptor(
      CGM.getLoc(RD->getBeginLoc()), CGM.getASTContext().getTagDeclType(RD));

  // Create and set the initializer.
  ConstantInitBuilder builder(CGM);
  auto components = builder.beginRecord();

  CGVT.createVTableInitializer(components, VTLayout, RTTI,
                               cir::isLocalLinkage(Linkage));
  components.finishAndSetAsInitializer(VTable, /*forVtable=*/true);

  // Set the correct linkage.
  VTable.setLinkage(Linkage);

  if (CGM.supportsCOMDAT() && cir::isWeakForLinker(Linkage)) {
    assert(!cir::MissingFeatures::setComdat());
  }

  // Set the right visibility.
  CGM.setGVProperties(VTable, RD);

  // If this is the magic class __cxxabiv1::__fundamental_type_info,
  // we will emit the typeinfo for the fundamental types. This is the
  // same behaviour as GCC.
  const DeclContext *DC = RD->getDeclContext();
  if (RD->getIdentifier() &&
      RD->getIdentifier()->isStr("__fundamental_type_info") &&
      isa<NamespaceDecl>(DC) && cast<NamespaceDecl>(DC)->getIdentifier() &&
      cast<NamespaceDecl>(DC)->getIdentifier()->isStr("__cxxabiv1") &&
      DC->getParent()->isTranslationUnit()) {
    llvm_unreachable("NYI");
    // EmitFundamentalRTTIDescriptors(RD);
  }

  auto VTableAsGlobalValue = dyn_cast<cir::CIRGlobalValueInterface>(*VTable);
  assert(VTableAsGlobalValue && "VTable must support CIRGlobalValueInterface");
  bool isDeclarationForLinker = VTableAsGlobalValue.isDeclarationForLinker();
  // Always emit type metadata on non-available_externally definitions, and on
  // available_externally definitions if we are performing whole program
  // devirtualization. For WPD we need the type metadata on all vtable
  // definitions to ensure we associate derived classes with base classes
  // defined in headers but with a strong definition only in a shared
  // library.
  if (!isDeclarationForLinker || CGM.getCodeGenOpts().WholeProgramVTables) {
    CGM.emitVTableTypeMetadata(RD, VTable, VTLayout);
    // For available_externally definitions, add the vtable to
    // @llvm.compiler.used so that it isn't deleted before whole program
    // analysis.
    if (isDeclarationForLinker) {
      llvm_unreachable("NYI");
      assert(CGM.getCodeGenOpts().WholeProgramVTables);
      assert(!cir::MissingFeatures::addCompilerUsedGlobal());
    }
  }

  if (VTContext.isRelativeLayout())
    llvm_unreachable("NYI");
}

mlir::Value CIRGenItaniumCXXABI::emitVirtualDestructorCall(
    CIRGenFunction &CGF, const CXXDestructorDecl *dtor, CXXDtorType dtorType,
    Address thisAddr, DeleteOrMemberCallExpr expr) {
  auto *callExpr = dyn_cast<const CXXMemberCallExpr *>(expr);
  auto *delExpr = dyn_cast<const CXXDeleteExpr *>(expr);
  assert((callExpr != nullptr) ^ (delExpr != nullptr));
  assert(callExpr == nullptr || callExpr->arg_begin() == callExpr->arg_end());
  assert(dtorType == Dtor_Deleting || dtorType == Dtor_Complete);

  GlobalDecl globalDecl(dtor, dtorType);
  const CIRGenFunctionInfo *fnInfo =
      &CGM.getTypes().arrangeCXXStructorDeclaration(globalDecl);
  auto fnTy = CGF.CGM.getTypes().GetFunctionType(*fnInfo);
  auto callee = CIRGenCallee::forVirtual(callExpr, globalDecl, thisAddr, fnTy);

  QualType thisTy;
  if (callExpr)
    thisTy = callExpr->getObjectType();
  else
    thisTy = delExpr->getDestroyedType();

  CGF.emitCXXDestructorCall(globalDecl, callee, thisAddr.emitRawPointer(),
                            thisTy, nullptr, QualType(), nullptr);
  return nullptr;
}

void CIRGenItaniumCXXABI::emitVirtualInheritanceTables(
    const CXXRecordDecl *RD) {
  CIRGenVTables &VTables = CGM.getVTables();
  auto VTT = VTables.getAddrOfVTT(RD);
  VTables.emitVTTDefinition(VTT, CGM.getVTableLinkage(RD), RD);
}

/// What sort of uniqueness rules should we use for the RTTI for the
/// given type?
CIRGenItaniumCXXABI::RTTIUniquenessKind
CIRGenItaniumCXXABI::classifyRTTIUniqueness(
    QualType CanTy, cir::GlobalLinkageKind Linkage) const {
  if (shouldRTTIBeUnique())
    return RUK_Unique;

  // It's only necessary for linkonce_odr or weak_odr linkage.
  if (Linkage != cir::GlobalLinkageKind::LinkOnceODRLinkage &&
      Linkage != cir::GlobalLinkageKind::WeakODRLinkage)
    return RUK_Unique;

  // It's only necessary with default visibility.
  if (CanTy->getVisibility() != DefaultVisibility)
    return RUK_Unique;

  // If we're not required to publish this symbol, hide it.
  if (Linkage == cir::GlobalLinkageKind::LinkOnceODRLinkage)
    return RUK_NonUniqueHidden;

  // If we're required to publish this symbol, as we might be under an
  // explicit instantiation, leave it with default visibility but
  // enable string-comparisons.
  assert(Linkage == cir::GlobalLinkageKind::WeakODRLinkage);
  return RUK_NonUniqueVisible;
}

void CIRGenItaniumCXXABI::emitDestructorCall(
    CIRGenFunction &CGF, const CXXDestructorDecl *DD, CXXDtorType Type,
    bool ForVirtualBase, bool Delegating, Address This, QualType ThisTy) {
  GlobalDecl GD(DD, Type);
  auto VTT =
      getCXXDestructorImplicitParam(CGF, DD, Type, ForVirtualBase, Delegating);
  QualType VTTTy = getContext().getPointerType(getContext().VoidPtrTy);
  CIRGenCallee Callee;
  if (getContext().getLangOpts().AppleKext && Type != Dtor_Base &&
      DD->isVirtual())
    llvm_unreachable("NYI");
  else
    Callee = CIRGenCallee::forDirect(CGM.getAddrOfCXXStructor(GD), GD);

  CGF.emitCXXDestructorCall(GD, Callee, This.getPointer(), ThisTy, VTT, VTTTy,
                            nullptr);
}

void CIRGenItaniumCXXABI::registerGlobalDtor(CIRGenFunction &CGF,
                                             const VarDecl *D, cir::FuncOp dtor,
                                             mlir::Value Addr) {
  if (D->isNoDestroy(CGM.getASTContext()))
    return;

  if (D->getTLSKind())
    llvm_unreachable("NYI");

  // HLSL doesn't support atexit.
  if (CGM.getLangOpts().HLSL)
    llvm_unreachable("NYI");

  // The default behavior is to use atexit. This is handled in lowering
  // prepare. Nothing to be done for CIR here.
}

mlir::Value CIRGenItaniumCXXABI::getCXXDestructorImplicitParam(
    CIRGenFunction &CGF, const CXXDestructorDecl *DD, CXXDtorType Type,
    bool ForVirtualBase, bool Delegating) {
  GlobalDecl GD(DD, Type);
  return CGF.GetVTTParameter(GD, ForVirtualBase, Delegating);
}

// The idea here is creating a separate block for the throw with an
// `UnreachableOp` as the terminator. So, we branch from the current block
// to the throw block and create a block for the remaining operations.
void insertThrowAndSplit(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value exceptionPtr,
                         mlir::FlatSymbolRefAttr typeInfo,
                         mlir::FlatSymbolRefAttr dtor) {
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Region *region = currentBlock->getParent();

  if (currentBlock->empty()) {
    builder.create<cir::ThrowOp>(loc, exceptionPtr, typeInfo, dtor);
    builder.create<cir::UnreachableOp>(loc);
  } else {
    mlir::Block *throwBlock = builder.createBlock(region);
    builder.create<cir::ThrowOp>(loc, exceptionPtr, typeInfo, dtor);
    builder.create<cir::UnreachableOp>(loc);

    builder.setInsertionPointToEnd(currentBlock);
    builder.create<cir::BrOp>(loc, throwBlock);
  }

  (void)builder.createBlock(region);
  // This will be erased during codegen, it acts as a placeholder for the
  // operations to be inserted (if any)
  builder.create<cir::ScopeOp>(loc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 b.create<cir::YieldOp>(loc);
                               });
}

void CIRGenItaniumCXXABI::emitRethrow(CIRGenFunction &CGF, bool isNoReturn) {
  // void __cxa_rethrow();

  if (isNoReturn) {
    auto &builder = CGF.getBuilder();
    assert(CGF.currSrcLoc && "expected source location");
    auto loc = *CGF.currSrcLoc;
    insertThrowAndSplit(builder, loc, mlir::Value{}, mlir::FlatSymbolRefAttr{},
                        mlir::FlatSymbolRefAttr{});
    CGF.mayThrow = true;
  } else {
    llvm_unreachable("NYI");
  }
}

void CIRGenItaniumCXXABI::emitThrow(CIRGenFunction &CGF,
                                    const CXXThrowExpr *E) {
  // This differs a bit from LLVM codegen, CIR has native operations for some
  // cxa functions, and defers allocation size computation, always pass the dtor
  // symbol, etc. CIRGen also does not use getAllocateExceptionFn / getThrowFn.

  // Now allocate the exception object.
  auto &builder = CGF.getBuilder();
  QualType clangThrowType = E->getSubExpr()->getType();
  auto throwTy = builder.getPointerTo(CGF.convertType(clangThrowType));
  uint64_t typeSize =
      CGF.getContext().getTypeSizeInChars(clangThrowType).getQuantity();
  auto subExprLoc = CGF.getLoc(E->getSubExpr()->getSourceRange());
  // Defer computing allocation size to some later lowering pass.
  auto exceptionPtr =
      builder
          .create<cir::AllocExceptionOp>(subExprLoc, throwTy,
                                         builder.getI64IntegerAttr(typeSize))
          .getAddr();

  // Build expression and store its result into exceptionPtr.
  CharUnits exnAlign = CGF.getContext().getExnObjectAlignment();
  CGF.emitAnyExprToExn(E->getSubExpr(), Address(exceptionPtr, exnAlign));

  // Get the RTTI symbol address.
  auto typeInfo = mlir::dyn_cast_if_present<cir::GlobalViewAttr>(
      CGM.getAddrOfRTTIDescriptor(subExprLoc, clangThrowType,
                                  /*ForEH=*/true));
  assert(typeInfo && "expected GlobalViewAttr typeinfo");
  assert(!typeInfo.getIndices() && "expected no indirection");

  // The address of the destructor.
  //
  // Note: LLVM codegen already optimizes out the dtor if the
  // type is a record with trivial dtor (by passing down a
  // null dtor). In CIR, we forward this info and allow for
  // LoweringPrepare or some other pass to skip passing the
  // trivial function.
  //
  // TODO(cir): However, such lowering is still NYI, and for
  // the sake of getting cir.throw right, the same OG path is
  // follows here.
  mlir::FlatSymbolRefAttr dtor{};
  if (const RecordType *recordTy = clangThrowType->getAs<RecordType>()) {
    CXXRecordDecl *rec = cast<CXXRecordDecl>(recordTy->getDecl());
    assert(!cir::MissingFeatures::isTrivialCtorOrDtor());
    if (!rec->hasTrivialDestructor()) {
      CXXDestructorDecl *dtorD = rec->getDestructor();
      dtor = mlir::FlatSymbolRefAttr::get(
          CGM.getAddrOfCXXStructor(GlobalDecl(dtorD, Dtor_Complete))
              .getSymNameAttr());
    }
  }

  // FIXME: When adding support for invoking, we should wrap the throw op
  // below into a try, and let CFG flatten pass to generate a cir.try_call.
  assert(!CGF.isInvokeDest() && "landing pad like logic NYI");

  // Now throw the exception.
  mlir::Location loc = CGF.getLoc(E->getSourceRange());
  insertThrowAndSplit(builder, loc, exceptionPtr, typeInfo.getSymbol(), dtor);

  CGF.mayThrow = true;
}

mlir::Value CIRGenItaniumCXXABI::getVirtualBaseClassOffset(
    mlir::Location loc, CIRGenFunction &CGF, Address This,
    const CXXRecordDecl *ClassDecl, const CXXRecordDecl *BaseClassDecl) {
  auto VTablePtr = CGF.getVTablePtr(loc, This, CGM.UInt8PtrTy, ClassDecl);
  CharUnits VBaseOffsetOffset =
      CGM.getItaniumVTableContext().getVirtualBaseOffsetOffset(ClassDecl,
                                                               BaseClassDecl);
  mlir::Value OffsetVal =
      CGF.getBuilder().getSInt64(VBaseOffsetOffset.getQuantity(), loc);
  auto VBaseOffsetPtr = CGF.getBuilder().create<cir::PtrStrideOp>(
      loc, VTablePtr.getType(), VTablePtr,
      OffsetVal); // vbase.offset.ptr

  mlir::Value VBaseOffset;
  if (CGM.getItaniumVTableContext().isRelativeLayout()) {
    VBaseOffset = CGF.getBuilder().createLoad(
        loc, Address(VBaseOffsetPtr, CGM.SInt32Ty,
                     CharUnits::fromQuantity(4))); // vbase.offset
  } else {
    auto OffsetPtr = CGF.getBuilder().createBitcast(
        VBaseOffsetPtr, CGF.getBuilder().getPointerTo(CGM.PtrDiffTy));
    VBaseOffset = CGF.getBuilder().createLoad(
        loc, Address(OffsetPtr, CGM.PtrDiffTy,
                     CGF.getPointerAlign())); // vbase.offset
  }
  return VBaseOffset;
}

static cir::FuncOp getBadCastFn(CIRGenFunction &CGF) {
  // Prototype: void __cxa_bad_cast();

  // TODO(cir): set the calling convention of the runtime function.
  assert(!cir::MissingFeatures::setCallingConv());

  cir::FuncType FTy =
      CGF.getBuilder().getFuncType({}, CGF.getBuilder().getVoidTy());
  return CGF.CGM.createRuntimeFunction(FTy, "__cxa_bad_cast");
}

static void emitCallToBadCast(CIRGenFunction &CGF, mlir::Location loc) {
  // TODO(cir): set the calling convention to the runtime function.
  assert(!cir::MissingFeatures::setCallingConv());

  CGF.emitRuntimeCall(loc, getBadCastFn(CGF));
  CGF.getBuilder().create<cir::UnreachableOp>(loc);
  CGF.getBuilder().clearInsertionPoint();
}

void CIRGenItaniumCXXABI::emitBadCastCall(CIRGenFunction &CGF,
                                          mlir::Location loc) {
  emitCallToBadCast(CGF, loc);
}

static CharUnits computeOffsetHint(ASTContext &astContext,
                                   const CXXRecordDecl *Src,
                                   const CXXRecordDecl *Dst) {
  CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);

  // If Dst is not derived from Src we can skip the whole computation below and
  // return that Src is not a public base of Dst.  Record all inheritance paths.
  if (!Dst->isDerivedFrom(Src, Paths))
    return CharUnits::fromQuantity(-2ULL);

  unsigned NumPublicPaths = 0;
  CharUnits Offset;

  // Now walk all possible inheritance paths.
  for (const CXXBasePath &Path : Paths) {
    if (Path.Access != AS_public) // Ignore non-public inheritance.
      continue;

    ++NumPublicPaths;

    for (const CXXBasePathElement &PathElement : Path) {
      // If the path contains a virtual base class we can't give any hint.
      // -1: no hint.
      if (PathElement.Base->isVirtual())
        return CharUnits::fromQuantity(-1ULL);

      if (NumPublicPaths > 1) // Won't use offsets, skip computation.
        continue;

      // Accumulate the base class offsets.
      const ASTRecordLayout &L =
          astContext.getASTRecordLayout(PathElement.Class);
      Offset += L.getBaseClassOffset(
          PathElement.Base->getType()->getAsCXXRecordDecl());
    }
  }

  // -2: Src is not a public base of Dst.
  if (NumPublicPaths == 0)
    return CharUnits::fromQuantity(-2ULL);

  // -3: Src is a multiple public base type but never a virtual base type.
  if (NumPublicPaths > 1)
    return CharUnits::fromQuantity(-3ULL);

  // Otherwise, the Src type is a unique public nonvirtual base type of Dst.
  // Return the offset of Src from the origin of Dst.
  return Offset;
}

static cir::FuncOp getItaniumDynamicCastFn(CIRGenFunction &CGF) {
  // Prototype:
  // void *__dynamic_cast(const void *sub,
  //                      global_as const abi::__class_type_info *src,
  //                      global_as const abi::__class_type_info *dst,
  //                      std::ptrdiff_t src2dst_offset);

  mlir::Type VoidPtrTy = CGF.VoidPtrTy;
  mlir::Type RTTIPtrTy = CGF.getBuilder().getUInt8PtrTy();
  mlir::Type PtrDiffTy = CGF.convertType(CGF.getContext().getPointerDiffType());

  // TODO(cir): mark the function as nowind readonly.

  // TODO(cir): set the calling convention of the runtime function.
  assert(!cir::MissingFeatures::setCallingConv());

  cir::FuncType FTy = CGF.getBuilder().getFuncType(
      {VoidPtrTy, RTTIPtrTy, RTTIPtrTy, PtrDiffTy}, VoidPtrTy);
  return CGF.CGM.createRuntimeFunction(FTy, "__dynamic_cast");
}

static Address emitDynamicCastToVoid(CIRGenFunction &CGF, mlir::Location Loc,
                                     QualType SrcRecordTy, Address Src) {
  auto vtableUsesRelativeLayout =
      CGF.CGM.getItaniumVTableContext().isRelativeLayout();
  auto ptr = CGF.getBuilder().createDynCastToVoid(Loc, Src.getPointer(),
                                                  vtableUsesRelativeLayout);
  return Address{ptr, Src.getAlignment()};
}

static mlir::Value emitExactDynamicCast(CIRGenItaniumCXXABI &ABI,
                                        CIRGenFunction &CGF, mlir::Location Loc,
                                        QualType SrcRecordTy,
                                        QualType DestRecordTy,
                                        cir::PointerType DestCIRTy,
                                        bool IsRefCast, Address Src) {
  // Find all the inheritance paths from SrcRecordTy to DestRecordTy.
  const CXXRecordDecl *SrcDecl = SrcRecordTy->getAsCXXRecordDecl();
  const CXXRecordDecl *DestDecl = DestRecordTy->getAsCXXRecordDecl();
  CXXBasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);
  (void)DestDecl->isDerivedFrom(SrcDecl, Paths);

  // Find an offset within `DestDecl` where a `SrcDecl` instance and its vptr
  // might appear.
  std::optional<CharUnits> Offset;
  for (const CXXBasePath &Path : Paths) {
    // dynamic_cast only finds public inheritance paths.
    if (Path.Access != AS_public)
      continue;

    CharUnits PathOffset;
    for (const CXXBasePathElement &PathElement : Path) {
      // Find the offset along this inheritance step.
      const CXXRecordDecl *Base =
          PathElement.Base->getType()->getAsCXXRecordDecl();
      if (PathElement.Base->isVirtual()) {
        // For a virtual base class, we know that the derived class is exactly
        // DestDecl, so we can use the vbase offset from its layout.
        const ASTRecordLayout &L =
            CGF.getContext().getASTRecordLayout(DestDecl);
        PathOffset = L.getVBaseClassOffset(Base);
      } else {
        const ASTRecordLayout &L =
            CGF.getContext().getASTRecordLayout(PathElement.Class);
        PathOffset += L.getBaseClassOffset(Base);
      }
    }

    if (!Offset)
      Offset = PathOffset;
    else if (Offset != PathOffset) {
      // Base appears in at least two different places. Find the most-derived
      // object and see if it's a DestDecl. Note that the most-derived object
      // must be at least as aligned as this base class subobject, and must
      // have a vptr at offset 0.
      Src = emitDynamicCastToVoid(CGF, Loc, SrcRecordTy, Src);
      SrcDecl = DestDecl;
      Offset = CharUnits::Zero();
      break;
    }
  }

  if (!Offset) {
    // If there are no public inheritance paths, the cast always fails.
    mlir::Value NullPtrValue = CGF.getBuilder().getNullPtr(DestCIRTy, Loc);
    if (IsRefCast) {
      auto *CurrentRegion = CGF.getBuilder().getBlock()->getParent();
      emitCallToBadCast(CGF, Loc);

      // The call to bad_cast will terminate the block. Create a new block to
      // hold any follow up code.
      CGF.getBuilder().createBlock(CurrentRegion, CurrentRegion->end());
    }

    return NullPtrValue;
  }

  // Compare the vptr against the expected vptr for the destination type at
  // this offset. Note that we do not know what type Src points to in the case
  // where the derived class multiply inherits from the base class so we can't
  // use GetVTablePtr, so we load the vptr directly instead.

  mlir::Value ExpectedVPtr =
      ABI.getVTableAddressPoint(BaseSubobject(SrcDecl, *Offset), DestDecl);

  // TODO(cir): handle address space here.
  assert(!cir::MissingFeatures::addressSpace());
  mlir::Type VPtrTy = ExpectedVPtr.getType();
  mlir::Type VPtrPtrTy = CGF.getBuilder().getPointerTo(VPtrTy);
  Address SrcVPtrPtr(
      CGF.getBuilder().createBitcast(Src.getPointer(), VPtrPtrTy),
      Src.getAlignment());
  mlir::Value SrcVPtr = CGF.getBuilder().createLoad(Loc, SrcVPtrPtr);

  // TODO(cir): decorate SrcVPtr with TBAA info.
  assert(!cir::MissingFeatures::tbaa());

  mlir::Value Success = CGF.getBuilder().createCompare(Loc, cir::CmpOpKind::eq,
                                                       SrcVPtr, ExpectedVPtr);

  auto emitCastResult = [&] {
    if (Offset->isZero())
      return CGF.getBuilder().createBitcast(Src.getPointer(), DestCIRTy);

    // TODO(cir): handle address space here.
    assert(!cir::MissingFeatures::addressSpace());
    mlir::Type U8PtrTy =
        CGF.getBuilder().getPointerTo(CGF.getBuilder().getUInt8Ty());

    mlir::Value StrideToApply = CGF.getBuilder().getConstInt(
        Loc, CGF.getBuilder().getUInt64Ty(), Offset->getQuantity());
    mlir::Value SrcU8Ptr =
        CGF.getBuilder().createBitcast(Src.getPointer(), U8PtrTy);
    mlir::Value ResultU8Ptr = CGF.getBuilder().create<cir::PtrStrideOp>(
        Loc, U8PtrTy, SrcU8Ptr, StrideToApply);
    return CGF.getBuilder().createBitcast(ResultU8Ptr, DestCIRTy);
  };

  if (IsRefCast) {
    mlir::Value Failed = CGF.getBuilder().createNot(Success);
    CGF.getBuilder().create<cir::IfOp>(Loc, Failed, /*withElseRegion=*/false,
                                       [&](mlir::OpBuilder &, mlir::Location) {
                                         emitCallToBadCast(CGF, Loc);
                                       });
    return emitCastResult();
  }

  return CGF.getBuilder()
      .create<cir::TernaryOp>(
          Loc, Success,
          [&](mlir::OpBuilder &, mlir::Location) {
            auto Result = emitCastResult();
            CGF.getBuilder().createYield(Loc, Result);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value NullPtrValue =
                CGF.getBuilder().getNullPtr(DestCIRTy, Loc);
            CGF.getBuilder().createYield(Loc, NullPtrValue);
          })
      .getResult();
}

static cir::DynamicCastInfoAttr emitDynamicCastInfo(CIRGenFunction &CGF,
                                                    mlir::Location Loc,
                                                    QualType SrcRecordTy,
                                                    QualType DestRecordTy) {
  auto srcRtti = mlir::cast<cir::GlobalViewAttr>(
      CGF.CGM.getAddrOfRTTIDescriptor(Loc, SrcRecordTy));
  auto destRtti = mlir::cast<cir::GlobalViewAttr>(
      CGF.CGM.getAddrOfRTTIDescriptor(Loc, DestRecordTy));

  auto runtimeFuncOp = getItaniumDynamicCastFn(CGF);
  auto badCastFuncOp = getBadCastFn(CGF);
  auto runtimeFuncRef = mlir::FlatSymbolRefAttr::get(runtimeFuncOp);
  auto badCastFuncRef = mlir::FlatSymbolRefAttr::get(badCastFuncOp);

  const CXXRecordDecl *srcDecl = SrcRecordTy->getAsCXXRecordDecl();
  const CXXRecordDecl *destDecl = DestRecordTy->getAsCXXRecordDecl();
  auto offsetHint = computeOffsetHint(CGF.getContext(), srcDecl, destDecl);

  mlir::Type ptrdiffTy = CGF.convertType(CGF.getContext().getPointerDiffType());
  auto offsetHintAttr = cir::IntAttr::get(ptrdiffTy, offsetHint.getQuantity());

  return cir::DynamicCastInfoAttr::get(srcRtti, destRtti, runtimeFuncRef,
                                       badCastFuncRef, offsetHintAttr);
}

mlir::Value CIRGenItaniumCXXABI::emitDynamicCast(CIRGenFunction &CGF,
                                                 mlir::Location Loc,
                                                 QualType SrcRecordTy,
                                                 QualType DestRecordTy,
                                                 cir::PointerType DestCIRTy,
                                                 bool isRefCast, Address Src) {
  bool isCastToVoid = DestRecordTy.isNull();
  assert((!isCastToVoid || !isRefCast) && "cannot cast to void reference");

  if (isCastToVoid)
    return emitDynamicCastToVoid(CGF, Loc, SrcRecordTy, Src).getPointer();

  // If the destination is effectively final, the cast succeeds if and only
  // if the dynamic type of the pointer is exactly the destination type.
  if (DestRecordTy->getAsCXXRecordDecl()->isEffectivelyFinal() &&
      CGF.CGM.getCodeGenOpts().OptimizationLevel > 0)
    return emitExactDynamicCast(*this, CGF, Loc, SrcRecordTy, DestRecordTy,
                                DestCIRTy, isRefCast, Src);

  auto castInfo = emitDynamicCastInfo(CGF, Loc, SrcRecordTy, DestRecordTy);
  return CGF.getBuilder().createDynCast(Loc, Src.getPointer(), DestCIRTy,
                                        isRefCast, castInfo);
}

cir::MethodAttr
CIRGenItaniumCXXABI::buildVirtualMethodAttr(cir::MethodType MethodTy,
                                            const CXXMethodDecl *MD) {
  assert(MD->isVirtual() && "only deal with virtual member functions");

  uint64_t Index = CGM.getItaniumVTableContext().getMethodVTableIndex(MD);
  uint64_t VTableOffset;
  if (CGM.getItaniumVTableContext().isRelativeLayout()) {
    // Multiply by 4-byte relative offsets.
    VTableOffset = Index * 4;
  } else {
    const ASTContext &astContext = getContext();
    CharUnits PointerWidth = astContext.toCharUnitsFromBits(
        astContext.getTargetInfo().getPointerWidth(LangAS::Default));
    VTableOffset = Index * PointerWidth.getQuantity();
  }

  return cir::MethodAttr::get(MethodTy, VTableOffset);
}

/// The Itanium ABI requires non-zero initialization only for data
/// member pointers, for which '0' is a valid offset.
bool CIRGenItaniumCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  return MPT->isMemberFunctionPointer();
}

/// The Itanium ABI always places an offset to the complete object
/// at entry -2 in the vtable.
void CIRGenItaniumCXXABI::emitVirtualObjectDelete(
    CIRGenFunction &CGF, const CXXDeleteExpr *delExpr, Address ptr,
    QualType elementType, const CXXDestructorDecl *dtor) {
  bool useGlobalDelete = delExpr->isGlobalDelete();
  if (useGlobalDelete)
    llvm_unreachable("NYI");

  CXXDtorType dtorType = useGlobalDelete ? Dtor_Complete : Dtor_Deleting;
  emitVirtualDestructorCall(CGF, dtor, dtorType, ptr, delExpr);

  if (useGlobalDelete)
    llvm_unreachable("NYI");
}

/************************** Array allocation cookies **************************/

CharUnits CIRGenItaniumCXXABI::getArrayCookieSizeImpl(QualType ElementType) {
  // The array cookie is a size_t; pad that up to the element alignment.
  // The cookie is actually right-justified in that space.
  return std::max(
      CharUnits::fromQuantity(CGM.SizeSizeInBytes),
      CGM.getASTContext().getPreferredTypeAlignInChars(ElementType));
}

Address CIRGenItaniumCXXABI::initializeArrayCookie(CIRGenFunction &CGF,
                                                   Address NewPtr,
                                                   mlir::Value NumElements,
                                                   const CXXNewExpr *E,
                                                   QualType ElementType) {
  assert(requiresArrayCookie(E));

  // TODO: Get the address space when sanitizer support is implemented

  ASTContext &Ctx = getContext();
  CharUnits SizeSize = CGF.getSizeSize();
  mlir::Location Loc = CGF.getLoc(E->getSourceRange());

  // The size of the cookie.
  CharUnits CookieSize =
      std::max(SizeSize, Ctx.getPreferredTypeAlignInChars(ElementType));
  assert(CookieSize == getArrayCookieSizeImpl(ElementType));

  // Compute an offset to the cookie.
  Address CookiePtr = NewPtr;
  CharUnits CookieOffset = CookieSize - SizeSize;
  if (!CookieOffset.isZero()) {
    auto CastOp = CGF.getBuilder().createPtrBitcast(
        CookiePtr.getPointer(), CGF.getBuilder().getUIntNTy(8));
    auto OffsetOp = CGF.getBuilder().getSignedInt(
        Loc, CookieOffset.getQuantity(), /*width=*/32);
    auto DataPtr = CGF.getBuilder().createPtrStride(Loc, CastOp, OffsetOp);
    CookiePtr =
        Address(DataPtr, CGF.getBuilder().getUIntNTy(8), NewPtr.getAlignment());
  }

  // Write the number of elements into the appropriate slot.
  Address NumElementsPtr =
      CookiePtr.withElementType(CGF.getBuilder(), CGF.SizeTy);
  CGF.getBuilder().createStore(Loc, NumElements, NumElementsPtr);

  if (CGF.SanOpts.has(SanitizerKind::Address))
    llvm_unreachable("NYI");

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  int64_t Offset = (CookieSize.getQuantity());
  auto CastOp = CGF.getBuilder().createPtrBitcast(
      NewPtr.getPointer(), CGF.getBuilder().getUIntNTy(8));
  auto OffsetOp = CGF.getBuilder().getSignedInt(Loc, Offset, /*width=*/32);
  auto DataPtr = CGF.getBuilder().createPtrStride(Loc, CastOp, OffsetOp);
  return Address(DataPtr, CGF.getBuilder().getUIntNTy(8),
                 NewPtr.getAlignment());
}

CharUnits CIRGenARMCXXABI::getArrayCookieSizeImpl(QualType elementType) {
  // ARM says that the cookie is always:
  //   struct array_cookie {
  //     std::size_t element_size; // element_size != 0
  //     std::size_t element_count;
  //   };
  // But the base ABI doesn't give anything an alignment greater than
  // 8, so we can dismiss this as typical ABI-author blindness to
  // actual language complexity and round up to the element alignment.
  return std::max(CharUnits::fromQuantity(2 * CGM.SizeSizeInBytes),
                  getContext().getTypeAlignInChars(elementType));
}

Address CIRGenARMCXXABI::initializeArrayCookie(CIRGenFunction &cgf,
                                               Address newPtr,
                                               mlir::Value numElements,
                                               const CXXNewExpr *expr,
                                               QualType elementType) {
  assert(requiresArrayCookie(expr));

  // The cookie is always at the start of the buffer.
  auto cookiePtr =
      cgf.getBuilder().createPtrBitcast(newPtr.getPointer(), cgf.SizeTy);
  Address cookie = Address(cookiePtr, cgf.SizeTy, newPtr.getAlignment());

  ASTContext &ctx = getContext();
  mlir::Location loc = cgf.getLoc(expr->getSourceRange());

  // The first element is the element size.
  mlir::Value elementSize = cgf.getBuilder().getConstInt(
      loc, cgf.SizeTy, ctx.getTypeSizeInChars(elementType).getQuantity());
  cgf.getBuilder().createStore(loc, elementSize, cookie);

  // The second element is the element count.
  auto offsetOp = cgf.getBuilder().getSignedInt(loc, 1, /*width=*/32);
  auto dataPtr =
      cgf.getBuilder().createPtrStride(loc, cookie.getPointer(), offsetOp);
  cookie = Address(dataPtr, cgf.SizeTy, newPtr.getAlignment());
  cgf.getBuilder().createStore(loc, numElements, cookie);

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  CharUnits cookieSize = CIRGenARMCXXABI::getArrayCookieSizeImpl(elementType);
  offsetOp = cgf.getBuilder().getSignedInt(loc, cookieSize.getQuantity(),
                                           /*width=*/32);
  auto castOp = cgf.getBuilder().createPtrBitcast(
      newPtr.getPointer(), cgf.getBuilder().getUIntNTy(8));
  dataPtr = cgf.getBuilder().createPtrStride(loc, castOp, offsetOp);
  return Address(dataPtr, cgf.getBuilder().getUIntNTy(8),
                 newPtr.getAlignment());
}
