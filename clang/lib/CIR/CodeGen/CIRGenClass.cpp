//===--- CIRGenClass.cpp - Emit CIR Code for C++ classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of classes
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

/// Checks whether the given constructor is a valid subject for the
/// complete-to-base constructor delgation optimization, i.e. emitting the
/// complete constructor as a simple call to the base constructor.
bool CIRGenFunction::IsConstructorDelegationValid(
    const CXXConstructorDecl *Ctor) {

  // Currently we disable the optimization for classes with virtual bases
  // because (1) the address of parameter variables need to be consistent across
  // all initializers but (2) the delegate function call necessarily creates a
  // second copy of the parameter variable.
  //
  // The limiting example (purely theoretical AFAIK):
  //   struct A { A(int &c) { c++; } };
  //   struct A : virtual A {
  //     B(int count) : A(count) { printf("%d\n", count); }
  //   };
  // ...although even this example could in principle be emitted as a delegation
  // since the address of the parameter doesn't escape.
  if (Ctor->getParent()->getNumVBases())
    return false;

  // We also disable the optimization for variadic functions because it's
  // impossible to "re-pass" varargs.
  if (Ctor->getType()->castAs<FunctionProtoType>()->isVariadic())
    return false;

  // FIXME: Decide if we can do a delegation of a delegating constructor.
  if (Ctor->isDelegatingConstructor())
    return false;

  return true;
}

/// TODO(cir): strong candidate for AST helper to be shared between LLVM and CIR
/// codegen.
static bool isMemcpyEquivalentSpecialMember(const CXXMethodDecl *D) {
  auto *CD = dyn_cast<CXXConstructorDecl>(D);
  if (!(CD && CD->isCopyOrMoveConstructor()) &&
      !D->isCopyAssignmentOperator() && !D->isMoveAssignmentOperator())
    return false;

  // We can emit a memcpy for a trivial copy or move constructor/assignment.
  if (D->isTrivial() && !D->getParent()->mayInsertExtraPadding())
    return true;

  // We *must* emit a memcpy for a defaulted union copy or move op.
  if (D->getParent()->isUnion() && D->isDefaulted())
    return true;

  return false;
}

namespace {
/// TODO(cir): a lot of what we see under this namespace is a strong candidate
/// to be shared between LLVM and CIR codegen.

/// RAII object to indicate that codegen is copying the value representation
/// instead of the object representation. Useful when copying a struct or
/// class which has uninitialized members and we're only performing
/// lvalue-to-rvalue conversion on the object but not its members.
class CopyingValueRepresentation {
public:
  explicit CopyingValueRepresentation(CIRGenFunction &CGF)
      : CGF(CGF), OldSanOpts(CGF.SanOpts) {
    CGF.SanOpts.set(SanitizerKind::Bool, false);
    CGF.SanOpts.set(SanitizerKind::Enum, false);
  }
  ~CopyingValueRepresentation() { CGF.SanOpts = OldSanOpts; }

private:
  CIRGenFunction &CGF;
  SanitizerSet OldSanOpts;
};

class FieldMemcpyizer {
public:
  FieldMemcpyizer(CIRGenFunction &CGF, const CXXMethodDecl *MethodDecl,
                  const VarDecl *SrcRec)
      : CGF(CGF), MethodDecl(MethodDecl), ClassDecl(MethodDecl->getParent()),
        SrcRec(SrcRec),
        RecLayout(CGF.getContext().getASTRecordLayout(ClassDecl)),
        FirstField(nullptr), LastField(nullptr), FirstFieldOffset(0),
        LastFieldOffset(0), LastAddedFieldIndex(0) {}

  bool isMemcpyableField(FieldDecl *F) const {
    // Never memcpy fields when we are adding poised paddings.
    if (CGF.getContext().getLangOpts().SanitizeAddressFieldPadding)
      return false;
    Qualifiers Qual = F->getType().getQualifiers();
    if (Qual.hasVolatile() || Qual.hasObjCLifetime())
      return false;
    return true;
  }

  void addMemcpyableField(FieldDecl *F) {
    assert(!cir::MissingFeatures::isEmptyFieldForLayout());
    if (F->isZeroSize(CGF.getContext()))
      return;
    if (!FirstField)
      addInitialField(F);
    else
      addNextField(F);
  }

  CharUnits getMemcpySize(uint64_t FirstByteOffset) const {
    ASTContext &astContext = CGF.getContext();
    unsigned LastFieldSize =
        LastField->isBitField()
            ? LastField->getBitWidthValue()
            : astContext.toBits(
                  astContext.getTypeInfoDataSizeInChars(LastField->getType())
                      .Width);
    uint64_t MemcpySizeBits = LastFieldOffset + LastFieldSize -
                              FirstByteOffset + astContext.getCharWidth() - 1;
    CharUnits MemcpySize = astContext.toCharUnitsFromBits(MemcpySizeBits);
    return MemcpySize;
  }

  void emitMemcpy() {
    // Give the subclass a chance to bail out if it feels the memcpy isn't worth
    // it (e.g. Hasn't aggregated enough data).
    if (!FirstField) {
      return;
    }

    uint64_t firstByteOffset;
    if (FirstField->isBitField()) {
      const CIRGenRecordLayout &rl =
          CGF.getTypes().getCIRGenRecordLayout(FirstField->getParent());
      const CIRGenBitFieldInfo &bfInfo = rl.getBitFieldInfo(FirstField);
      // FirstFieldOffset is not appropriate for bitfields,
      // we need to use the storage offset instead.
      firstByteOffset = CGF.getContext().toBits(bfInfo.StorageOffset);
    } else {
      firstByteOffset = FirstFieldOffset;
    }

    CharUnits memcpySize = getMemcpySize(firstByteOffset);
    QualType recordTy = CGF.getContext().getTypeDeclType(ClassDecl);
    Address thisPtr = CGF.LoadCXXThisAddress();
    LValue destLv = CGF.makeAddrLValue(thisPtr, recordTy);
    LValue dest = CGF.emitLValueForFieldInitialization(destLv, FirstField,
                                                       FirstField->getName());
    cir::LoadOp srcPtr = CGF.getBuilder().createLoad(
        CGF.getLoc(MethodDecl->getLocation()), CGF.GetAddrOfLocalVar(SrcRec));
    LValue srcLv = CGF.MakeNaturalAlignAddrLValue(srcPtr, recordTy);
    LValue src = CGF.emitLValueForFieldInitialization(srcLv, FirstField,
                                                      FirstField->getName());

    emitMemcpyIR(dest.isBitField() ? dest.getBitFieldAddress()
                                   : dest.getAddress(),
                 src.isBitField() ? src.getBitFieldAddress() : src.getAddress(),
                 memcpySize);
    reset();
  }

  void reset() { FirstField = nullptr; }

protected:
  CIRGenFunction &CGF;
  const CXXMethodDecl *MethodDecl;
  const CXXRecordDecl *ClassDecl;

private:
  void emitMemcpyIR(Address DestPtr, Address SrcPtr, CharUnits Size) {
    mlir::Location loc = CGF.getLoc(MethodDecl->getLocation());
    cir::ConstantOp sizeOp =
        CGF.getBuilder().getConstInt(loc, CGF.SizeTy, Size.getQuantity());
    mlir::Value dest =
        CGF.getBuilder().createBitcast(DestPtr.getPointer(), CGF.VoidPtrTy);
    mlir::Value src =
        CGF.getBuilder().createBitcast(SrcPtr.getPointer(), CGF.VoidPtrTy);
    CGF.getBuilder().createMemCpy(loc, dest, src, sizeOp);
  }

  void addInitialField(FieldDecl *F) {
    FirstField = F;
    LastField = F;
    FirstFieldOffset = RecLayout.getFieldOffset(F->getFieldIndex());
    LastFieldOffset = FirstFieldOffset;
    LastAddedFieldIndex = F->getFieldIndex();
  }

  void addNextField(FieldDecl *F) {
    // For the most part, the following invariant will hold:
    //   F->getFieldIndex() == LastAddedFieldIndex + 1
    // The one exception is that Sema won't add a copy-initializer for an
    // unnamed bitfield, which will show up here as a gap in the sequence.
    assert(F->getFieldIndex() >= LastAddedFieldIndex + 1 &&
           "Cannot aggregate fields out of order.");
    LastAddedFieldIndex = F->getFieldIndex();

    // The 'first' and 'last' fields are chosen by offset, rather than field
    // index. This allows the code to support bitfields, as well as regular
    // fields.
    uint64_t FOffset = RecLayout.getFieldOffset(F->getFieldIndex());
    if (FOffset < FirstFieldOffset) {
      FirstField = F;
      FirstFieldOffset = FOffset;
    } else if (FOffset >= LastFieldOffset) {
      LastField = F;
      LastFieldOffset = FOffset;
    }
  }

  const VarDecl *SrcRec;
  const ASTRecordLayout &RecLayout;
  FieldDecl *FirstField;
  FieldDecl *LastField;
  uint64_t FirstFieldOffset, LastFieldOffset;
  unsigned LastAddedFieldIndex;
};

static void emitLValueForAnyFieldInitialization(CIRGenFunction &CGF,
                                                CXXCtorInitializer *MemberInit,
                                                LValue &LHS) {
  FieldDecl *Field = MemberInit->getAnyMember();
  if (MemberInit->isIndirectMemberInitializer()) {
    // If we are initializing an anonymous union field, drill down to the field.
    IndirectFieldDecl *IndirectField = MemberInit->getIndirectMember();
    for (const auto *I : IndirectField->chain()) {
      auto *fd = cast<clang::FieldDecl>(I);
      LHS = CGF.emitLValueForFieldInitialization(LHS, fd, fd->getName());
    }
  } else {
    LHS = CGF.emitLValueForFieldInitialization(LHS, Field, Field->getName());
  }
}

static void emitMemberInitializer(CIRGenFunction &CGF,
                                  const CXXRecordDecl *ClassDecl,
                                  CXXCtorInitializer *MemberInit,
                                  const CXXConstructorDecl *Constructor,
                                  FunctionArgList &Args) {
  // TODO: ApplyDebugLocation
  assert(MemberInit->isAnyMemberInitializer() &&
         "Mush have member initializer!");
  assert(MemberInit->getInit() && "Must have initializer!");

  // non-static data member initializers
  FieldDecl *Field = MemberInit->getAnyMember();
  QualType FieldType = Field->getType();

  auto ThisPtr = CGF.LoadCXXThis();
  QualType RecordTy = CGF.getContext().getTypeDeclType(ClassDecl);
  LValue LHS;

  // If a base constructor is being emitted, create an LValue that has the
  // non-virtual alignment.
  if (CGF.CurGD.getCtorType() == Ctor_Base)
    LHS = CGF.MakeNaturalAlignPointeeAddrLValue(ThisPtr, RecordTy);
  else
    LHS = CGF.MakeNaturalAlignAddrLValue(ThisPtr, RecordTy);

  emitLValueForAnyFieldInitialization(CGF, MemberInit, LHS);

  // Special case: If we are in a copy or move constructor, and we are copying
  // an array off PODs or classes with tirival copy constructors, ignore the AST
  // and perform the copy we know is equivalent.
  // FIXME: This is hacky at best... if we had a bit more explicit information
  // in the AST, we could generalize it more easily.
  const ConstantArrayType *Array =
      CGF.getContext().getAsConstantArrayType(FieldType);
  if (Array && Constructor->isDefaulted() &&
      Constructor->isCopyOrMoveConstructor()) {
    QualType baseElementTy = CGF.getContext().getBaseElementType(Array);
    // NOTE(cir): CodeGen allows record types to be memcpy'd if applicable,
    // whereas ClangIR wants to represent all object construction explicitly.
    if (!baseElementTy->isRecordType()) {
      unsigned srcArgIndex =
          CGF.CGM.getCXXABI().getSrcArgforCopyCtor(Constructor, Args);
      cir::LoadOp srcPtr = CGF.getBuilder().createLoad(
          CGF.getLoc(MemberInit->getSourceLocation()),
          CGF.GetAddrOfLocalVar(Args[srcArgIndex]));
      LValue thisRhslv = CGF.MakeNaturalAlignAddrLValue(srcPtr, RecordTy);
      LValue src = CGF.emitLValueForFieldInitialization(thisRhslv, Field,
                                                        Field->getName());

      // Copy the aggregate.
      CGF.emitAggregateCopy(LHS, src, FieldType,
                            CGF.getOverlapForFieldInit(Field),
                            LHS.isVolatileQualified());
      // Ensure that we destroy the objects if an exception is thrown later in
      // the constructor.
      QualType::DestructionKind dtorKind = FieldType.isDestructedType();
      assert(!CGF.needsEHCleanup(dtorKind) &&
             "Arrays of non-record types shouldn't need EH cleanup");
      return;
    }
  }

  CGF.emitInitializerForField(Field, LHS, MemberInit->getInit());
}

class ConstructorMemcpyizer : public FieldMemcpyizer {
private:
  /// Get source argument for copy constructor. Returns null if not a copy
  /// constructor.
  static const VarDecl *getTrivialCopySource(CIRGenFunction &CGF,
                                             const CXXConstructorDecl *CD,
                                             FunctionArgList &Args) {
    if (CD->isCopyOrMoveConstructor() && CD->isDefaulted())
      return Args[CGF.CGM.getCXXABI().getSrcArgforCopyCtor(CD, Args)];

    return nullptr;
  }

  // Returns true if a CXXCtorInitializer represents a member initialization
  // that can be rolled into a memcpy.
  bool isMemberInitMemcpyable(CXXCtorInitializer *MemberInit) const {
    if (!MemcpyableCtor)
      return false;
    FieldDecl *field = MemberInit->getMember();
    assert(field && "No field for member init.");
    QualType fieldType = field->getType();
    CXXConstructExpr *ce = dyn_cast<CXXConstructExpr>(MemberInit->getInit());

    // Bail out on any members of record type (unlike CodeGen, which emits a
    // memcpy for trivially-copyable record types).
    if (ce || (fieldType->isArrayType() &&
               CGF.getContext().getBaseElementType(fieldType)->isRecordType()))
      return false;

    // Bail out on volatile fields.
    if (!isMemcpyableField(field))
      return false;

    // Otherwise we're good.
    return true;
  }

public:
  ConstructorMemcpyizer(CIRGenFunction &CGF, const CXXConstructorDecl *CD,
                        FunctionArgList &Args)
      : FieldMemcpyizer(CGF, CD, getTrivialCopySource(CGF, CD, Args)),
        ConstructorDecl(CD),
        MemcpyableCtor(CD->isDefaulted() && CD->isCopyOrMoveConstructor() &&
                       CGF.getLangOpts().getGC() == LangOptions::NonGC),
        Args(Args) {}

  void addMemberInitializer(CXXCtorInitializer *MemberInit) {
    if (isMemberInitMemcpyable(MemberInit)) {
      AggregatedInits.push_back(MemberInit);
      addMemcpyableField(MemberInit->getMember());
    } else {
      emitAggregatedInits();
      emitMemberInitializer(CGF, ConstructorDecl->getParent(), MemberInit,
                            ConstructorDecl, Args);
    }
  }

  void emitAggregatedInits() {
    if (AggregatedInits.size() <= 1) {
      // This memcpy is too small to be worthwhile. Fall back on default
      // codegen.
      if (!AggregatedInits.empty()) {
        CopyingValueRepresentation cvr(CGF);
        emitMemberInitializer(CGF, ConstructorDecl->getParent(),
                              AggregatedInits[0], ConstructorDecl, Args);
        AggregatedInits.clear();
      }
      reset();
      return;
    }

    pushEHDestructors();
    emitMemcpy();
    AggregatedInits.clear();
  }

  void pushEHDestructors() {
#ifndef NDEBUG
    for (CXXCtorInitializer *memberInit : AggregatedInits) {
      QualType fieldType = memberInit->getAnyMember()->getType();
      QualType::DestructionKind dtorKind = fieldType.isDestructedType();
      assert(!CGF.needsEHCleanup(dtorKind) &&
             "Non-record types shouldn't need EH cleanup");
    }
#endif
  }

  void finish() { emitAggregatedInits(); }

private:
  const CXXConstructorDecl *ConstructorDecl;
  bool MemcpyableCtor;
  FunctionArgList &Args;
  SmallVector<CXXCtorInitializer *, 16> AggregatedInits;
};

class AssignmentMemcpyizer : public FieldMemcpyizer {
private:
  // Returns the memcpyable field copied by the given statement, if one
  // exists. Otherwise returns null.
  FieldDecl *getMemcpyableField(Stmt *S) {
    if (!AssignmentsMemcpyable)
      return nullptr;
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(S)) {
      // Recognise trivial assignments.
      if (BO->getOpcode() != BO_Assign)
        return nullptr;
      MemberExpr *ME = dyn_cast<MemberExpr>(BO->getLHS());
      if (!ME)
        return nullptr;
      FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
      if (!Field || !isMemcpyableField(Field))
        return nullptr;
      Stmt *RHS = BO->getRHS();
      if (ImplicitCastExpr *EC = dyn_cast<ImplicitCastExpr>(RHS))
        RHS = EC->getSubExpr();
      if (!RHS)
        return nullptr;
      if (MemberExpr *ME2 = dyn_cast<MemberExpr>(RHS)) {
        if (ME2->getMemberDecl() == Field)
          return Field;
      }
      return nullptr;
    } else if (isa<CXXMemberCallExpr>(S)) {
      // We want to represent all calls explicitly for analysis purposes.
      return nullptr;
    } else if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
      // TODO(cir): https://github.com/llvm/clangir/issues/1177: This can result
      // in memcpys instead of calls to trivial member functions.
      FunctionDecl *FD = dyn_cast<FunctionDecl>(CE->getCalleeDecl());
      if (!FD || FD->getBuiltinID() != Builtin::BI__builtin_memcpy)
        return nullptr;
      Expr *DstPtr = CE->getArg(0);
      if (ImplicitCastExpr *DC = dyn_cast<ImplicitCastExpr>(DstPtr))
        DstPtr = DC->getSubExpr();
      UnaryOperator *DUO = dyn_cast<UnaryOperator>(DstPtr);
      if (!DUO || DUO->getOpcode() != UO_AddrOf)
        return nullptr;
      MemberExpr *ME = dyn_cast<MemberExpr>(DUO->getSubExpr());
      if (!ME)
        return nullptr;
      FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
      if (!Field || !isMemcpyableField(Field))
        return nullptr;
      Expr *SrcPtr = CE->getArg(1);
      if (ImplicitCastExpr *SC = dyn_cast<ImplicitCastExpr>(SrcPtr))
        SrcPtr = SC->getSubExpr();
      UnaryOperator *SUO = dyn_cast<UnaryOperator>(SrcPtr);
      if (!SUO || SUO->getOpcode() != UO_AddrOf)
        return nullptr;
      MemberExpr *ME2 = dyn_cast<MemberExpr>(SUO->getSubExpr());
      if (!ME2 || Field != dyn_cast<FieldDecl>(ME2->getMemberDecl()))
        return nullptr;
      return Field;
    }

    return nullptr;
  }

  bool AssignmentsMemcpyable;
  SmallVector<Stmt *, 16> AggregatedStmts;

public:
  AssignmentMemcpyizer(CIRGenFunction &CGF, const CXXMethodDecl *AD,
                       FunctionArgList &Args)
      : FieldMemcpyizer(CGF, AD, Args[Args.size() - 1]),
        AssignmentsMemcpyable(CGF.getLangOpts().getGC() == LangOptions::NonGC) {
    assert(Args.size() == 2);
  }

  void emitAssignment(Stmt *S) {
    FieldDecl *F = getMemcpyableField(S);
    if (F) {
      addMemcpyableField(F);
      AggregatedStmts.push_back(S);
    } else {
      emitAggregatedStmts();
      if (CGF.emitStmt(S, /*useCurrentScope=*/true).failed())
        llvm_unreachable("Should not get here!");
    }
  }

  void emitAggregatedStmts() {
    if (AggregatedStmts.size() <= 1) {
      if (!AggregatedStmts.empty()) {
        CopyingValueRepresentation CVR(CGF);
        if (CGF.emitStmt(AggregatedStmts[0], /*useCurrentScope=*/true).failed())
          llvm_unreachable("Should not get here!");
      }
      reset();
    }

    emitMemcpy();
    AggregatedStmts.clear();
  }

  void finish() { emitAggregatedStmts(); }
};
} // namespace

static bool isInitializerOfDynamicClass(const CXXCtorInitializer *BaseInit) {
  const Type *BaseType = BaseInit->getBaseClass();
  const auto *BaseClassDecl =
      cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());
  return BaseClassDecl->isDynamicClass();
}

namespace {
/// Call the destructor for a direct base class.
struct CallBaseDtor final : EHScopeStack::Cleanup {
  const CXXRecordDecl *BaseClass;
  bool BaseIsVirtual;
  CallBaseDtor(const CXXRecordDecl *Base, bool BaseIsVirtual)
      : BaseClass(Base), BaseIsVirtual(BaseIsVirtual) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    const CXXRecordDecl *DerivedClass =
        cast<CXXMethodDecl>(CGF.CurCodeDecl)->getParent();

    const CXXDestructorDecl *D = BaseClass->getDestructor();
    // We are already inside a destructor, so presumably the object being
    // destroyed should have the expected type.
    QualType ThisTy = D->getFunctionObjectParameterType();
    assert(CGF.currSrcLoc && "expected source location");
    Address Addr = CGF.getAddressOfDirectBaseInCompleteClass(
        *CGF.currSrcLoc, CGF.LoadCXXThisAddress(), DerivedClass, BaseClass,
        BaseIsVirtual);
    CGF.emitCXXDestructorCall(D, Dtor_Base, BaseIsVirtual,
                              /*Delegating=*/false, Addr, ThisTy);
  }
};

/// A visitor which checks whether an initializer uses 'this' in a
/// way which requires the vtable to be properly set.
struct DynamicThisUseChecker
    : ConstEvaluatedExprVisitor<DynamicThisUseChecker> {
  typedef ConstEvaluatedExprVisitor<DynamicThisUseChecker> super;

  bool UsesThis;

  DynamicThisUseChecker(const ASTContext &C) : super(C), UsesThis(false) {}

  // Black-list all explicit and implicit references to 'this'.
  //
  // Do we need to worry about external references to 'this' derived
  // from arbitrary code?  If so, then anything which runs arbitrary
  // external code might potentially access the vtable.
  void VisitCXXThisExpr(const CXXThisExpr *E) { UsesThis = true; }
};
} // end anonymous namespace

static bool BaseInitializerUsesThis(ASTContext &C, const Expr *Init) {
  DynamicThisUseChecker Checker(C);
  Checker.Visit(Init);
  return Checker.UsesThis;
}

/// Gets the address of a direct base class within a complete object.
/// This should only be used for (1) non-virtual bases or (2) virtual bases
/// when the type is known to be complete (e.g. in complete destructors).
///
/// The object pointed to by 'This' is assumed to be non-null.
Address CIRGenFunction::getAddressOfDirectBaseInCompleteClass(
    mlir::Location loc, Address This, const CXXRecordDecl *Derived,
    const CXXRecordDecl *Base, bool BaseIsVirtual) {
  // 'this' must be a pointer (in some address space) to Derived.
  assert(This.getElementType() == convertType(Derived));

  // Compute the offset of the virtual base.
  CharUnits Offset;
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(Derived);
  if (BaseIsVirtual)
    Offset = Layout.getVBaseClassOffset(Base);
  else
    Offset = Layout.getBaseClassOffset(Base);

  return builder.createBaseClassAddr(loc, This, convertType(Base),
                                     Offset.getQuantity(),
                                     /*assume_not_null=*/true);
}

static void emitBaseInitializer(mlir::Location loc, CIRGenFunction &CGF,
                                const CXXRecordDecl *ClassDecl,
                                CXXCtorInitializer *BaseInit) {
  assert(BaseInit->isBaseInitializer() && "Must have base initializer!");

  Address ThisPtr = CGF.LoadCXXThisAddress();

  const Type *BaseType = BaseInit->getBaseClass();
  const auto *BaseClassDecl =
      cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());

  bool isBaseVirtual = BaseInit->isBaseVirtual();

  // If the initializer for the base (other than the constructor
  // itself) accesses 'this' in any way, we need to initialize the
  // vtables.
  if (BaseInitializerUsesThis(CGF.getContext(), BaseInit->getInit()))
    CGF.initializeVTablePointers(loc, ClassDecl);

  // We can pretend to be a complete class because it only matters for
  // virtual bases, and we only do virtual bases for complete ctors.
  Address V = CGF.getAddressOfDirectBaseInCompleteClass(
      loc, ThisPtr, ClassDecl, BaseClassDecl, isBaseVirtual);
  AggValueSlot AggSlot = AggValueSlot::forAddr(
      V, Qualifiers(), AggValueSlot::IsDestructed,
      AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
      CGF.getOverlapForBaseInit(ClassDecl, BaseClassDecl, isBaseVirtual));

  CGF.emitAggExpr(BaseInit->getInit(), AggSlot);

  if (CGF.CGM.getLangOpts().Exceptions &&
      !BaseClassDecl->hasTrivialDestructor())
    CGF.EHStack.pushCleanup<CallBaseDtor>(EHCleanup, BaseClassDecl,
                                          isBaseVirtual);
}

/// This routine generates necessary code to initialize base classes and
/// non-static data members belonging to this constructor.
void CIRGenFunction::emitCtorPrologue(const CXXConstructorDecl *CD,
                                      CXXCtorType CtorType,
                                      FunctionArgList &Args) {
  if (CD->isDelegatingConstructor())
    return emitDelegatingCXXConstructorCall(CD, Args);

  const CXXRecordDecl *ClassDecl = CD->getParent();

  CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
                                          E = CD->init_end();

  // Virtual base initializers first, if any. They aren't needed if:
  // - This is a base ctor variant
  // - There are no vbases
  // - The class is abstract, so a complete object of it cannot be constructed
  //
  // The check for an abstract class is necessary because sema may not have
  // marked virtual base destructors referenced.
  bool ConstructVBases = CtorType != Ctor_Base &&
                         ClassDecl->getNumVBases() != 0 &&
                         !ClassDecl->isAbstract();

  // In the Microsoft C++ ABI, there are no constructor variants. Instead, the
  // constructor of a class with virtual bases takes an additional parameter to
  // conditionally construct the virtual bases. Emit that check here.
  mlir::Block *BaseCtorContinueBB = nullptr;
  if (ConstructVBases &&
      !CGM.getTarget().getCXXABI().hasConstructorVariants()) {
    llvm_unreachable("NYI");
  }

  auto const OldThis = CXXThisValue;
  for (; B != E && (*B)->isBaseInitializer() && (*B)->isBaseVirtual(); B++) {
    if (!ConstructVBases)
      continue;
    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        CGM.getCodeGenOpts().OptimizationLevel > 0 &&
        isInitializerOfDynamicClass(*B))
      llvm_unreachable("NYI");
    emitBaseInitializer(getLoc(CD->getBeginLoc()), *this, ClassDecl, *B);
  }

  if (BaseCtorContinueBB) {
    llvm_unreachable("NYI");
  }

  // Then, non-virtual base initializers.
  for (; B != E && (*B)->isBaseInitializer(); B++) {
    assert(!(*B)->isBaseVirtual());

    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        CGM.getCodeGenOpts().OptimizationLevel > 0 &&
        isInitializerOfDynamicClass(*B))
      llvm_unreachable("NYI");
    emitBaseInitializer(getLoc(CD->getBeginLoc()), *this, ClassDecl, *B);
  }

  CXXThisValue = OldThis;

  initializeVTablePointers(getLoc(CD->getBeginLoc()), ClassDecl);

  // And finally, initialize class members.
  FieldConstructionScope FCS(*this, LoadCXXThisAddress());
  ConstructorMemcpyizer CM(*this, CD, Args);
  for (; B != E; B++) {
    CXXCtorInitializer *Member = (*B);
    assert(!Member->isBaseInitializer());
    assert(Member->isAnyMemberInitializer() &&
           "Delegating initializer on non-delegating constructor");
    CM.addMemberInitializer(Member);
  }
  CM.finish();
}

static Address ApplyNonVirtualAndVirtualOffset(
    mlir::Location loc, CIRGenFunction &CGF, Address addr,
    CharUnits nonVirtualOffset, mlir::Value virtualOffset,
    const CXXRecordDecl *derivedClass, const CXXRecordDecl *nearestVBase,
    mlir::Type baseValueTy = {}, bool assumeNotNull = true) {
  // Assert that we have something to do.
  assert(!nonVirtualOffset.isZero() || virtualOffset != nullptr);

  // Compute the offset from the static and dynamic components.
  mlir::Value baseOffset;
  if (!nonVirtualOffset.isZero()) {
    if (virtualOffset) {
      mlir::Type OffsetType =
          (CGF.CGM.getTarget().getCXXABI().isItaniumFamily() &&
           CGF.CGM.getItaniumVTableContext().isRelativeLayout())
              ? CGF.SInt32Ty
              : CGF.PtrDiffTy;
      baseOffset = CGF.getBuilder().getConstInt(loc, OffsetType,
                                                nonVirtualOffset.getQuantity());
      baseOffset = CGF.getBuilder().createBinop(
          virtualOffset, cir::BinOpKind::Add, baseOffset);
    } else {
      assert(baseValueTy && "expected base type");
      // If no virtualOffset is present this is the final stop.
      return CGF.getBuilder().createBaseClassAddr(
          loc, addr, baseValueTy, nonVirtualOffset.getQuantity(),
          assumeNotNull);
    }
  } else {
    baseOffset = virtualOffset;
  }

  // Apply the base offset.  cir.ptr_stride adjusts by a number of elements,
  // not bytes.  So the pointer must be cast to a byte pointer and back.

  mlir::Value ptr = addr.getPointer();
  mlir::Type charPtrType = CGF.CGM.UInt8PtrTy;
  mlir::Value charPtr =
      CGF.getBuilder().createCast(cir::CastKind::bitcast, ptr, charPtrType);
  mlir::Value adjusted = CGF.getBuilder().create<cir::PtrStrideOp>(
      loc, charPtrType, charPtr, baseOffset);
  ptr = CGF.getBuilder().createCast(cir::CastKind::bitcast, adjusted,
                                    ptr.getType());

  // If we have a virtual component, the alignment of the result will
  // be relative only to the known alignment of that vbase.
  CharUnits alignment;
  if (virtualOffset) {
    assert(nearestVBase && "virtual offset without vbase?");
    alignment = CGF.CGM.getVBaseAlignment(addr.getAlignment(), derivedClass,
                                          nearestVBase);
  } else {
    alignment = addr.getAlignment();
  }
  alignment = alignment.alignmentAtOffset(nonVirtualOffset);

  return Address(ptr, alignment);
}

void CIRGenFunction::initializeVTablePointer(mlir::Location loc,
                                             const VPtr &Vptr) {
  // Compute the address point.
  auto VTableAddressPoint = CGM.getCXXABI().getVTableAddressPointInStructor(
      *this, Vptr.VTableClass, Vptr.Base, Vptr.NearestVBase);

  if (!VTableAddressPoint)
    return;

  // Compute where to store the address point.
  mlir::Value VirtualOffset{};
  CharUnits NonVirtualOffset = CharUnits::Zero();

  mlir::Type BaseValueTy;
  if (CGM.getCXXABI().isVirtualOffsetNeededForVTableField(*this, Vptr)) {
    // We need to use the virtual base offset offset because the virtual base
    // might have a different offset in the most derived class.
    VirtualOffset = CGM.getCXXABI().getVirtualBaseClassOffset(
        loc, *this, LoadCXXThisAddress(), Vptr.VTableClass, Vptr.NearestVBase);
    NonVirtualOffset = Vptr.OffsetFromNearestVBase;
  } else {
    // We can just use the base offset in the complete class.
    NonVirtualOffset = Vptr.Base.getBaseOffset();
    BaseValueTy = convertType(getContext().getTagDeclType(Vptr.Base.getBase()));
  }

  // Apply the offsets.
  Address VTableField = LoadCXXThisAddress();
  if (!NonVirtualOffset.isZero() || VirtualOffset) {
    VTableField = ApplyNonVirtualAndVirtualOffset(
        loc, *this, VTableField, NonVirtualOffset, VirtualOffset,
        Vptr.VTableClass, Vptr.NearestVBase, BaseValueTy);
  }

  // Finally, store the address point. Use the same CIR types as the field.
  //
  // vtable field is derived from `this` pointer, therefore they should be in
  // the same addr space.
  assert(!cir::MissingFeatures::addressSpace());
  VTableField = builder.createElementBitCast(loc, VTableField,
                                             VTableAddressPoint.getType());
  auto storeOp = builder.createStore(loc, VTableAddressPoint, VTableField);
  TBAAAccessInfo TBAAInfo =
      CGM.getTBAAVTablePtrAccessInfo(VTableAddressPoint.getType());
  CGM.decorateOperationWithTBAA(storeOp, TBAAInfo);
  if (CGM.getCodeGenOpts().OptimizationLevel > 0 &&
      CGM.getCodeGenOpts().StrictVTablePointers) {
    assert(!cir::MissingFeatures::createInvariantGroup());
  }
}

void CIRGenFunction::initializeVTablePointers(mlir::Location loc,
                                              const CXXRecordDecl *RD) {
  // Ignore classes without a vtable.
  if (!RD->isDynamicClass())
    return;

  // Initialize the vtable pointers for this class and all of its bases.
  if (CGM.getCXXABI().doStructorsInitializeVPtrs(RD))
    for (const auto &Vptr : getVTablePointers(RD))
      initializeVTablePointer(loc, Vptr);

  if (RD->getNumVBases())
    CGM.getCXXABI().initializeHiddenVirtualInheritanceMembers(*this, RD);
}

CIRGenFunction::VPtrsVector
CIRGenFunction::getVTablePointers(const CXXRecordDecl *VTableClass) {
  CIRGenFunction::VPtrsVector VPtrsResult;
  VisitedVirtualBasesSetTy VBases;
  getVTablePointers(BaseSubobject(VTableClass, CharUnits::Zero()),
                    /*NearestVBase=*/nullptr,
                    /*OffsetFromNearestVBase=*/CharUnits::Zero(),
                    /*BaseIsNonVirtualPrimaryBase=*/false, VTableClass, VBases,
                    VPtrsResult);
  return VPtrsResult;
}

void CIRGenFunction::getVTablePointers(BaseSubobject Base,
                                       const CXXRecordDecl *NearestVBase,
                                       CharUnits OffsetFromNearestVBase,
                                       bool BaseIsNonVirtualPrimaryBase,
                                       const CXXRecordDecl *VTableClass,
                                       VisitedVirtualBasesSetTy &VBases,
                                       VPtrsVector &Vptrs) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!BaseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    VPtr Vptr = {Base, NearestVBase, OffsetFromNearestVBase, VTableClass};
    Vptrs.push_back(Vptr);
  }

  const CXXRecordDecl *RD = Base.getBase();

  // Traverse bases.
  for (const auto &I : RD->bases()) {
    auto *BaseDecl =
        cast<CXXRecordDecl>(I.getType()->castAs<RecordType>()->getDecl());

    // Ignore classes without a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    CharUnits BaseOffset;
    CharUnits BaseOffsetFromNearestVBase;
    bool BaseDeclIsNonVirtualPrimaryBase;

    if (I.isVirtual()) {
      // Check if we've visited this virtual base before.
      if (!VBases.insert(BaseDecl).second)
        continue;

      const ASTRecordLayout &Layout =
          getContext().getASTRecordLayout(VTableClass);

      BaseOffset = Layout.getVBaseClassOffset(BaseDecl);
      BaseOffsetFromNearestVBase = CharUnits::Zero();
      BaseDeclIsNonVirtualPrimaryBase = false;
    } else {
      const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);

      BaseOffset = Base.getBaseOffset() + Layout.getBaseClassOffset(BaseDecl);
      BaseOffsetFromNearestVBase =
          OffsetFromNearestVBase + Layout.getBaseClassOffset(BaseDecl);
      BaseDeclIsNonVirtualPrimaryBase = Layout.getPrimaryBase() == BaseDecl;
    }

    getVTablePointers(
        BaseSubobject(BaseDecl, BaseOffset),
        I.isVirtual() ? BaseDecl : NearestVBase, BaseOffsetFromNearestVBase,
        BaseDeclIsNonVirtualPrimaryBase, VTableClass, VBases, Vptrs);
  }
}

Address CIRGenFunction::LoadCXXThisAddress() {
  assert(CurFuncDecl && "loading 'this' without a func declaration?");
  assert(isa<CXXMethodDecl>(CurFuncDecl));

  // Lazily compute CXXThisAlignment.
  if (CXXThisAlignment.isZero()) {
    // Just use the best known alignment for the parent.
    // TODO: if we're currently emitting a complete-object ctor/dtor, we can
    // always use the complete-object alignment.
    auto RD = cast<CXXMethodDecl>(CurFuncDecl)->getParent();
    CXXThisAlignment = CGM.getClassPointerAlignment(RD);
  }

  return Address(LoadCXXThis(), CXXThisAlignment);
}

void CIRGenFunction::emitInitializerForField(FieldDecl *Field, LValue LHS,
                                             Expr *Init) {
  QualType FieldType = Field->getType();
  switch (getEvaluationKind(FieldType)) {
  case cir::TEK_Scalar:
    if (LHS.isSimple()) {
      emitExprAsInit(Init, Field, LHS, false);
    } else {
      llvm_unreachable("NYI");
    }
    break;
  case cir::TEK_Complex:
    llvm_unreachable("NYI");
    break;
  case cir::TEK_Aggregate: {
    AggValueSlot Slot = AggValueSlot::forLValue(
        LHS, AggValueSlot::IsDestructed, AggValueSlot::DoesNotNeedGCBarriers,
        AggValueSlot::IsNotAliased, getOverlapForFieldInit(Field),
        AggValueSlot::IsNotZeroed,
        // Checks are made by the code that calls constructor.
        AggValueSlot::IsSanitizerChecked);
    emitAggExpr(Init, Slot);
    break;
  }
  }

  // Ensure that we destroy this object if an exception is thrown later in the
  // constructor.
  QualType::DestructionKind dtorKind = FieldType.isDestructedType();
  (void)dtorKind;
  if (cir::MissingFeatures::cleanups())
    llvm_unreachable("NYI");
}

void CIRGenFunction::emitDelegateCXXConstructorCall(
    const CXXConstructorDecl *Ctor, CXXCtorType CtorType,
    const FunctionArgList &Args, SourceLocation Loc) {
  CallArgList DelegateArgs;

  FunctionArgList::const_iterator I = Args.begin(), E = Args.end();
  assert(I != E && "no parameters to constructor");

  // this
  Address This = LoadCXXThisAddress();
  DelegateArgs.add(RValue::get(This.getPointer()), (*I)->getType());
  ++I;

  // FIXME: The location of the VTT parameter in the parameter list is specific
  // to the Itanium ABI and shouldn't be hardcoded here.
  if (CGM.getCXXABI().NeedsVTTParameter(CurGD)) {
    llvm_unreachable("NYI");
  }

  // Explicit arguments.
  for (; I != E; ++I) {
    const VarDecl *param = *I;
    // FIXME: per-argument source location
    emitDelegateCallArg(DelegateArgs, param, Loc);
  }

  emitCXXConstructorCall(Ctor, CtorType, /*ForVirtualBase=*/false,
                         /*Delegating=*/true, This, DelegateArgs,
                         AggValueSlot::MayOverlap, Loc,
                         /*NewPointerIsChecked=*/true);
}

void CIRGenFunction::emitImplicitAssignmentOperatorBody(FunctionArgList &Args) {
  const CXXMethodDecl *AssignOp = cast<CXXMethodDecl>(CurGD.getDecl());
  const Stmt *RootS = AssignOp->getBody();
  assert(isa<CompoundStmt>(RootS) &&
         "Body of an implicit assignment operator should be compound stmt.");
  const CompoundStmt *RootCS = cast<CompoundStmt>(RootS);

  // LexicalScope Scope(*this, RootCS->getSourceRange());
  // FIXME(cir): add all of the below under a new scope.

  assert(!cir::MissingFeatures::incrementProfileCounter());
  AssignmentMemcpyizer AM(*this, AssignOp, Args);
  for (auto *I : RootCS->body())
    AM.emitAssignment(I);
  AM.finish();
}

void CIRGenFunction::emitForwardingCallToLambda(
    const CXXMethodDecl *callOperator, CallArgList &callArgs) {
  // Get the address of the call operator.
  const auto &calleeFnInfo =
      CGM.getTypes().arrangeCXXMethodDeclaration(callOperator);
  auto calleePtr = CGM.GetAddrOfFunction(
      GlobalDecl(callOperator), CGM.getTypes().GetFunctionType(calleeFnInfo));

  // Prepare the return slot.
  const FunctionProtoType *FPT =
      callOperator->getType()->castAs<FunctionProtoType>();
  QualType resultType = FPT->getReturnType();
  ReturnValueSlot returnSlot;

  // We don't need to separately arrange the call arguments because
  // the call can't be variadic anyway --- it's impossible to forward
  // variadic arguments.

  // Now emit our call.
  auto callee = CIRGenCallee::forDirect(calleePtr, GlobalDecl(callOperator));
  RValue RV = emitCall(calleeFnInfo, callee, returnSlot, callArgs);

  // If necessary, copy the returned value into the slot.
  if (!resultType->isVoidType() && returnSlot.isNull()) {
    if (getLangOpts().ObjCAutoRefCount && resultType->isObjCRetainableType())
      llvm_unreachable("NYI");
    emitReturnOfRValue(*currSrcLoc, RV, resultType);
  } else {
    llvm_unreachable("NYI");
  }
}

void CIRGenFunction::emitLambdaDelegatingInvokeBody(const CXXMethodDecl *MD) {
  const CXXRecordDecl *Lambda = MD->getParent();

  // Start building arguments for forwarding call
  CallArgList CallArgs;

  QualType LambdaType = getContext().getRecordType(Lambda);
  QualType ThisType = getContext().getPointerType(LambdaType);
  Address ThisPtr =
      CreateMemTemp(LambdaType, getLoc(MD->getSourceRange()), "unused.capture");
  CallArgs.add(RValue::get(ThisPtr.getPointer()), ThisType);

  // Add the rest of the parameters.
  for (auto *Param : MD->parameters())
    emitDelegateCallArg(CallArgs, Param, Param->getBeginLoc());

  const CXXMethodDecl *CallOp = Lambda->getLambdaCallOperator();
  // For a generic lambda, find the corresponding call operator specialization
  // to which the call to the static-invoker shall be forwarded.
  if (Lambda->isGenericLambda()) {
    assert(MD->isFunctionTemplateSpecialization());
    const TemplateArgumentList *TAL = MD->getTemplateSpecializationArgs();
    FunctionTemplateDecl *CallOpTemplate =
        CallOp->getDescribedFunctionTemplate();
    void *InsertPos = nullptr;
    FunctionDecl *CorrespondingCallOpSpecialization =
        CallOpTemplate->findSpecialization(TAL->asArray(), InsertPos);
    assert(CorrespondingCallOpSpecialization);
    CallOp = cast<CXXMethodDecl>(CorrespondingCallOpSpecialization);
  }
  emitForwardingCallToLambda(CallOp, CallArgs);
}

void CIRGenFunction::emitLambdaStaticInvokeBody(const CXXMethodDecl *MD) {
  if (MD->isVariadic()) {
    // Codgen for LLVM doesn't emit code for this as well, it says:
    // FIXME: Making this work correctly is nasty because it requires either
    // cloning the body of the call operator or making the call operator
    // forward.
    llvm_unreachable("NYI");
  }

  emitLambdaDelegatingInvokeBody(MD);
}

void CIRGenFunction::destroyCXXObject(CIRGenFunction &CGF, Address addr,
                                      QualType type) {
  const RecordType *rtype = type->castAs<RecordType>();
  const CXXRecordDecl *record = cast<CXXRecordDecl>(rtype->getDecl());
  const CXXDestructorDecl *dtor = record->getDestructor();
  // TODO(cir): Unlike traditional codegen, CIRGen should actually emit trivial
  // dtors which shall be removed on later CIR passes. However, only remove this
  // assertion once we get a testcase to exercise this path.
  assert(!dtor->isTrivial());
  CGF.emitCXXDestructorCall(dtor, Dtor_Complete, /*for vbase*/ false,
                            /*Delegating=*/false, addr, type);
}

static bool FieldHasTrivialDestructorBody(ASTContext &astContext,
                                          const FieldDecl *Field);

// FIXME(cir): this should be shared with traditional codegen.
static bool
HasTrivialDestructorBody(ASTContext &astContext,
                         const CXXRecordDecl *BaseClassDecl,
                         const CXXRecordDecl *MostDerivedClassDecl) {
  // If the destructor is trivial we don't have to check anything else.
  if (BaseClassDecl->hasTrivialDestructor())
    return true;

  if (!BaseClassDecl->getDestructor()->hasTrivialBody())
    return false;

  // Check fields.
  for (const auto *Field : BaseClassDecl->fields())
    if (!FieldHasTrivialDestructorBody(astContext, Field))
      return false;

  // Check non-virtual bases.
  for (const auto &I : BaseClassDecl->bases()) {
    if (I.isVirtual())
      continue;

    const CXXRecordDecl *NonVirtualBase =
        cast<CXXRecordDecl>(I.getType()->castAs<RecordType>()->getDecl());
    if (!HasTrivialDestructorBody(astContext, NonVirtualBase,
                                  MostDerivedClassDecl))
      return false;
  }

  if (BaseClassDecl == MostDerivedClassDecl) {
    // Check virtual bases.
    for (const auto &I : BaseClassDecl->vbases()) {
      const CXXRecordDecl *VirtualBase =
          cast<CXXRecordDecl>(I.getType()->castAs<RecordType>()->getDecl());
      if (!HasTrivialDestructorBody(astContext, VirtualBase,
                                    MostDerivedClassDecl))
        return false;
    }
  }

  return true;
}

// FIXME(cir): this should be shared with traditional codegen.
static bool FieldHasTrivialDestructorBody(ASTContext &astContext,
                                          const FieldDecl *Field) {
  QualType FieldBaseElementType =
      astContext.getBaseElementType(Field->getType());

  const RecordType *RT = FieldBaseElementType->getAs<RecordType>();
  if (!RT)
    return true;

  CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());

  // The destructor for an implicit anonymous union member is never invoked.
  if (FieldClassDecl->isUnion() && FieldClassDecl->isAnonymousStructOrUnion())
    return false;

  return HasTrivialDestructorBody(astContext, FieldClassDecl, FieldClassDecl);
}

/// Check whether we need to initialize any vtable pointers before calling this
/// destructor.
/// FIXME(cir): this should be shared with traditional codegen.
static bool CanSkipVTablePointerInitialization(CIRGenFunction &CGF,
                                               const CXXDestructorDecl *Dtor) {
  const CXXRecordDecl *ClassDecl = Dtor->getParent();
  if (!ClassDecl->isDynamicClass())
    return true;

  // For a final class, the vtable pointer is known to already point to the
  // class's vtable.
  if (ClassDecl->isEffectivelyFinal())
    return true;

  if (!Dtor->hasTrivialBody())
    return false;

  // Check the fields.
  for (const auto *Field : ClassDecl->fields())
    if (!FieldHasTrivialDestructorBody(CGF.getContext(), Field))
      return false;

  return true;
}

/// Emits the body of the current destructor.
void CIRGenFunction::emitDestructorBody(FunctionArgList &Args) {
  const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CurGD.getDecl());
  CXXDtorType DtorType = CurGD.getDtorType();

  // For an abstract class, non-base destructors are never used (and can't
  // be emitted in general, because vbase dtors may not have been validated
  // by Sema), but the Itanium ABI doesn't make them optional and Clang may
  // in fact emit references to them from other compilations, so emit them
  // as functions containing a trap instruction.
  if (DtorType != Dtor_Base && Dtor->getParent()->isAbstract()) {
    SourceLocation Loc =
        Dtor->hasBody() ? Dtor->getBody()->getBeginLoc() : Dtor->getLocation();
    builder.create<cir::TrapOp>(getLoc(Loc));
    // The corresponding clang/CodeGen logic clears the insertion point here,
    // but MLIR's builder requires a valid insertion point, so we create a dummy
    // block (since the trap is a block terminator).
    builder.createBlock(builder.getBlock()->getParent());
    return;
  }

  Stmt *Body = Dtor->getBody();
  if (Body)
    assert(!cir::MissingFeatures::incrementProfileCounter());

  // The call to operator delete in a deleting destructor happens
  // outside of the function-try-block, which means it's always
  // possible to delegate the destructor body to the complete
  // destructor.  Do so.
  if (DtorType == Dtor_Deleting) {
    RunCleanupsScope DtorEpilogue(*this);
    EnterDtorCleanups(Dtor, Dtor_Deleting);
    if (HaveInsertPoint()) {
      QualType ThisTy = Dtor->getFunctionObjectParameterType();
      emitCXXDestructorCall(Dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                            /*Delegating=*/false, LoadCXXThisAddress(), ThisTy);
    }
    return;
  }

  // If the body is a function-try-block, enter the try before
  // anything else.
  bool isTryBody = (Body && isa<CXXTryStmt>(Body));
  if (isTryBody) {
    llvm_unreachable("NYI");
    // EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);
  }
  if (cir::MissingFeatures::emitAsanPrologueOrEpilogue())
    llvm_unreachable("NYI");

  // Enter the epilogue cleanups.
  RunCleanupsScope DtorEpilogue(*this);

  // If this is the complete variant, just invoke the base variant;
  // the epilogue will destruct the virtual bases.  But we can't do
  // this optimization if the body is a function-try-block, because
  // we'd introduce *two* handler blocks.  In the Microsoft ABI, we
  // always delegate because we might not have a definition in this TU.
  switch (DtorType) {
  case Dtor_Comdat:
    llvm_unreachable("not expecting a COMDAT");
  case Dtor_Deleting:
    llvm_unreachable("already handled deleting case");

  case Dtor_Complete:
    assert((Body || getTarget().getCXXABI().isMicrosoft()) &&
           "can't emit a dtor without a body for non-Microsoft ABIs");

    // Enter the cleanup scopes for virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Complete);

    if (!isTryBody) {
      QualType ThisTy = Dtor->getFunctionObjectParameterType();
      emitCXXDestructorCall(Dtor, Dtor_Base, /*ForVirtualBase=*/false,
                            /*Delegating=*/false, LoadCXXThisAddress(), ThisTy);
      break;
    }

    // Fallthrough: act like we're in the base variant.
    [[fallthrough]];

  case Dtor_Base:
    assert(Body);

    // Enter the cleanup scopes for fields and non-virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Base);

    // Initialize the vtable pointers before entering the body.
    if (!CanSkipVTablePointerInitialization(*this, Dtor)) {
      // Insert the llvm.launder.invariant.group intrinsic before initializing
      // the vptrs to cancel any previous assumptions we might have made.
      if (CGM.getCodeGenOpts().StrictVTablePointers &&
          CGM.getCodeGenOpts().OptimizationLevel > 0)
        llvm_unreachable("NYI");
      initializeVTablePointers(getLoc(Dtor->getSourceRange()),
                               Dtor->getParent());
    }

    if (isTryBody)
      llvm_unreachable("NYI");
    else if (Body)
      (void)emitStmt(Body, /*useCurrentScope=*/true);
    else {
      assert(Dtor->isImplicit() && "bodyless dtor not implicit");
      // nothing to do besides what's in the epilogue
    }
    // -fapple-kext must inline any call to this dtor into
    // the caller's body.
    if (getLangOpts().AppleKext)
      llvm_unreachable("NYI");

    break;
  }

  // Jump out through the epilogue cleanups.
  DtorEpilogue.ForceCleanup();

  // Exit the try if applicable.
  if (isTryBody)
    llvm_unreachable("NYI");
}

namespace {
[[maybe_unused]] mlir::Value
LoadThisForDtorDelete(CIRGenFunction &CGF, const CXXDestructorDecl *DD) {
  if (Expr *ThisArg = DD->getOperatorDeleteThisArg())
    return CGF.emitScalarExpr(ThisArg);
  return CGF.LoadCXXThis();
}

/// Call the operator delete associated with the current destructor.
struct CallDtorDelete final : EHScopeStack::Cleanup {
  CallDtorDelete() {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CGF.CurCodeDecl);
    const CXXRecordDecl *ClassDecl = Dtor->getParent();
    CGF.emitDeleteCall(Dtor->getOperatorDelete(),
                       LoadThisForDtorDelete(CGF, Dtor),
                       CGF.getContext().getTagDeclType(ClassDecl));
  }
};
} // namespace

class DestroyField final : public EHScopeStack::Cleanup {
  const FieldDecl *field;
  CIRGenFunction::Destroyer *destroyer;
  bool useEHCleanupForArray;

public:
  DestroyField(const FieldDecl *field, CIRGenFunction::Destroyer *destroyer,
               bool useEHCleanupForArray)
      : field(field), destroyer(destroyer),
        useEHCleanupForArray(useEHCleanupForArray) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    // Find the address of the field.
    Address thisValue = CGF.LoadCXXThisAddress();
    QualType RecordTy = CGF.getContext().getTagDeclType(field->getParent());
    LValue ThisLV = CGF.makeAddrLValue(thisValue, RecordTy);
    LValue LV = CGF.emitLValueForField(ThisLV, field);
    assert(LV.isSimple());

    CGF.emitDestroy(LV.getAddress(), field->getType(), destroyer,
                    flags.isForNormalCleanup() && useEHCleanupForArray);
  }
};

/// Emit all code that comes at the end of class's destructor. This is to call
/// destructors on members and base classes in reverse order of their
/// construction.
///
/// For a deleting destructor, this also handles the case where a destroying
/// operator delete completely overrides the definition.
void CIRGenFunction::EnterDtorCleanups(const CXXDestructorDecl *DD,
                                       CXXDtorType DtorType) {
  assert((!DD->isTrivial() || DD->hasAttr<DLLExportAttr>()) &&
         "Should not emit dtor epilogue for non-exported trivial dtor!");

  // The deleting-destructor phase just needs to call the appropriate
  // operator delete that Sema picked up.
  if (DtorType == Dtor_Deleting) {
    assert(DD->getOperatorDelete() &&
           "operator delete missing - EnterDtorCleanups");
    if (CXXStructorImplicitParamValue) {
      llvm_unreachable("NYI");
    } else {
      if (DD->getOperatorDelete()->isDestroyingOperatorDelete()) {
        llvm_unreachable("NYI");
      } else {
        EHStack.pushCleanup<CallDtorDelete>(NormalAndEHCleanup);
      }
    }
    return;
  }

  const CXXRecordDecl *ClassDecl = DD->getParent();

  // Unions have no bases and do not call field destructors.
  if (ClassDecl->isUnion())
    return;

  // The complete-destructor phase just destructs all the virtual bases.
  if (DtorType == Dtor_Complete) {
    // Poison the vtable pointer such that access after the base
    // and member destructors are invoked is invalid.
    if (CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
        SanOpts.has(SanitizerKind::Memory) && ClassDecl->getNumVBases() &&
        ClassDecl->isPolymorphic())
      assert(!cir::MissingFeatures::sanitizeDtor());

    // We push them in the forward order so that they'll be popped in
    // the reverse order.
    for (const auto &Base : ClassDecl->vbases()) {
      auto *BaseClassDecl =
          cast<CXXRecordDecl>(Base.getType()->castAs<RecordType>()->getDecl());

      if (BaseClassDecl->hasTrivialDestructor()) {
        // Under SanitizeMemoryUseAfterDtor, poison the trivial base class
        // memory. For non-trival base classes the same is done in the class
        // destructor.
        assert(!cir::MissingFeatures::sanitizeDtor());
      } else {
        EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, BaseClassDecl,
                                          /*BaseIsVirtual*/ true);
      }
    }

    return;
  }

  assert(DtorType == Dtor_Base);
  // Poison the vtable pointer if it has no virtual bases, but inherits
  // virtual functions.
  if (CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
      SanOpts.has(SanitizerKind::Memory) && !ClassDecl->getNumVBases() &&
      ClassDecl->isPolymorphic())
    assert(!cir::MissingFeatures::sanitizeDtor());

  // Destroy non-virtual bases.
  for (const auto &Base : ClassDecl->bases()) {
    // Ignore virtual bases.
    if (Base.isVirtual())
      continue;

    CXXRecordDecl *BaseClassDecl = Base.getType()->getAsCXXRecordDecl();

    if (BaseClassDecl->hasTrivialDestructor()) {
      if (CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
          SanOpts.has(SanitizerKind::Memory) && !BaseClassDecl->isEmpty())
        assert(!cir::MissingFeatures::sanitizeDtor());
    } else {
      EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, BaseClassDecl,
                                        /*BaseIsVirtual*/ false);
    }
  }

  // Poison fields such that access after their destructors are
  // invoked, and before the base class destructor runs, is invalid.
  bool SanitizeFields = CGM.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
                        SanOpts.has(SanitizerKind::Memory);
  assert(!cir::MissingFeatures::sanitizeDtor());

  // Destroy direct fields.
  for (const auto *Field : ClassDecl->fields()) {
    if (SanitizeFields)
      assert(!cir::MissingFeatures::sanitizeDtor());

    QualType type = Field->getType();
    QualType::DestructionKind dtorKind = type.isDestructedType();
    if (!dtorKind)
      continue;

    // Anonymous union members do not have their destructors called.
    const RecordType *RT = type->getAsUnionType();
    if (RT && RT->getDecl()->isAnonymousStructOrUnion())
      continue;

    CleanupKind cleanupKind = getCleanupKind(dtorKind);
    EHStack.pushCleanup<DestroyField>(
        cleanupKind, Field, getDestroyer(dtorKind), cleanupKind & EHCleanup);
  }

  if (SanitizeFields)
    assert(!cir::MissingFeatures::sanitizeDtor());
}

namespace {
struct CallDelegatingCtorDtor final : EHScopeStack::Cleanup {
  const CXXDestructorDecl *Dtor;
  Address Addr;
  CXXDtorType Type;

  CallDelegatingCtorDtor(const CXXDestructorDecl *D, Address Addr,
                         CXXDtorType Type)
      : Dtor(D), Addr(Addr), Type(Type) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    // We are calling the destructor from within the constructor.
    // Therefore, "this" should have the expected type.
    QualType ThisTy = Dtor->getFunctionObjectParameterType();
    CGF.emitCXXDestructorCall(Dtor, Type, /*ForVirtualBase=*/false,
                              /*Delegating=*/true, Addr, ThisTy);
  }
};
} // end anonymous namespace

void CIRGenFunction::emitDelegatingCXXConstructorCall(
    const CXXConstructorDecl *Ctor, const FunctionArgList &Args) {
  assert(Ctor->isDelegatingConstructor());

  Address ThisPtr = LoadCXXThisAddress();

  AggValueSlot AggSlot = AggValueSlot::forAddr(
      ThisPtr, Qualifiers(), AggValueSlot::IsDestructed,
      AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
      AggValueSlot::MayOverlap, AggValueSlot::IsNotZeroed,
      // Checks are made by the code that calls constructor.
      AggValueSlot::IsSanitizerChecked);

  emitAggExpr(Ctor->init_begin()[0]->getInit(), AggSlot);

  const CXXRecordDecl *ClassDecl = Ctor->getParent();
  if (CGM.getLangOpts().Exceptions && !ClassDecl->hasTrivialDestructor()) {
    CXXDtorType Type =
        CurGD.getCtorType() == Ctor_Complete ? Dtor_Complete : Dtor_Base;

    EHStack.pushCleanup<CallDelegatingCtorDtor>(
        EHCleanup, ClassDecl->getDestructor(), ThisPtr, Type);
  }
}

void CIRGenFunction::emitCXXDestructorCall(const CXXDestructorDecl *DD,
                                           CXXDtorType Type,
                                           bool ForVirtualBase, bool Delegating,
                                           Address This, QualType ThisTy) {
  CGM.getCXXABI().emitDestructorCall(*this, DD, Type, ForVirtualBase,
                                     Delegating, This, ThisTy);
}

mlir::Value CIRGenFunction::GetVTTParameter(GlobalDecl GD, bool ForVirtualBase,
                                            bool Delegating) {
  if (!CGM.getCXXABI().NeedsVTTParameter(GD)) {
    // This constructor/destructor does not need a VTT parameter.
    return nullptr;
  }

  const CXXRecordDecl *RD = cast<CXXMethodDecl>(CurCodeDecl)->getParent();
  const CXXRecordDecl *Base = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  uint64_t SubVTTIndex;

  if (Delegating) {
    llvm_unreachable("NYI");
  } else if (RD == Base) {
    // If the record matches the base, this is the complete ctor/dtor
    // variant calling the base variant in a class with virtual bases.
    assert(!CGM.getCXXABI().NeedsVTTParameter(CurGD) &&
           "doing no-op VTT offset in base dtor/ctor?");
    assert(!ForVirtualBase && "Can't have same class as virtual base!");
    SubVTTIndex = 0;
  } else {
    const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);
    CharUnits BaseOffset = ForVirtualBase ? Layout.getVBaseClassOffset(Base)
                                          : Layout.getBaseClassOffset(Base);

    SubVTTIndex =
        CGM.getVTables().getSubVTTIndex(RD, BaseSubobject(Base, BaseOffset));
    assert(SubVTTIndex != 0 && "Sub-VTT index must be greater than zero!");
  }

  auto Loc = CGM.getLoc(RD->getBeginLoc());
  if (CGM.getCXXABI().NeedsVTTParameter(CurGD)) {
    // A VTT parameter was passed to the constructor, use it.
    auto VTT = LoadCXXVTT();
    return CGM.getBuilder().createVTTAddrPoint(Loc, VTT.getType(), VTT,
                                               SubVTTIndex);
  } else {
    // We're the complete constructor, so get the VTT by name.
    auto VTT = CGM.getVTables().getAddrOfVTT(RD);
    return CGM.getBuilder().createVTTAddrPoint(
        Loc, CGM.getBuilder().getPointerTo(CGM.VoidPtrTy),
        mlir::FlatSymbolRefAttr::get(VTT.getSymNameAttr()), SubVTTIndex);
  }
}

CharUnits CIRGenModule::getNonVirtualBaseClassOffset(
    const CXXRecordDecl *classDecl, CastExpr::path_const_iterator pathBegin,
    CastExpr::path_const_iterator pathEnd) {
  assert(pathBegin != pathEnd && "Base path should not be empty!");

  CharUnits Offset =
      computeNonVirtualBaseClassOffset(classDecl, pathBegin, pathEnd);
  return Offset;
}

Address CIRGenFunction::getAddressOfDerivedClass(
    Address baseAddr, const CXXRecordDecl *derived,
    CastExpr::path_const_iterator pathBegin,
    CastExpr::path_const_iterator pathEnd, bool nullCheckValue) {
  assert(pathBegin != pathEnd && "Base path should not be empty!");

  QualType derivedTy =
      getContext().getCanonicalType(getContext().getTagDeclType(derived));
  mlir::Type derivedValueTy = convertType(derivedTy);
  CharUnits nonVirtualOffset =
      CGM.getNonVirtualBaseClassOffset(derived, pathBegin, pathEnd);

  // Note that in OG, no offset (nonVirtualOffset.getQuantity() == 0) means it
  // just gives the address back. In CIR a `cir.derived_class` is created and
  // made into a nop later on during lowering.
  return builder.createDerivedClassAddr(getLoc(derived->getSourceRange()),
                                        baseAddr, derivedValueTy,
                                        nonVirtualOffset.getQuantity(),
                                        /*assumeNotNull=*/not nullCheckValue);
}

Address
CIRGenFunction::getAddressOfBaseClass(Address Value,
                                      const CXXRecordDecl *Derived,
                                      CastExpr::path_const_iterator PathBegin,
                                      CastExpr::path_const_iterator PathEnd,
                                      bool NullCheckValue, SourceLocation Loc) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  CastExpr::path_const_iterator Start = PathBegin;
  const CXXRecordDecl *VBase = nullptr;

  // Sema has done some convenient canonicalization here: if the
  // access path involved any virtual steps, the conversion path will
  // *start* with a step down to the correct virtual base subobject,
  // and hence will not require any further steps.
  if ((*Start)->isVirtual()) {
    VBase = cast<CXXRecordDecl>(
        (*Start)->getType()->castAs<RecordType>()->getDecl());
    ++Start;
  }

  // Compute the static offset of the ultimate destination within its
  // allocating subobject (the virtual base, if there is one, or else
  // the "complete" object that we see).
  CharUnits NonVirtualOffset = CGM.computeNonVirtualBaseClassOffset(
      VBase ? VBase : Derived, Start, PathEnd);

  // If there's a virtual step, we can sometimes "devirtualize" it.
  // For now, that's limited to when the derived type is final.
  // TODO: "devirtualize" this for accesses to known-complete objects.
  if (VBase && Derived->hasAttr<FinalAttr>()) {
    const ASTRecordLayout &layout = getContext().getASTRecordLayout(Derived);
    CharUnits vBaseOffset = layout.getVBaseClassOffset(VBase);
    NonVirtualOffset += vBaseOffset;
    VBase = nullptr; // we no longer have a virtual step
  }

  // Get the base pointer type.
  auto BaseValueTy = convertType((PathEnd[-1])->getType());
  assert(!cir::MissingFeatures::addressSpace());

  // If there is no virtual base, use cir.base_class_addr.  It takes care of
  // the adjustment and the null pointer check.
  if (NonVirtualOffset.isZero() && !VBase) {
    if (sanitizePerformTypeCheck()) {
      llvm_unreachable("NYI: sanitizePerformTypeCheck");
    }
    return builder.createBaseClassAddr(getLoc(Loc), Value, BaseValueTy, 0,
                                       /*assumeNotNull=*/true);
  }

  if (sanitizePerformTypeCheck()) {
    assert(!cir::MissingFeatures::sanitizeOther());
  }

  // Compute the virtual offset.
  mlir::Value VirtualOffset = nullptr;
  if (VBase) {
    VirtualOffset = CGM.getCXXABI().getVirtualBaseClassOffset(
        getLoc(Loc), *this, Value, Derived, VBase);
  }

  // Apply both offsets.
  Value = ApplyNonVirtualAndVirtualOffset(
      getLoc(Loc), *this, Value, NonVirtualOffset, VirtualOffset, Derived,
      VBase, BaseValueTy, not NullCheckValue);

  // Cast to the destination type.
  Value = Value.withElementType(builder, BaseValueTy);

  return Value;
}

// TODO(cir): this can be shared with LLVM codegen.
bool CIRGenFunction::shouldEmitVTableTypeCheckedLoad(const CXXRecordDecl *RD) {
  if (!CGM.getCodeGenOpts().WholeProgramVTables ||
      !CGM.HasHiddenLTOVisibility(RD))
    return false;

  if (CGM.getCodeGenOpts().VirtualFunctionElimination)
    return true;

  if (!SanOpts.has(SanitizerKind::CFIVCall) ||
      !CGM.getCodeGenOpts().SanitizeTrap.has(SanitizerKind::CFIVCall))
    return false;

  std::string TypeName = RD->getQualifiedNameAsString();
  return !getContext().getNoSanitizeList().containsType(SanitizerKind::CFIVCall,
                                                        TypeName);
}

void CIRGenFunction::emitTypeMetadataCodeForVCall(const CXXRecordDecl *RD,
                                                  mlir::Value VTable,
                                                  SourceLocation Loc) {
  if (SanOpts.has(SanitizerKind::CFIVCall)) {
    llvm_unreachable("NYI");
  } else if (CGM.getCodeGenOpts().WholeProgramVTables &&
             // Don't insert type test assumes if we are forcing public
             // visibility.
             !CGM.AlwaysHasLTOVisibilityPublic(RD)) {
    llvm_unreachable("NYI");
  }
}

mlir::Value CIRGenFunction::getVTablePtr(mlir::Location Loc, Address This,
                                         mlir::Type VTableTy,
                                         const CXXRecordDecl *RD) {
  Address VTablePtrSrc = builder.createElementBitCast(Loc, This, VTableTy);
  auto VTable = builder.createLoad(Loc, VTablePtrSrc);
  assert(!cir::MissingFeatures::tbaa());

  if (CGM.getCodeGenOpts().OptimizationLevel > 0 &&
      CGM.getCodeGenOpts().StrictVTablePointers) {
    assert(!cir::MissingFeatures::createInvariantGroup());
  }

  return VTable;
}

Address CIRGenFunction::emitCXXMemberDataPointerAddress(
    const Expr *E, Address base, mlir::Value memberPtr,
    const MemberPointerType *memberPtrType, LValueBaseInfo *baseInfo,
    TBAAAccessInfo *tbaaInfo) {
  assert(!cir::MissingFeatures::cxxABI());

  auto op = builder.createGetIndirectMember(getLoc(E->getSourceRange()),
                                            base.getPointer(), memberPtr);

  QualType memberType = memberPtrType->getPointeeType();
  CharUnits memberAlign =
      CGM.getNaturalTypeAlignment(memberType, baseInfo, tbaaInfo);
  memberAlign = CGM.getDynamicOffsetAlignment(
      base.getAlignment(), memberPtrType->getClass()->getAsCXXRecordDecl(),
      memberAlign);

  return Address(op, convertTypeForMem(memberPtrType->getPointeeType()),
                 memberAlign);
}

clang::CharUnits
CIRGenModule::getDynamicOffsetAlignment(clang::CharUnits actualBaseAlign,
                                        const clang::CXXRecordDecl *baseDecl,
                                        clang::CharUnits expectedTargetAlign) {
  // If the base is an incomplete type (which is, alas, possible with
  // member pointers), be pessimistic.
  if (!baseDecl->isCompleteDefinition())
    return std::min(actualBaseAlign, expectedTargetAlign);

  auto &baseLayout = getASTContext().getASTRecordLayout(baseDecl);
  CharUnits expectedBaseAlign = baseLayout.getNonVirtualAlignment();

  // If the class is properly aligned, assume the target offset is, too.
  //
  // This actually isn't necessarily the right thing to do --- if the
  // class is a complete object, but it's only properly aligned for a
  // base subobject, then the alignments of things relative to it are
  // probably off as well.  (Note that this requires the alignment of
  // the target to be greater than the NV alignment of the derived
  // class.)
  //
  // However, our approach to this kind of under-alignment can only
  // ever be best effort; after all, we're never going to propagate
  // alignments through variables or parameters.  Note, in particular,
  // that constructing a polymorphic type in an address that's less
  // than pointer-aligned will generally trap in the constructor,
  // unless we someday add some sort of attribute to change the
  // assumed alignment of 'this'.  So our goal here is pretty much
  // just to allow the user to explicitly say that a pointer is
  // under-aligned and then safely access its fields and vtables.
  if (actualBaseAlign >= expectedBaseAlign) {
    return expectedTargetAlign;
  }

  // Otherwise, we might be offset by an arbitrary multiple of the
  // actual alignment.  The correct adjustment is to take the min of
  // the two alignments.
  return std::min(actualBaseAlign, expectedTargetAlign);
}

/// Return the best known alignment for a pointer to a virtual base,
/// given the alignment of a pointer to the derived class.
clang::CharUnits
CIRGenModule::getVBaseAlignment(CharUnits actualDerivedAlign,
                                const CXXRecordDecl *derivedClass,
                                const CXXRecordDecl *vbaseClass) {
  // The basic idea here is that an underaligned derived pointer might
  // indicate an underaligned base pointer.

  assert(vbaseClass->isCompleteDefinition());
  auto &baseLayout = getASTContext().getASTRecordLayout(vbaseClass);
  CharUnits expectedVBaseAlign = baseLayout.getNonVirtualAlignment();

  return getDynamicOffsetAlignment(actualDerivedAlign, derivedClass,
                                   expectedVBaseAlign);
}

/// Emit a loop to call a particular constructor for each of several members
/// of an array.
///
/// \param ctor the constructor to call for each element
/// \param arrayType the type of the array to initialize
/// \param arrayBegin an arrayType*
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CIRGenFunction::emitCXXAggrConstructorCall(
    const CXXConstructorDecl *ctor, const clang::ArrayType *arrayType,
    Address arrayBegin, const CXXConstructExpr *E, bool NewPointerIsChecked,
    bool zeroInitialize) {
  QualType elementType;
  auto numElements = emitArrayLength(arrayType, elementType, arrayBegin);
  emitCXXAggrConstructorCall(ctor, numElements, arrayBegin, E,
                             NewPointerIsChecked, zeroInitialize);
}

/// Emit a loop to call a particular constructor for each of several members
/// of an array.
///
/// \param ctor the constructor to call for each element
/// \param numElements the number of elements in the array;
///   may be zero
/// \param arrayBase a T*, where T is the type constructed by ctor
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CIRGenFunction::emitCXXAggrConstructorCall(
    const CXXConstructorDecl *ctor, mlir::Value numElements, Address arrayBase,
    const CXXConstructExpr *E, bool NewPointerIsChecked, bool zeroInitialize) {
  // It's legal for numElements to be zero.  This can happen both
  // dynamically, because x can be zero in 'new A[x]', and statically,
  // because of GCC extensions that permit zero-length arrays.  There
  // are probably legitimate places where we could assume that this
  // doesn't happen, but it's not clear that it's worth it.
  // llvm::BranchInst *zeroCheckBranch = nullptr;

  // Optimize for a constant count.
  if (auto constantCount = numElements.getDefiningOp<cir::ConstantOp>()) {
    if (auto constIntAttr = constantCount.getValueAttr<cir::IntAttr>()) {
      // Just skip out if the constant count is zero.
      if (constIntAttr.getUInt() == 0)
        return;
      // Otherwise, emit the check.
    }

    if (constantCount.use_empty())
      constantCount.erase();
  } else {
    llvm_unreachable("NYI");
  }

  auto arrayTy = mlir::dyn_cast<cir::ArrayType>(arrayBase.getElementType());
  assert(arrayTy && "expected array type");
  auto elementType = arrayTy.getElementType();
  auto ptrToElmType = builder.getPointerTo(elementType);

  // Tradional LLVM codegen emits a loop here.
  // TODO(cir): Lower to a loop as part of LoweringPrepare.

  // The alignment of the base, adjusted by the size of a single element,
  // provides a conservative estimate of the alignment of every element.
  // (This assumes we never start tracking offsetted alignments.)
  //
  // Note that these are complete objects and so we don't need to
  // use the non-virtual size or alignment.
  QualType type = getContext().getTypeDeclType(ctor->getParent());
  CharUnits eltAlignment = arrayBase.getAlignment().alignmentOfArrayElement(
      getContext().getTypeSizeInChars(type));

  // Zero initialize the storage, if requested.
  if (zeroInitialize) {
    llvm_unreachable("NYI");
  }

  // C++ [class.temporary]p4:
  // There are two contexts in which temporaries are destroyed at a different
  // point than the end of the full-expression. The first context is when a
  // default constructor is called to initialize an element of an array.
  // If the constructor has one or more default arguments, the destruction of
  // every temporary created in a default argument expression is sequenced
  // before the construction of the next array element, if any.
  {
    RunCleanupsScope Scope(*this);

    // Evaluate the constructor and its arguments in a regular
    // partial-destroy cleanup.
    if (getLangOpts().Exceptions &&
        !ctor->getParent()->hasTrivialDestructor()) {
      llvm_unreachable("NYI");
    }

    // Emit the constructor call that will execute for every array element.
    auto arrayOp = builder.createPtrBitcast(arrayBase.getPointer(), arrayTy);
    builder.create<cir::ArrayCtor>(
        *currSrcLoc, arrayOp, [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto arg = b.getInsertionBlock()->addArgument(ptrToElmType, loc);
          Address curAddr = Address(arg, elementType, eltAlignment);
          auto currAVS = AggValueSlot::forAddr(
              curAddr, type.getQualifiers(), AggValueSlot::IsDestructed,
              AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
              AggValueSlot::DoesNotOverlap, AggValueSlot::IsNotZeroed,
              NewPointerIsChecked ? AggValueSlot::IsSanitizerChecked
                                  : AggValueSlot::IsNotSanitizerChecked);
          emitCXXConstructorCall(ctor, Ctor_Complete,
                                 /*ForVirtualBase=*/false,
                                 /*Delegating=*/false, currAVS, E);
          builder.create<cir::YieldOp>(loc);
        });
  }
}

static bool canEmitDelegateCallArgs(CIRGenFunction &CGF,
                                    const CXXConstructorDecl *Ctor,
                                    CXXCtorType Type, CallArgList &Args) {
  // We can't forward a variadic call.
  if (Ctor->isVariadic())
    return false;

  if (CGF.getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
    // If the parameters are callee-cleanup, it's not safe to forward.
    for (auto *P : Ctor->parameters())
      if (P->needsDestruction(CGF.getContext()))
        return false;

    // Likewise if they're inalloca.
    const CIRGenFunctionInfo &Info =
        CGF.CGM.getTypes().arrangeCXXConstructorCall(Args, Ctor, Type, 0, 0);
    if (Info.usesInAlloca())
      return false;
  }

  // Anything else should be OK.
  return true;
}

void CIRGenFunction::emitCXXConstructorCall(const clang::CXXConstructorDecl *D,
                                            clang::CXXCtorType Type,
                                            bool ForVirtualBase,
                                            bool Delegating,
                                            AggValueSlot ThisAVS,
                                            const clang::CXXConstructExpr *E) {
  CallArgList Args;
  Address This = ThisAVS.getAddress();
  LangAS SlotAS = ThisAVS.getQualifiers().getAddressSpace();
  QualType ThisType = D->getThisType();
  LangAS ThisAS = ThisType.getTypePtr()->getPointeeType().getAddressSpace();
  mlir::Value ThisPtr = This.getPointer();

  assert(SlotAS == ThisAS && "This edge case NYI");

  Args.add(RValue::get(ThisPtr), D->getThisType());

  // In LLVM Codegen: If this is a trivial constructor, just emit what's needed.
  // If this is a union copy constructor, we must emit a memcpy, because the AST
  // does not model that copy.
  if (isMemcpyEquivalentSpecialMember(D)) {
    assert(!cir::MissingFeatures::isMemcpyEquivalentSpecialMember());
  }

  const FunctionProtoType *FPT = D->getType()->castAs<FunctionProtoType>();
  EvaluationOrder Order = E->isListInitialization()
                              ? EvaluationOrder::ForceLeftToRight
                              : EvaluationOrder::Default;

  emitCallArgs(Args, FPT, E->arguments(), E->getConstructor(),
               /*ParamsToSkip*/ 0, Order);

  emitCXXConstructorCall(D, Type, ForVirtualBase, Delegating, This, Args,
                         ThisAVS.mayOverlap(), E->getExprLoc(),
                         ThisAVS.isSanitizerChecked());
}

void CIRGenFunction::emitCXXConstructorCall(
    const CXXConstructorDecl *D, CXXCtorType Type, bool ForVirtualBase,
    bool Delegating, Address This, CallArgList &Args,
    AggValueSlot::Overlap_t Overlap, SourceLocation Loc,
    bool NewPointerIsChecked) {

  const auto *ClassDecl = D->getParent();

  if (!NewPointerIsChecked)
    emitTypeCheck(CIRGenFunction::TCK_ConstructorCall, Loc, This.getPointer(),
                  getContext().getRecordType(ClassDecl), CharUnits::Zero());

  // If this is a call to a trivial default constructor:
  // In LLVM: do nothing.
  // In CIR: emit as a regular call, other later passes should lower the
  // ctor call into trivial initialization.
  assert(!cir::MissingFeatures::isTrivialCtorOrDtor());

  if (isMemcpyEquivalentSpecialMember(D)) {
    assert(!cir::MissingFeatures::isMemcpyEquivalentSpecialMember());
  }

  bool PassPrototypeArgs = true;

  // Check whether we can actually emit the constructor before trying to do so.
  if (auto Inherited = D->getInheritedConstructor()) {
    PassPrototypeArgs = getTypes().inheritingCtorHasParams(Inherited, Type);
    if (PassPrototypeArgs && !canEmitDelegateCallArgs(*this, D, Type, Args)) {
      llvm_unreachable("NYI");
      return;
    }
  }

  // Insert any ABI-specific implicit constructor arguments.
  CIRGenCXXABI::AddedStructorArgCounts ExtraArgs =
      CGM.getCXXABI().addImplicitConstructorArgs(*this, D, Type, ForVirtualBase,
                                                 Delegating, Args);

  // Emit the call.
  auto CalleePtr = CGM.getAddrOfCXXStructor(GlobalDecl(D, Type));
  const CIRGenFunctionInfo &Info = CGM.getTypes().arrangeCXXConstructorCall(
      Args, D, Type, ExtraArgs.Prefix, ExtraArgs.Suffix, PassPrototypeArgs);
  CIRGenCallee Callee = CIRGenCallee::forDirect(CalleePtr, GlobalDecl(D, Type));
  cir::CIRCallOpInterface C;
  emitCall(Info, Callee, ReturnValueSlot(), Args, &C, false, getLoc(Loc));

  assert(CGM.getCodeGenOpts().OptimizationLevel == 0 ||
         ClassDecl->isDynamicClass() || Type == Ctor_Base ||
         !CGM.getCodeGenOpts().StrictVTablePointers &&
             "vtable assumption loads NYI");
}

void CIRGenFunction::emitInheritedCXXConstructorCall(
    const CXXConstructorDecl *D, bool ForVirtualBase, Address This,
    bool InheritedFromVBase, const CXXInheritedCtorInitExpr *E) {
  CallArgList Args;
  CallArg ThisArg(RValue::get(getAsNaturalPointerTo(
                      This, D->getThisType()->getPointeeType())),
                  D->getThisType());

  // Forward the parameters.
  if (InheritedFromVBase &&
      CGM.getTarget().getCXXABI().hasConstructorVariants()) {
    llvm_unreachable("NYI");
  } else if (!CXXInheritedCtorInitExprArgs.empty()) {
    // The inheriting constructor was inlined; just inject its arguments.
    llvm_unreachable("NYI");
  } else {
    // The inheriting constructor was not inlined. Emit delegating arguments.
    Args.push_back(ThisArg);
    const auto *OuterCtor = cast<CXXConstructorDecl>(CurCodeDecl);
    assert(OuterCtor->getNumParams() == D->getNumParams());
    assert(!OuterCtor->isVariadic() && "should have been inlined");
    for (const auto *Param : OuterCtor->parameters()) {
      assert(getContext().hasSameUnqualifiedType(
          OuterCtor->getParamDecl(Param->getFunctionScopeIndex())->getType(),
          Param->getType()));
      emitDelegateCallArg(Args, Param, E->getLocation());

      // Forward __attribute__(pass_object_size).
      if (Param->hasAttr<clang::PassObjectSizeAttr>()) {
        auto *POSParam = SizeArguments[Param];
        assert(POSParam && "missing pass_object_size value for forwarding");
        emitDelegateCallArg(Args, POSParam, E->getLocation());
      }
    }
  }

  emitCXXConstructorCall(D, Ctor_Base, ForVirtualBase, /*Delegating*/ false,
                         This, Args, AggValueSlot::MayOverlap, E->getLocation(),
                         /*NewPointerIsChecked*/ true);
}

void CIRGenFunction::emitInlinedInheritingCXXConstructorCall(
    const CXXConstructorDecl *Ctor, CXXCtorType CtorType, bool ForVirtualBase,
    bool Delegating, CallArgList &Args) {
  GlobalDecl GD(Ctor, CtorType);
  llvm_unreachable("NYI");
  InlinedInheritingConstructorScope Scope(*this, GD);
  // TODO(cir): ApplyInlineDebugLocation
  assert(!cir::MissingFeatures::generateDebugInfo());
  RunCleanupsScope RunCleanups(*this);

  // Save the arguments to be passed to the inherited constructor.
  CXXInheritedCtorInitExprArgs = Args;

  FunctionArgList Params;
  QualType RetType = buildFunctionArgList(CurGD, Params);
  FnRetTy = RetType;

  // Insert any ABI-specific implicit constructor arguments.
  CGM.getCXXABI().addImplicitConstructorArgs(*this, Ctor, CtorType,
                                             ForVirtualBase, Delegating, Args);

  // Emit a simplified prolog. We only need to emit the implicit params.
  assert(Args.size() >= Params.size() && "too few arguments for call");
  for (unsigned I = 0, N = Args.size(); I != N; ++I) {
    if (I < Params.size() && isa<ImplicitParamDecl>(Params[I])) {
      const RValue &RV =
          Args[I].getRValue(*this, getLoc(Ctor->getSourceRange()));
      assert(!RV.isComplex() && "complex indirect params not supported");
      llvm_unreachable("NYI");
    }
  }

  llvm_unreachable("NYI");
}
