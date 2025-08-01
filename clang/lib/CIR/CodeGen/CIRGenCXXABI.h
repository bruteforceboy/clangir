//===----- CIRGenCXXABI.h - Interface to C++ ABIs ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENCXXABI_H
#define LLVM_CLANG_LIB_CIR_CIRGENCXXABI_H

#include "CIRGenCall.h"
#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "mlir/IR/Attributes.h"
#include "clang/AST/Mangle.h"

namespace clang::CIRGen {

class CIRGenFunction;
class CIRGenFunctionInfo;

/// Implements C++ ABI-specific code generation functions.
class CIRGenCXXABI {
protected:
  CIRGenModule &CGM;
  std::unique_ptr<clang::MangleContext> MangleContext;

  CIRGenCXXABI(CIRGenModule &CGM)
      : CGM{CGM}, MangleContext(CGM.getASTContext().createMangleContext()) {}

  clang::ASTContext &getContext() const { return CGM.getASTContext(); }

  virtual bool requiresArrayCookie(const CXXNewExpr *E);

public:
  /// Similar to AddedStructorArgs, but only notes the number of additional
  /// arguments.
  struct AddedStructorArgCounts {
    unsigned Prefix = 0;
    unsigned Suffix = 0;
    AddedStructorArgCounts() = default;
    AddedStructorArgCounts(unsigned P, unsigned S) : Prefix(P), Suffix(S) {}
    static AddedStructorArgCounts prefix(unsigned N) { return {N, 0}; }
    static AddedStructorArgCounts suffix(unsigned N) { return {0, N}; }
  };

  /// Additional implicit arguments to add to the beginning (Prefix) and end
  /// (Suffix) of a constructor / destructor arg list.
  ///
  /// Note that Prefix should actually be inserted *after* the first existing
  /// arg; `this` arguments always come first.
  struct AddedStructorArgs {
    struct Arg {
      mlir::Value Value;
      clang::QualType Type;
    };
    llvm::SmallVector<Arg, 1> Prefix;
    llvm::SmallVector<Arg, 1> Suffix;
    AddedStructorArgs() = default;
    AddedStructorArgs(llvm::SmallVector<Arg, 1> P, llvm::SmallVector<Arg, 1> S)
        : Prefix(std::move(P)), Suffix(std::move(S)) {}
    static AddedStructorArgs prefix(llvm::SmallVector<Arg, 1> Args) {
      return {std::move(Args), {}};
    }
    static AddedStructorArgs suffix(llvm::SmallVector<Arg, 1> Args) {
      return {{}, std::move(Args)};
    }
  };

  /// Build the signature of the given constructor or destructor vairant by
  /// adding any required parameters. For convenience, ArgTys has been
  /// initialized with the type of 'this'.
  virtual AddedStructorArgCounts
  buildStructorSignature(clang::GlobalDecl GD,
                         llvm::SmallVectorImpl<clang::CanQualType> &ArgTys) = 0;

  AddedStructorArgCounts
  addImplicitConstructorArgs(CIRGenFunction &CGF,
                             const clang::CXXConstructorDecl *D,
                             clang::CXXCtorType Type, bool ForVirtualBase,
                             bool Delegating, CallArgList &Args);

  clang::ImplicitParamDecl *getThisDecl(CIRGenFunction &CGF) {
    return CGF.CXXABIThisDecl;
  }

  virtual AddedStructorArgs getImplicitConstructorArgs(
      CIRGenFunction &CGF, const clang::CXXConstructorDecl *D,
      clang::CXXCtorType Type, bool ForVirtualBase, bool Delegating) = 0;

  /// Emit the ABI-specific prolog for the function
  virtual void emitInstanceFunctionProlog(SourceLocation Loc,
                                          CIRGenFunction &CGF) = 0;

  /// Get the type of the implicit "this" parameter used by a method. May return
  /// zero if no specific type is applicable, e.g. if the ABI expects the "this"
  /// parameter to point to some artificial offset in a complete object due to
  /// vbases being reordered.
  virtual const clang::CXXRecordDecl *
  getThisArgumentTypeForMethod(const clang::CXXMethodDecl *MD) {
    return MD->getParent();
  }

  /// Return whether the given global decl needs a VTT parameter.
  virtual bool NeedsVTTParameter(clang::GlobalDecl GD);

  /// If the C++ ABI requires the given type be returned in a particular way,
  /// this method sets RetAI and returns true.
  virtual bool classifyReturnType(CIRGenFunctionInfo &FI) const = 0;

  /// Gets the mangle context.
  clang::MangleContext &getMangleContext() { return *MangleContext; }

  clang::ImplicitParamDecl *&getStructorImplicitParamDecl(CIRGenFunction &CGF) {
    return CGF.CXXStructorImplicitParamDecl;
  }

  mlir::Value getStructorImplicitParamValue(CIRGenFunction &CGF) {
    return CGF.CXXStructorImplicitParamValue;
  }

  void setStructorImplicitParamValue(CIRGenFunction &CGF, mlir::Value val) {
    CGF.CXXStructorImplicitParamValue = val;
  }

  /// Perform ABI-specific "this" argument adjustment required prior to
  /// a call of a virtual function.
  /// The "VirtualCall" argument is true iff the call itself is virtual.
  virtual Address adjustThisArgumentForVirtualFunctionCall(CIRGenFunction &CGF,
                                                           GlobalDecl GD,
                                                           Address This,
                                                           bool VirtualCall) {
    return This;
  }

  /// Build a parameter variable suitable for 'this'.
  void buildThisParam(CIRGenFunction &CGF, FunctionArgList &Params);

  /// Loads the incoming C++ this pointer as it was passed by the caller.
  mlir::Value loadIncomingCXXThis(CIRGenFunction &CGF);

  virtual CatchTypeInfo getCatchAllTypeInfo();

  /// Determine whether there's something special about the rules of the ABI
  /// tell us that 'this' is a complete object within the given function.
  /// Obvious common logic like being defined on a final class will have been
  /// taken care of by the caller.
  virtual bool isThisCompleteObject(clang::GlobalDecl GD) const = 0;

  /// Get the implicit (second) parameter that comes after the "this" pointer,
  /// or nullptr if there is isn't one.
  virtual mlir::Value getCXXDestructorImplicitParam(CIRGenFunction &CGF,
                                                    const CXXDestructorDecl *DD,
                                                    CXXDtorType Type,
                                                    bool ForVirtualBase,
                                                    bool Delegating) = 0;

  /// Emit constructor variants required by this ABI.
  virtual void emitCXXConstructors(const clang::CXXConstructorDecl *D) = 0;
  /// Emit dtor variants required by this ABI.
  virtual void emitCXXDestructors(const clang::CXXDestructorDecl *D) = 0;

  /// Emit the destructor call.
  virtual void emitDestructorCall(CIRGenFunction &CGF,
                                  const CXXDestructorDecl *DD, CXXDtorType Type,
                                  bool ForVirtualBase, bool Delegating,
                                  Address This, QualType ThisTy) = 0;

  /// Emit code to force the execution of a destructor during global
  /// teardown.  The default implementation of this uses atexit.
  ///
  /// \param Dtor - a function taking a single pointer argument
  /// \param Addr - a pointer to pass to the destructor function.
  virtual void registerGlobalDtor(CIRGenFunction &CGF, const VarDecl *D,
                                  cir::FuncOp dtor, mlir::Value Addr) = 0;

  virtual void emitVirtualObjectDelete(CIRGenFunction &CGF,
                                       const CXXDeleteExpr *DE, Address Ptr,
                                       QualType ElementType,
                                       const CXXDestructorDecl *Dtor) = 0;

  virtual size_t getSrcArgforCopyCtor(const CXXConstructorDecl *,
                                      FunctionArgList &Args) const = 0;

  virtual void emitBeginCatch(CIRGenFunction &CGF, const CXXCatchStmt *C) = 0;

  /// Get the address of the vtable for the given record decl which should be
  /// used for the vptr at the given offset in RD.
  virtual cir::GlobalOp getAddrOfVTable(const CXXRecordDecl *RD,
                                        CharUnits VPtrOffset) = 0;

  /// Build a virtual function pointer in the ABI-specific way.
  virtual CIRGenCallee getVirtualFunctionPointer(CIRGenFunction &CGF,
                                                 GlobalDecl GD, Address This,
                                                 mlir::Type Ty,
                                                 SourceLocation Loc) = 0;

  /// Checks if ABI requires extra virtual offset for vtable field.
  virtual bool
  isVirtualOffsetNeededForVTableField(CIRGenFunction &CGF,
                                      CIRGenFunction::VPtr Vptr) = 0;

  /// Determine whether it's possible to emit a vtable for \p RD, even
  /// though we do not know that the vtable has been marked as used by semantic
  /// analysis.
  virtual bool canSpeculativelyEmitVTable(const CXXRecordDecl *RD) const = 0;

  /// Emits the VTable definitions required for the given record type.
  virtual void emitVTableDefinitions(CIRGenVTables &CGVT,
                                     const CXXRecordDecl *RD) = 0;

  using DeleteOrMemberCallExpr =
      llvm::PointerUnion<const CXXDeleteExpr *, const CXXMemberCallExpr *>;

  virtual mlir::Value emitVirtualDestructorCall(CIRGenFunction &CGF,
                                                const CXXDestructorDecl *Dtor,
                                                CXXDtorType DtorType,
                                                Address This,
                                                DeleteOrMemberCallExpr E) = 0;

  /// Emit any tables needed to implement virtual inheritance.  For Itanium,
  /// this emits virtual table tables.
  virtual void emitVirtualInheritanceTables(const CXXRecordDecl *RD) = 0;

  virtual mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc,
                                                  QualType Ty) = 0;
  virtual CatchTypeInfo
  getAddrOfCXXCatchHandlerType(mlir::Location loc, QualType Ty,
                               QualType CatchHandlerType) = 0;

  /// Returns true if the given destructor type should be emitted as a linkonce
  /// delegating thunk, regardless of whether the dtor is defined in this TU or
  /// not.
  virtual bool useThunkForDtorVariant(const CXXDestructorDecl *Dtor,
                                      CXXDtorType DT) const = 0;

  virtual cir::GlobalLinkageKind
  getCXXDestructorLinkage(GVALinkage Linkage, const CXXDestructorDecl *Dtor,
                          CXXDtorType DT) const;

  /// Get the address point of the vtable for the given base subobject.
  virtual mlir::Value
  getVTableAddressPoint(BaseSubobject Base,
                        const CXXRecordDecl *VTableClass) = 0;

  /// Get the address point of the vtable for the given base subobject while
  /// building a constructor or a destructor.
  virtual mlir::Value
  getVTableAddressPointInStructor(CIRGenFunction &CGF, const CXXRecordDecl *RD,
                                  BaseSubobject Base,
                                  const CXXRecordDecl *NearestVBase) = 0;

  /// Gets the pure virtual member call function.
  virtual llvm::StringRef getPureVirtualCallName() = 0;

  /// Gets the deleted virtual member call name.
  virtual llvm::StringRef getDeletedVirtualCallName() = 0;

  /// Specify how one should pass an argument of a record type.
  enum class RecordArgABI {
    /// Pass it using the normal C aggregate rules for the ABI, potentially
    /// introducing extra copies and passing some or all of it in registers.
    Default = 0,

    /// Pass it on the stack using its defined layout. The argument must be
    /// evaluated directly into the correct stack position in the arguments
    /// area, and the call machinery must not move it or introduce extra copies.
    DirectInMemory,

    /// Pass it as a pointer to temporary memory.
    Indirect
  };

  /// Returns how an argument of the given record type should be passed.
  virtual RecordArgABI
  getRecordArgABI(const clang::CXXRecordDecl *RD) const = 0;

  /// Return true if the given member pointer can be zero-initialized
  /// (in the C++ sense) with an LLVM zeroinitializer.
  virtual bool isZeroInitializable(const MemberPointerType *MPT);

  /// Return whether or not a member pointers type is convertible to an IR type.
  virtual bool isMemberPointerConvertible(const MemberPointerType *MPT) const {
    return true;
  }

  /// Gets the offsets of all the virtual base pointers in a given class.
  virtual std::vector<CharUnits> getVBPtrOffsets(const CXXRecordDecl *RD);

  /// Insert any ABI-specific implicit parameters into the parameter list for a
  /// function. This generally involves extra data for constructors and
  /// destructors.
  ///
  /// ABIs may also choose to override the return type, which has been
  /// initialized with the type of 'this' if HasThisReturn(CGF.CurGD) is true or
  /// the formal return type of the function otherwise.
  virtual void addImplicitStructorParams(CIRGenFunction &CGF,
                                         clang::QualType &ResTy,
                                         FunctionArgList &Params) = 0;

  /// Checks if ABI requires to initialize vptrs for given dynamic class.
  virtual bool
  doStructorsInitializeVPtrs(const clang::CXXRecordDecl *VTableClass) = 0;

  /// Returns true if the given constructor or destructor is one of the kinds
  /// that the ABI says returns 'this' (only applies when called non-virtually
  /// for destructors).
  ///
  /// There currently is no way to indicate if a destructor returns 'this' when
  /// called virtually, and CIR generation does not support this case.
  virtual bool HasThisReturn(clang::GlobalDecl GD) const { return false; }

  virtual bool hasMostDerivedReturn(clang::GlobalDecl GD) const {
    return false;
  }

  /// Returns true if the target allows calling a function through a pointer
  /// with a different signature than the actual function (or equivalently,
  /// bitcasting a function or function pointer to a different function type).
  /// In principle in the most general case this could depend on the target, the
  /// calling convention, and the actual types of the arguments and return
  /// value. Here it just means whether the signature mismatch could *ever* be
  /// allowed; in other words, does the target do strict checking of signatures
  /// for all calls.
  virtual bool canCallMismatchedFunctionType() const { return true; }

  virtual ~CIRGenCXXABI();

  void setCXXABIThisValue(CIRGenFunction &CGF, mlir::Value ThisPtr);

  // Determine if references to thread_local global variables can be made
  // directly or require access through a thread wrapper function.
  virtual bool usesThreadWrapperFunction(const VarDecl *VD) const = 0;

  /// Emit the code to initialize hidden members required to handle virtual
  /// inheritance, if needed by the ABI.
  virtual void
  initializeHiddenVirtualInheritanceMembers(CIRGenFunction &CGF,
                                            const CXXRecordDecl *RD) {}

  /// Emit a single constructor/destructor with the gien type from a C++
  /// constructor Decl.
  virtual void emitCXXStructor(clang::GlobalDecl GD) = 0;

  virtual void emitRethrow(CIRGenFunction &CGF, bool isNoReturn) = 0;
  virtual void emitThrow(CIRGenFunction &CGF, const CXXThrowExpr *E) = 0;

  virtual void emitBadCastCall(CIRGenFunction &CGF, mlir::Location loc) = 0;

  virtual mlir::Value
  getVirtualBaseClassOffset(mlir::Location loc, CIRGenFunction &CGF,
                            Address This, const CXXRecordDecl *ClassDecl,
                            const CXXRecordDecl *BaseClassDecl) = 0;

  virtual mlir::Value emitDynamicCast(CIRGenFunction &CGF, mlir::Location Loc,
                                      QualType SrcRecordTy,
                                      QualType DestRecordTy,
                                      cir::PointerType DestCIRTy,
                                      bool isRefCast, Address Src) = 0;

  virtual cir::MethodAttr buildVirtualMethodAttr(cir::MethodType MethodTy,
                                                 const CXXMethodDecl *MD) = 0;

  /**************************** Array cookies ******************************/

  /// Returns the extra size required in order to store the array
  /// cookie for the given new-expression.  May return 0 to indicate that no
  /// array cookie is required.
  ///
  /// Several cases are filtered out before this method is called:
  ///   - non-array allocations never need a cookie
  ///   - calls to \::operator new(size_t, void*) never need a cookie
  ///
  /// \param E - the new-expression being allocated.
  virtual CharUnits getArrayCookieSize(const CXXNewExpr *E);

  /// Initialize the array cookie for the given allocation.
  ///
  /// \param NewPtr - a char* which is the presumed-non-null
  ///   return value of the allocation function
  /// \param NumElements - the computed number of elements,
  ///   potentially collapsed from the multidimensional array case;
  ///   always a size_t
  /// \param ElementType - the base element allocated type,
  ///   i.e. the allocated type after stripping all array types
  virtual Address initializeArrayCookie(CIRGenFunction &CGF, Address NewPtr,
                                        mlir::Value NumElements,
                                        const CXXNewExpr *E,
                                        QualType ElementType) = 0;

protected:
  /// Returns the extra size required in order to store the array
  /// cookie for the given type.  Assumes that an array cookie is
  /// required.
  virtual CharUnits getArrayCookieSizeImpl(QualType ElementType) = 0;
};

/// Creates and Itanium-family ABI
CIRGenCXXABI *CreateCIRGenItaniumCXXABI(CIRGenModule &CGM);

} // namespace clang::CIRGen

#endif
