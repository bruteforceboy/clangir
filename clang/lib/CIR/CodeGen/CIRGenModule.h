//===--- CIRGenModule.h - Per-Module state for CIR gen ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
#define LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H

#include "Address.h"
#include "CIRGenBuilder.h"
#include "CIRGenCUDARuntime.h"
#include "CIRGenCall.h"
#include "CIRGenOpenCLRuntime.h"
#include "CIRGenTBAA.h"
#include "CIRGenTypeCache.h"
#include "CIRGenTypes.h"
#include "CIRGenVTables.h"
#include "CIRGenValue.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

namespace clang::CIRGen {

class CIRGenFunction;
class CIRGenCXXABI;
class TargetCIRGenInfo;
class CIRGenOpenMPRuntime;

enum ForDefinition_t : bool { NotForDefinition = false, ForDefinition = true };

/// Implementation of a CIR/MLIR emission from Clang AST.
///
/// This will emit operations that are specific to C(++)/ObjC(++) language,
/// preserving the semantics of the language and (hopefully) allow to perform
/// accurate analysis and transformation based on these high level semantics.
class CIRGenModule : public CIRGenTypeCache {
  CIRGenModule(CIRGenModule &) = delete;
  CIRGenModule &operator=(CIRGenModule &) = delete;

public:
  CIRGenModule(mlir::MLIRContext &mlirContext, clang::ASTContext &astContext,
               const clang::CodeGenOptions &CGO,
               clang::DiagnosticsEngine &Diags);

  ~CIRGenModule();

  const std::string &getModuleNameHash() const { return ModuleNameHash; }

private:
  mutable std::unique_ptr<TargetCIRGenInfo> theTargetCIRGenInfo;

  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point": this
  /// is where the next operations will be introduced.
  CIRGenBuilderTy builder;

  /// Hold Clang AST information.
  clang::ASTContext &astContext;

  const clang::LangOptions &langOpts;

  const clang::CodeGenOptions &codeGenOpts;

  /// A "module" matches a c/cpp source file: containing a list of functions.
  mlir::ModuleOp theModule;

  clang::DiagnosticsEngine &Diags;

  const clang::TargetInfo &target;

  std::unique_ptr<CIRGenCXXABI> ABI;

  std::unique_ptr<CIRGenTBAA> tbaa;

  /// Used for `UniqueInternalLinkageNames` option
  std::string ModuleNameHash = "";

  /// Per-module type mapping from clang AST to CIR.
  CIRGenTypes genTypes;

  /// Holds information about C++ vtables.
  CIRGenVTables VTables;

  /// Holds the OpenCL runtime
  std::unique_ptr<CIRGenOpenCLRuntime> openCLRuntime;

  /// Holds the OpenMP runtime
  std::unique_ptr<CIRGenOpenMPRuntime> openMPRuntime;

  /// Holds the CUDA runtime
  std::unique_ptr<CIRGenCUDARuntime> cudaRuntime;

  /// Per-function codegen information. Updated everytime emitCIR is called
  /// for FunctionDecls's.
  CIRGenFunction *CurCGF = nullptr;

  // A set of references that have only been set via a weakref so far. This is
  // used to remove the weak of the reference if we ever see a direct reference
  // or a definition.
  llvm::SmallPtrSet<mlir::Operation *, 10> WeakRefReferences;

  /// -------
  /// Declaring variables
  /// -------

  /// Set of global decls for which we already diagnosed mangled name conflict.
  /// Required to not issue a warning (on a mangling conflict) multiple times
  /// for the same decl.
  llvm::DenseSet<clang::GlobalDecl> DiagnosedConflictingDefinitions;

  /// -------
  /// Annotations
  /// -------

  /// We do not store global annotations in the module here, instead, we store
  /// each annotation as attribute of GlobalOp and FuncOp.
  /// We defer creation of global annotation variable to LoweringPrepare
  /// as CIR passes do not need to have a global view of all annotations.

  /// Used for uniquing of annotation arguments.
  llvm::DenseMap<unsigned, mlir::ArrayAttr> annotationArgs;

  /// Store deferred function annotations so they can be emitted at the end with
  /// most up to date ValueDecl that will have all the inherited annotations.
  llvm::DenseMap<llvm::StringRef, const ValueDecl *> deferredAnnotations;

  llvm::DenseMap<const Expr *, mlir::Operation *>
      materializedGlobalTemporaryMap;

public:
  mlir::ModuleOp getModule() const { return theModule; }
  CIRGenBuilderTy &getBuilder() { return builder; }
  clang::ASTContext &getASTContext() const { return astContext; }
  const clang::TargetInfo &getTarget() const { return target; }
  const clang::CodeGenOptions &getCodeGenOpts() const { return codeGenOpts; }
  clang::DiagnosticsEngine &getDiags() const { return Diags; }
  CIRGenTypes &getTypes() { return genTypes; }
  const clang::LangOptions &getLangOpts() const { return langOpts; }
  CIRGenFunction *getCurrCIRGenFun() const { return CurCGF; }
  const cir::CIRDataLayout getDataLayout() const {
    // FIXME(cir): instead of creating a CIRDataLayout every time, set it as an
    // attribute for the CIRModule class.
    return {theModule};
  }

  CIRGenCXXABI &getCXXABI() const { return *ABI; }
  mlir::MLIRContext &getMLIRContext() { return *builder.getContext(); }

  /// -------
  /// Handling globals
  /// -------

  // TODO(cir): does this really need to be a state for CIR emission?
  GlobalDecl initializedGlobalDecl;

  /// Global variables with initializers that need to run before main.
  /// TODO(cir): for now track a generation operation, this is so far only
  /// used to sync with DelayedCXXInitPosition. Improve it when we actually
  /// use function calls for initialization
  std::vector<mlir::Operation *> CXXGlobalInits;

  /// Emit the function that initializes C++ globals.
  void emitCXXGlobalInitFunc();

  /// Track whether the CIRGenModule is currently building an initializer
  /// for a global (e.g. as opposed to a regular cir.func).
  cir::GlobalOp globalOpContext = nullptr;

  /// When a C++ decl with an initializer is deferred, null is
  /// appended to CXXGlobalInits, and the index of that null is placed
  /// here so that the initializer will be performed in the correct
  /// order. Once the decl is emitted, the index is replaced with ~0U to ensure
  /// that we don't re-emit the initializer.
  llvm::DenseMap<const Decl *, unsigned> DelayedCXXInitPosition;

  /// Keep track of a map between lambda fields and names, this needs to be per
  /// module since lambdas might get generated later as part of defered work,
  /// and since the pointers are supposed to be uniqued, should be fine. Revisit
  /// this if it ends up taking too much memory.
  llvm::DenseMap<const clang::FieldDecl *, llvm::StringRef> LambdaFieldToName;

  /// If the declaration has internal linkage but is inside an
  /// extern "C" linkage specification, prepare to emit an alias for it
  /// to the expected name.
  template <typename SomeDecl>
  void maybeHandleStaticInExternC(const SomeDecl *D, cir::GlobalOp GV);

  /// Tell the consumer that this variable has been instantiated.
  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD);

  llvm::DenseMap<const Decl *, cir::GlobalOp> StaticLocalDeclMap;
  llvm::DenseMap<llvm::StringRef, mlir::Value> Globals;
  mlir::Operation *getGlobalValue(llvm::StringRef Ref);
  mlir::Value getGlobalValue(const clang::Decl *D);

  /// If the specified mangled name is not in the module, create and return an
  /// mlir::GlobalOp value
  cir::GlobalOp
  getOrCreateCIRGlobal(llvm::StringRef MangledName, mlir::Type Ty,
                       LangAS AddrSpace, const VarDecl *D,
                       ForDefinition_t IsForDefinition = NotForDefinition);

  cir::GlobalOp getStaticLocalDeclAddress(const VarDecl *D) {
    return StaticLocalDeclMap[D];
  }

  void setStaticLocalDeclAddress(const VarDecl *D, cir::GlobalOp C) {
    StaticLocalDeclMap[D] = C;
  }

  cir::GlobalOp getOrCreateStaticVarDecl(const VarDecl &D,
                                         cir::GlobalLinkageKind Linkage);

  cir::GlobalOp getOrCreateCIRGlobal(const VarDecl *D, mlir::Type Ty,
                                     ForDefinition_t IsForDefinition);

  /// TODO(cir): once we have cir.module, add this as a convenience method
  /// there instead of here.
  ///
  /// Look up the specified global in the module symbol table.
  ///   1. If it does not exist, add a declaration of the global and return it.
  ///   2. Else, the global exists but has the wrong type: return the function
  ///      with a constantexpr cast to the right type.
  ///   3. Finally, if the existing global is the correct declaration, return
  ///      the existing global.
  cir::GlobalOp
  getOrInsertGlobal(mlir::Location loc, llvm::StringRef Name, mlir::Type Ty,
                    llvm::function_ref<cir::GlobalOp()> CreateGlobalCallback);

  // Overload to construct a global variable using its constructor's defaults.
  cir::GlobalOp getOrInsertGlobal(mlir::Location loc, llvm::StringRef Name,
                                  mlir::Type Ty);

  static cir::GlobalOp createGlobalOp(
      CIRGenModule &cgm, mlir::Location loc, llvm::StringRef name, mlir::Type t,
      bool isConstant = false,
      cir::AddressSpace addrSpace = cir::AddressSpace::Default,
      mlir::Operation *insertPoint = nullptr,
      cir::GlobalLinkageKind linkage = cir::GlobalLinkageKind::ExternalLinkage);

  /// Add a global constructor or destructor to the module.
  /// The priority is optional, if not specified, the default priority is used.
  void AddGlobalCtor(cir::FuncOp ctor,
                     std::optional<int> priority = std::nullopt);
  void AddGlobalDtor(cir::FuncOp dtor,
                     std::optional<int> priority = std::nullopt,
                     bool isDtorAttrFunc = false);

  // Return whether structured convergence intrinsics should be generated for
  // this target.

  bool shouldEmitCUDAGlobalVar(const VarDecl *global) const;

  /// Print the postfix for externalized static variable or kernels for single
  /// source offloading languages CUDA and HIP. The unique postfix is created
  /// using either the CUID argument, or the file's UniqueID and active macros.
  /// The fallback method without a CUID requires that the offloading toolchain
  /// does not define separate macros via the -cc1 options.
  void printPostfixForExternalizedDecl(llvm::raw_ostream &OS,
                                       const Decl *D) const;

  bool shouldEmitConvergenceTokens() const {
    // TODO: this shuld probably become unconditional once the controlled
    // convergence becomes the norm.
    return getTriple().isSPIRVLogical();
  }

  bool shouldZeroInitPadding() const {
    // In C23 (N3096) $6.7.10:
    // """
    // If any object is initialized with an empty initializer, then it is
    // subject to default initialization:
    //  - if it is an aggregate, every member is initialized (recursively)
    //  according to these rules, and any padding is initialized to zero bits;
    //  - if it is a union, the first named member is initialized (recursively)
    //  according to these rules, and any padding is initialized to zero bits.
    //
    // If the aggregate or union contains elements or members that are
    // aggregates or unions, these rules apply recursively to the subaggregates
    // or contained unions.
    //
    // If there are fewer initializers in a brace-enclosed list than there are
    // elements or members of an aggregate, or fewer characters in a string
    // literal used to initialize an array of known size than there are elements
    // in the array, the remainder of the aggregate is subject to default
    // initialization.
    // """
    //
    // The standard seems ambiguous in the following two areas:
    // 1. For a union type with empty initializer, if the first named member is
    // not the largest member, then the bytes comes after the first named member
    // but before padding are left unspecified. An example is:
    //    union U { int a; long long b;};
    //    union U u = {};  // The first 4 bytes are 0, but 4-8 bytes are left
    //    unspecified.
    //
    // 2. It only mentions padding for empty initializer, but doesn't mention
    // padding for a non empty initialization list. And if the aggregation or
    // union contains elements or members that are aggregates or unions, and
    // some are non empty initializers, while others are empty initializers,
    // the padding initialization is unclear. An example is:
    //    struct S1 { int a; long long b; };
    //    struct S2 { char c; struct S1 s1; };
    //    // The values for paddings between s2.c and s2.s1.a, between s2.s1.a
    //    and s2.s1.b are unclear.
    //    struct S2 s2 = { 'c' };
    //
    // Here we choose to zero initiailize left bytes of a union type because
    // projects like the Linux kernel are relying on this behavior. If we don't
    // explicitly zero initialize them, the undef values can be optimized to
    // return garbage data. We also choose to zero initialize paddings for
    // aggregates and unions, no matter they are initialized by empty
    // initializers or non empty initializers. This can provide a consistent
    // behavior. So projects like the Linux kernel can rely on it.
    return !getLangOpts().CPlusPlus;
  }

  llvm::StringMap<unsigned> cgGlobalNames;
  std::string getUniqueGlobalName(const std::string &baseName);

  /// Return the mlir::Value for the address of the given global variable.
  /// If Ty is non-null and if the global doesn't exist, then it will be created
  /// with the specified type instead of whatever the normal requested type
  /// would be. If IsForDefinition is true, it is guaranteed that an actual
  /// global with type Ty will be returned, not conversion of a variable with
  /// the same mangled name but some other type.
  mlir::Value
  getAddrOfGlobalVar(const VarDecl *D, mlir::Type Ty = {},
                     ForDefinition_t IsForDefinition = NotForDefinition);

  /// Return the mlir::GlobalViewAttr for the address of the given global.
  cir::GlobalViewAttr
  getAddrOfGlobalVarAttr(const VarDecl *D, mlir::Type Ty = {},
                         ForDefinition_t IsForDefinition = NotForDefinition);

  /// Get a reference to the target of VD.
  mlir::Operation *getWeakRefReference(const ValueDecl *VD);

  CharUnits
  computeNonVirtualBaseClassOffset(const CXXRecordDecl *DerivedClass,
                                   CastExpr::path_const_iterator Start,
                                   CastExpr::path_const_iterator End);

  /// Returns the offset from a derived class to a class. Returns null if the
  /// offset is 0.
  CharUnits
  getNonVirtualBaseClassOffset(const CXXRecordDecl *classDecl,
                               CastExpr::path_const_iterator pathBegin,
                               CastExpr::path_const_iterator pathEnd);

  /// Get the CIR attributes and calling convention to use for a particular
  /// function type.
  ///
  /// \param Name - The function name.
  /// \param Info - The function type information.
  /// \param CalleeInfo - The callee information these attributes are being
  /// constructed for. If valid, the attributes applied to this decl may
  /// contribute to the function attributes and calling convention.
  /// \param Attrs [out] - On return, the attribute list to use.
  void constructAttributeList(llvm::StringRef Name,
                              const CIRGenFunctionInfo &Info,
                              CIRGenCalleeInfo CalleeInfo,
                              mlir::NamedAttrList &Attrs,
                              cir::CallingConv &callingConv,
                              cir::SideEffect &sideEffect, bool AttrOnCallSite,
                              bool IsThunk);

  /// Helper function for getDefaultFunctionAttributes. Builds a set of function
  /// attributes which can be simply added to a function.
  void getTrivialDefaultFunctionAttributes(llvm::StringRef name,
                                           bool hasOptnone, bool attrOnCallSite,
                                           mlir::NamedAttrList &funcAttrs);

  /// Helper function for constructAttributeList and
  /// addDefaultFunctionDefinitionAttributes.  Builds a set of function
  /// attributes to add to a function with the given properties.
  void getDefaultFunctionAttributes(llvm::StringRef name, bool hasOptnone,
                                    bool attrOnCallSite,
                                    mlir::NamedAttrList &funcAttrs);

  /// Will return a global variable of the given type. If a variable with a
  /// different type already exists then a new variable with the right type
  /// will be created and all uses of the old variable will be replaced with a
  /// bitcast to the new variable.
  cir::GlobalOp createOrReplaceCXXRuntimeVariable(
      mlir::Location loc, llvm::StringRef Name, mlir::Type Ty,
      cir::GlobalLinkageKind Linkage, clang::CharUnits Alignment);

  /// Emit any vtables which we deferred and still have a use for.
  void emitDeferredVTables();
  bool shouldOpportunisticallyEmitVTables();

  void emitVTable(CXXRecordDecl *rd);

  void setDSOLocal(cir::CIRGlobalValueInterface GV) const;

  /// Return the appropriate linkage for the vtable, VTT, and type information
  /// of the given class.
  cir::GlobalLinkageKind getVTableLinkage(const CXXRecordDecl *RD);

  /// Emit type metadata for the given vtable using the given layout.
  void emitVTableTypeMetadata(const CXXRecordDecl *RD, cir::GlobalOp VTable,
                              const VTableLayout &VTLayout);

  /// Get the address of the RTTI descriptor for the given type.
  mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc, QualType Ty,
                                          bool ForEH = false);

  /// TODO(cir): add CIR visibility bits.
  static mlir::SymbolTable::Visibility getCIRVisibility(Visibility V) {
    switch (V) {
    case DefaultVisibility:
      return mlir::SymbolTable::Visibility::Public;
    case HiddenVisibility:
      return mlir::SymbolTable::Visibility::Private;
    case ProtectedVisibility:
      llvm_unreachable("NYI");
    }
    llvm_unreachable("unknown visibility!");
  }

  llvm::DenseMap<mlir::Attribute, cir::GlobalOp> ConstantStringMap;

  /// Return a constant array for the given string.
  mlir::Attribute getConstantArrayFromStringLiteral(const StringLiteral *E);

  /// Return a global op for the given string literal.
  cir::GlobalOp getGlobalForStringLiteral(const StringLiteral *s,
                                          llvm::StringRef name = ".str");

  /// Return a global symbol reference to a constant array for the given string
  /// literal.
  cir::GlobalViewAttr
  getAddrOfConstantStringFromLiteral(const StringLiteral *S,
                                     llvm::StringRef Name = ".str");

  unsigned CompoundLitaralCnt = 0;
  /// Return the unique name for global compound literal
  std::string createGlobalCompoundLiteralName() {
    return (Twine(".compoundLiteral.") + Twine(CompoundLitaralCnt++)).str();
  }

  /// Return the AST address space of the underlying global variable for D, as
  /// determined by its declaration. Normally this is the same as the address
  /// space of D's type, but in CUDA, address spaces are associated with
  /// declarations, not types. If D is nullptr, return the default address
  /// space for global variable.
  ///
  /// For languages without explicit address spaces, if D has default address
  /// space, target-specific global or constant address space may be returned.
  LangAS getGlobalVarAddressSpace(const VarDecl *D);

  /// Return the AST address space of constant literal, which is used to emit
  /// the constant literal as global variable in LLVM IR.
  /// Note: This is not necessarily the address space of the constant literal
  /// in AST. For address space agnostic language, e.g. C++, constant literal
  /// in AST is always in default address space.
  LangAS getGlobalConstantAddressSpace() const;

  /// Returns the address space for temporary allocations in the language. This
  /// ensures that the allocated variable's address space matches the
  /// expectations of the AST, rather than using the target's allocation address
  /// space, which may lead to type mismatches in other parts of the IR.
  LangAS getLangTempAllocaAddressSpace() const;

  /// Set attributes which are common to any form of a global definition (alias,
  /// Objective-C method, function, global variable).
  ///
  /// NOTE: This should only be called for definitions.
  void setCommonAttributes(GlobalDecl GD, mlir::Operation *GV);

  const TargetCIRGenInfo &getTargetCIRGenInfo();
  const ABIInfo &getABIInfo();

  /// Helpers to convert Clang's SourceLocation to a MLIR Location.
  mlir::Location getLoc(clang::SourceLocation SLoc);
  mlir::Location getLoc(clang::SourceRange SLoc);
  mlir::Location getLoc(mlir::Location lhs, mlir::Location rhs);

  /// Helper to convert Clang's alignment to CIR alignment
  mlir::IntegerAttr getSize(CharUnits size);

  /// Returns whether the given record has public LTO visibility (regardless of
  /// -lto-whole-program-visibility) and therefore may not participate in
  /// (single-module) CFI and whole-program vtable optimization.
  bool AlwaysHasLTOVisibilityPublic(const CXXRecordDecl *RD);

  /// Returns whether the given record has hidden LTO visibility and therefore
  /// may participate in (single-module) CFI and whole-program vtable
  /// optimization.
  bool HasHiddenLTOVisibility(const CXXRecordDecl *RD);

  /// Determine whether an object of this type can be emitted
  /// as a constant.
  ///
  /// If ExcludeCtor is true, the duration when the object's constructor runs
  /// will not be considered. The caller will need to verify that the object is
  /// not written to during its construction.
  /// FIXME: in LLVM codegen path this is part of CGM, which doesn't seem
  /// like necessary, since (1) it doesn't use CGM at all and (2) is AST type
  /// query specific.
  bool isTypeConstant(clang::QualType Ty, bool ExcludeCtor, bool ExcludeDtor);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  /// Return the best known alignment for an unknown pointer to a
  /// particular class.
  clang::CharUnits getClassPointerAlignment(const clang::CXXRecordDecl *RD);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  clang::CharUnits
  getNaturalPointeeTypeAlignment(clang::QualType ty,
                                 LValueBaseInfo *baseInfo = nullptr,
                                 TBAAAccessInfo *tbaaInfo = nullptr);

  /// FIXME: this could likely be a common helper and not necessarily related
  /// with codegen.
  clang::CharUnits getNaturalTypeAlignment(clang::QualType T,
                                           LValueBaseInfo *BaseInfo = nullptr,
                                           TBAAAccessInfo *tbaaInfo = nullptr,
                                           bool forPointeeType = false);

  /// TODO: Add TBAAAccessInfo
  clang::CharUnits
  getDynamicOffsetAlignment(clang::CharUnits actualBaseAlign,
                            const clang::CXXRecordDecl *baseDecl,
                            clang::CharUnits expectedTargetAlign);

  /// Returns the assumed alignment of a virtual base of a class.
  clang::CharUnits getVBaseAlignment(CharUnits DerivedAlign,
                                     const CXXRecordDecl *Derived,
                                     const CXXRecordDecl *VBase);

  cir::FuncOp
  getAddrOfCXXStructor(clang::GlobalDecl GD,
                       const CIRGenFunctionInfo *FnInfo = nullptr,
                       cir::FuncType FnType = nullptr, bool DontDefer = false,
                       ForDefinition_t IsForDefinition = NotForDefinition) {

    return getAddrAndTypeOfCXXStructor(GD, FnInfo, FnType, DontDefer,
                                       IsForDefinition)
        .second;
  }

  /// A queue of (optional) vtables to consider emitting.
  std::vector<const clang::CXXRecordDecl *> DeferredVTables;

  /// A queue of (optional) vtables that may be emitted opportunistically.
  std::vector<const clang::CXXRecordDecl *> opportunisticVTables;

  mlir::Type getVTableComponentType();
  CIRGenVTables &getVTables() { return VTables; }

  ItaniumVTableContext &getItaniumVTableContext() {
    return VTables.getItaniumVTableContext();
  }
  const ItaniumVTableContext &getItaniumVTableContext() const {
    return VTables.getItaniumVTableContext();
  }

  /// Get attribute used to describe accesses to objects of
  /// the given type.
  cir::TBAAAttr getTBAATypeInfo(QualType QTy);

  /// Get TBAA information that describes an access to an object of the given
  /// type.
  TBAAAccessInfo getTBAAAccessInfo(QualType accessType);

  /// Get the TBAA information that describes an access to a virtual table
  /// pointer.
  TBAAAccessInfo getTBAAVTablePtrAccessInfo(mlir::Type VTablePtrType);

  mlir::ArrayAttr getTBAAStructInfo(QualType QTy);

  /// Get metadata that describes the given base access type. Return null if the
  /// type is not suitable for use in TBAA access tags.
  cir::TBAAAttr getTBAABaseTypeInfo(QualType QTy);

  cir::TBAAAttr getTBAAAccessTagInfo(TBAAAccessInfo tbaaInfo);

  /// Get merged TBAA information for the purposes of type casts.
  TBAAAccessInfo mergeTBAAInfoForCast(TBAAAccessInfo SourceInfo,
                                      TBAAAccessInfo TargetInfo);

  /// Get merged TBAA information for the purposes of conditional operator.
  TBAAAccessInfo mergeTBAAInfoForConditionalOperator(TBAAAccessInfo InfoA,
                                                     TBAAAccessInfo InfoB);

  /// Get merged TBAA information for the purposes of memory transfer calls.
  TBAAAccessInfo mergeTBAAInfoForMemoryTransfer(TBAAAccessInfo DestInfo,
                                                TBAAAccessInfo SrcInfo);

  /// Get TBAA information for an access with a given base lvalue.
  TBAAAccessInfo getTBAAInfoForSubobject(LValue Base, QualType AccessType) {
    if (Base.getTBAAInfo().isMayAlias())
      return TBAAAccessInfo::getMayAliasInfo();
    return getTBAAAccessInfo(AccessType);
  }

  template <typename Op>
  void decorateOperationWithTBAA(Op op, TBAAAccessInfo tbaaInfo) {
    if (auto tag = getTBAAAccessTagInfo(tbaaInfo)) {
      op.setTbaaAttr(tag);
    }
  }

  /// This contains all the decls which have definitions but which are deferred
  /// for emission and therefore should only be output if they are actually
  /// used. If a decl is in this, then it is known to have not been referenced
  /// yet.
  std::map<llvm::StringRef, clang::GlobalDecl> DeferredDecls;

  // This is a list of deferred decls which we have seen that *are* actually
  // referenced. These get code generated when the module is done.
  std::vector<clang::GlobalDecl> DeferredDeclsToEmit;
  void addDeferredDeclToEmit(clang::GlobalDecl GD) {
    DeferredDeclsToEmit.emplace_back(GD);
  }

  std::pair<cir::FuncType, cir::FuncOp> getAddrAndTypeOfCXXStructor(
      clang::GlobalDecl GD, const CIRGenFunctionInfo *FnInfo = nullptr,
      cir::FuncType FnType = nullptr, bool Dontdefer = false,
      ForDefinition_t IsForDefinition = NotForDefinition);

  void emitTopLevelDecl(clang::Decl *decl);
  void emitLinkageSpec(const LinkageSpecDecl *D);

  /// Emit code for a single global function or var decl. Forward declarations
  /// are emitted lazily.
  void emitGlobal(clang::GlobalDecl D);

  bool tryEmitBaseDestructorAsAlias(const CXXDestructorDecl *D);

  void emitAliasForGlobal(llvm::StringRef mangledName, mlir::Operation *op,
                          GlobalDecl aliasGD, cir::FuncOp aliasee,
                          cir::GlobalLinkageKind linkage);

  mlir::Type convertType(clang::QualType type);

  /// Set the visibility for the given global.
  void setGlobalVisibility(mlir::Operation *Op, const NamedDecl *D) const;
  void setDSOLocal(mlir::Operation *Op) const;
  /// Set visibility, dllimport/dllexport and dso_local.
  /// This must be called after dllimport/dllexport is set.
  void setGVProperties(mlir::Operation *Op, const NamedDecl *D) const;
  void setGVPropertiesAux(mlir::Operation *Op, const NamedDecl *D) const;

  /// Set the TLS mode for the given global Op for the thread-local
  /// variable declaration D.
  void setTLSMode(mlir::Operation *Op, const VarDecl &D) const;

  /// Get TLS mode from CodeGenOptions.
  cir::TLS_Model GetDefaultCIRTLSModel() const;

  /// Replace the present global `Old` with the given global `New`. Their symbol
  /// names must match; their types can be different. Usages of the old global
  /// will be automatically updated if their types mismatch.
  ///
  /// This function will erase the old global. This function will NOT insert the
  /// new global into the module.
  void replaceGlobal(cir::GlobalOp Old, cir::GlobalOp New);

  /// Determine whether the definition must be emitted; if this returns \c
  /// false, the definition can be emitted lazily if it's used.
  bool MustBeEmitted(const clang::ValueDecl *D);

  /// Whether this function's return type has no side effects, and thus may be
  /// trivially discared if it is unused.
  bool MayDropFunctionReturn(const clang::ASTContext &astContext,
                             clang::QualType ReturnType);

  bool isInNoSanitizeList(clang::SanitizerMask Kind, cir::FuncOp Fn,
                          clang::SourceLocation) const;

  /// Determine whether the definition can be emitted eagerly, or should be
  /// delayed until the end of the translation unit. This is relevant for
  /// definitions whose linkage can change, e.g. implicit function instantions
  /// which may later be explicitly instantiated.
  bool MayBeEmittedEagerly(const clang::ValueDecl *D);

  bool verifyModule();

  /// Return the address of the given function. If Ty is non-null, then this
  /// function will use the specified type if it has to create it.
  // TODO: this is a bit weird as `GetAddr` given we give back a FuncOp?
  cir::FuncOp
  GetAddrOfFunction(clang::GlobalDecl GD, mlir::Type Ty = nullptr,
                    bool ForVTable = false, bool Dontdefer = false,
                    ForDefinition_t IsForDefinition = NotForDefinition);

  mlir::Operation *
  GetAddrOfGlobal(clang::GlobalDecl GD,
                  ForDefinition_t IsForDefinition = NotForDefinition);

  // Return whether RTTI information should be emitted for this target.
  bool shouldEmitRTTI(bool ForEH = false) {
    return (ForEH || getLangOpts().RTTI) && !getLangOpts().CUDAIsDevice &&
           !(getLangOpts().OpenMP && getLangOpts().OpenMPIsTargetDevice &&
             getTriple().isNVPTX());
  }

  // C++ related functions.
  void emitDeclContext(const DeclContext *DC);

  /// Return the result of value-initializing the given type, i.e. a null
  /// expression of the given type.  This is usually, but not always, an LLVM
  /// null constant.
  mlir::Value emitNullConstant(QualType T, mlir::Location loc);

  /// Return a null constant appropriate for zero-initializing a base class with
  /// the given type. This is usually, but not always, an LLVM null constant.
  mlir::TypedAttr emitNullConstantForBase(const CXXRecordDecl *Record);

  mlir::Value emitMemberPointerConstant(const UnaryOperator *E);

  llvm::StringRef getMangledName(clang::GlobalDecl GD);

  void emitTentativeDefinition(const VarDecl *D);

  // Make sure that this type is translated.
  void UpdateCompletedType(const clang::TagDecl *TD);

  /// Set function attributes for a function declaration.
  void setFunctionAttributes(GlobalDecl GD, cir::FuncOp F,
                             bool IsIncompleteFunction, bool IsThunk);

  /// Set the CIR function attributes (sext, zext, etc).
  void setCIRFunctionAttributes(GlobalDecl GD, const CIRGenFunctionInfo &info,
                                cir::FuncOp func, bool isThunk);

  /// Set the CIR function attributes which only apply to a function
  /// definition.
  void setCIRFunctionAttributesForDefinition(const Decl *decl,
                                             cir::FuncOp func);

  void emitGlobalDefinition(clang::GlobalDecl D, mlir::Operation *Op = nullptr);
  void emitGlobalFunctionDefinition(clang::GlobalDecl D, mlir::Operation *Op);
  void emitGlobalVarDefinition(const clang::VarDecl *D,
                               bool IsTentative = false);

  /// Emit the function that initializes the specified global
  void emitCXXGlobalVarDeclInit(const VarDecl *varDecl, cir::GlobalOp addr,
                                bool performInit);

  void emitCXXGlobalVarDeclInitFunc(const VarDecl *D, cir::GlobalOp Addr,
                                    bool PerformInit);

  void addDeferredVTable(const CXXRecordDecl *RD) {
    DeferredVTables.push_back(RD);
  }

  /// Stored a deferred empty coverage mapping for an unused and thus
  /// uninstrumented top level declaration.
  void AddDeferredUnusedCoverageMapping(clang::Decl *D);

  std::nullptr_t getModuleDebugInfo() { return nullptr; }

  /// Emit any needed decls for which code generation was deferred.
  void emitDeferred(unsigned recursionLimit);

  /// Try to emit external vtables as available_externally if they have emitted
  /// all inlined virtual functions.  It runs after EmitDeferred() and therefore
  /// is not allowed to create new references to things that need to be emitted
  /// lazily.
  void emitVTablesOpportunistically();

  /// Helper for `emitDeferred` to apply actual codegen.
  void emitGlobalDecl(clang::GlobalDecl &D);

  const llvm::Triple &getTriple() const { return target.getTriple(); }

  // Finalize CIR code generation.
  void Release();

  bool isTriviallyRecursive(const clang::FunctionDecl *func);

  bool shouldEmitFunction(clang::GlobalDecl globalDecl);

  /// Returns a pointer to a global variable representing a temporary with
  /// static or thread storage duration.
  mlir::Operation *
  getAddrOfGlobalTemporary(const MaterializeTemporaryExpr *expr,
                           const Expr *init);

  // Produce code for this constructor/destructor. This method doesn't try to
  // apply any ABI rules about which other constructors/destructors are needed
  // or if they are alias to each other.
  cir::FuncOp codegenCXXStructor(clang::GlobalDecl GD);

  bool lookupRepresentativeDecl(llvm::StringRef MangledName,
                                clang::GlobalDecl &Result) const;

  bool supportsCOMDAT() const;
  void maybeSetTrivialComdat(const clang::Decl &d, mlir::Operation *op);

  void emitError(const llvm::Twine &message) { theModule.emitError(message); }

  /// -------
  /// Visibility and Linkage
  /// -------

  static void setInitializer(cir::GlobalOp &op, mlir::Attribute value);
  static mlir::SymbolTable::Visibility
  getMLIRVisibilityFromCIRLinkage(cir::GlobalLinkageKind GLK);
  static cir::VisibilityKind getGlobalVisibilityKindFromClangVisibility(
      clang::VisibilityAttr::VisibilityType visibility);
  cir::VisibilityAttr getGlobalVisibilityAttrFromDecl(const Decl *decl);
  static mlir::SymbolTable::Visibility getMLIRVisibility(cir::GlobalOp op);
  cir::GlobalLinkageKind getFunctionLinkage(GlobalDecl GD);
  cir::GlobalLinkageKind getCIRLinkageForDeclarator(const DeclaratorDecl *D,
                                                    GVALinkage Linkage,
                                                    bool IsConstantVariable);
  void setFunctionLinkage(GlobalDecl GD, cir::FuncOp f) {
    auto L = getFunctionLinkage(GD);
    f.setLinkageAttr(cir::GlobalLinkageKindAttr::get(&getMLIRContext(), L));
    mlir::SymbolTable::setSymbolVisibility(f,
                                           getMLIRVisibilityFromCIRLinkage(L));
  }

  cir::GlobalLinkageKind getCIRLinkageVarDefinition(const VarDecl *VD,
                                                    bool IsConstant);

  void addReplacement(llvm::StringRef Name, mlir::Operation *Op);

  mlir::Location getLocForFunction(const clang::FunctionDecl *FD);

  void ReplaceUsesOfNonProtoTypeWithRealFunction(mlir::Operation *Old,
                                                 cir::FuncOp NewFn);

  // TODO: CodeGen also passes an AttributeList here. We'll have to match that
  // in CIR
  cir::FuncOp
  GetOrCreateCIRFunction(llvm::StringRef MangledName, mlir::Type Ty,
                         clang::GlobalDecl D, bool ForVTable,
                         bool DontDefer = false, bool IsThunk = false,
                         ForDefinition_t IsForDefinition = NotForDefinition,
                         mlir::ArrayAttr ExtraAttrs = {});
  // Effectively create the CIR instruction, properly handling insertion
  // points.
  cir::FuncOp createCIRFunction(mlir::Location loc, llvm::StringRef name,
                                cir::FuncType Ty,
                                const clang::FunctionDecl *FD);

  cir::FuncOp createRuntimeFunction(cir::FuncType Ty, llvm::StringRef Name,
                                    mlir::ArrayAttr = {}, bool Local = false,
                                    bool AssumeConvergent = false);

  /// Emit type info if type of an expression is a variably modified
  /// type. Also emit proper debug info for cast types.
  void emitExplicitCastExprType(const ExplicitCastExpr *E,
                                CIRGenFunction *CGF = nullptr);

  static constexpr const char *builtinCoroId = "__builtin_coro_id";
  static constexpr const char *builtinCoroAlloc = "__builtin_coro_alloc";
  static constexpr const char *builtinCoroBegin = "__builtin_coro_begin";
  static constexpr const char *builtinCoroEnd = "__builtin_coro_end";

  /// Given a builtin id for a function like "__builtin_fabsf", return a
  /// Function* for "fabsf".
  cir::FuncOp getBuiltinLibFunction(const FunctionDecl *FD, unsigned BuiltinID);

  /// Emit a general error that something can't be done.
  void Error(SourceLocation loc, llvm::StringRef error);

  /// Print out an error that codegen doesn't support the specified stmt yet.
  void ErrorUnsupported(const Stmt *S, const char *Type);

  /// Print out an error that codegen doesn't support the specified decl yet.
  void ErrorUnsupported(const Decl *D, const char *Type);

  /// Return a reference to the configured OpenCL runtime.
  CIRGenOpenCLRuntime &getOpenCLRuntime() {
    assert(openCLRuntime != nullptr);
    return *openCLRuntime;
  }

  /// Return a reference to the configured CUDA runtime.
  CIRGenCUDARuntime &getCUDARuntime() {
    assert(cudaRuntime != nullptr);
    return *cudaRuntime;
  }

  void createOpenCLRuntime() {
    openCLRuntime.reset(new CIRGenOpenCLRuntime(*this));
  }

  /// Return a reference to the configured OpenMP runtime.
  CIRGenOpenMPRuntime &getOpenMPRuntime() {
    assert(openMPRuntime != nullptr);
    return *openMPRuntime;
  }

  /// OpenCL v1.2 s5.6.4.6 allows the compiler to store kernel argument
  /// information in the program executable. The argument information stored
  /// includes the argument name, its type, the address and access qualifiers
  /// used. This helper can be used to generate metadata for source code kernel
  /// function as well as generated implicitly kernels. If a kernel is generated
  /// implicitly null value has to be passed to the last two parameters,
  /// otherwise all parameters must have valid non-null values.
  /// \param FN is a pointer to IR function being generated.
  /// \param FD is a pointer to function declaration if any.
  /// \param CGF is a pointer to CIRGenFunction that generates this function.
  void genKernelArgMetadata(cir::FuncOp FN, const FunctionDecl *FD = nullptr,
                            CIRGenFunction *CGF = nullptr);

  /// Emits OpenCL specific Metadata e.g. OpenCL version.
  void emitOpenCLMetadata();

  /// Create cir::AnnotationAttr which contains the annotation
  /// information for a given GlobalValue. Notice that a GlobalValue could
  /// have multiple annotations, and this function creates attribute for
  /// one of them.
  cir::AnnotationAttr emitAnnotateAttr(const clang::AnnotateAttr *aa);

private:
  // An ordered map of canonical GlobalDecls to their mangled names.
  llvm::MapVector<clang::GlobalDecl, llvm::StringRef> MangledDeclNames;
  llvm::StringMap<clang::GlobalDecl, llvm::BumpPtrAllocator> Manglings;

  // FIXME: should we use llvm::TrackingVH<mlir::Operation> here?
  typedef llvm::StringMap<mlir::Operation *> ReplacementsTy;
  ReplacementsTy Replacements;
  /// Call replaceAllUsesWith on all pairs in Replacements.
  void applyReplacements();

  /// A helper function to replace all uses of OldF to NewF that replace
  /// the type of pointer arguments. This is not needed to tradtional
  /// pipeline since LLVM has opaque pointers but CIR not.
  void replacePointerTypeArgs(cir::FuncOp OldF, cir::FuncOp NewF);

  void setNonAliasAttributes(GlobalDecl GD, mlir::Operation *GV);
  /// Map source language used to a CIR attribute.
  cir::SourceLanguage getCIRSourceLanguage();

  /// Emit all the global annotations.
  /// This actually only emits annotations for deffered declarations of
  /// functions, because global variables need no deffred emission.
  void emitGlobalAnnotations();

  /// Emit additional args of the annotation.
  mlir::ArrayAttr emitAnnotationArgs(const clang::AnnotateAttr *attr);

  /// Add global annotations for a global value.
  /// Those annotations are emitted during lowering to the LLVM code.
  void addGlobalAnnotations(const ValueDecl *d, mlir::Operation *gv);
};
} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CODEGEN_CIRGENMODULE_H
