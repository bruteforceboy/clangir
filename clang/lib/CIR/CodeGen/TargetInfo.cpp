
#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"

#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/Target/x86.h"

using namespace clang;
using namespace clang::CIRGen;

static bool isAggregateTypeForABI(QualType T) {
  return !CIRGenFunction::hasScalarEvaluationKind(T) ||
         T->isMemberFunctionPointerType();
}

/// Pass transparent unions as if they were the type of the first element. Sema
/// should ensure that all elements of the union have the same "machine type".
static QualType useFirstFieldIfTransparentUnion(QualType Ty) {
  assert(!Ty->getAsUnionType() && "NYI");
  return Ty;
}

bool clang::CIRGen::isEmptyRecordForLayout(const ASTContext &Context,
                                           QualType T) {
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return false;

  const RecordDecl *RD = RT->getDecl();

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (CXXRD->isDynamicClass())
      return false;

    for (const auto &I : CXXRD->bases())
      if (!isEmptyRecordForLayout(Context, I.getType()))
        return false;
  }

  for (const auto *I : RD->fields())
    if (!isEmptyFieldForLayout(Context, I))
      return false;

  return true;
}

bool clang::CIRGen::isEmptyFieldForLayout(const ASTContext &Context,
                                          const FieldDecl *FD) {
  if (FD->isZeroLengthBitField())
    return true;

  if (FD->isUnnamedBitField())
    return false;

  return isEmptyRecordForLayout(Context, FD->getType());
}

namespace {

/// The default implementation for ABI specific
/// details. This implementation provides information which results in
/// self-consistent and sensible LLVM IR generation, but does not
/// conform to any particular ABI.
class DefaultABIInfo : public ABIInfo {
public:
  DefaultABIInfo(CIRGenTypes &CGT) : ABIInfo(CGT) {}

  virtual ~DefaultABIInfo() = default;

  cir::ABIArgInfo classifyReturnType(QualType RetTy) const {
    if (RetTy->isVoidType())
      return cir::ABIArgInfo::getIgnore();

    if (isAggregateTypeForABI(RetTy))
      llvm_unreachable("NYI");

    // Treat an enum type as its underlying type.
    if (RetTy->getAs<EnumType>())
      llvm_unreachable("NYI");

    if (RetTy->getAs<BitIntType>())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(RetTy)
                ? cir::ABIArgInfo::getExtend(RetTy)
                : cir::ABIArgInfo::getDirect());
  }

  cir::ABIArgInfo classifyArgumentType(QualType Ty) const {
    Ty = useFirstFieldIfTransparentUnion(Ty);

    if (isAggregateTypeForABI(Ty)) {
      llvm_unreachable("NYI");
    }

    // Treat an enum type as its underlying type.
    if (Ty->getAs<EnumType>())
      llvm_unreachable("NYI");

    if (Ty->getAs<BitIntType>())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(Ty) ? cir::ABIArgInfo::getExtend(Ty)
                                              : cir::ABIArgInfo::getDirect());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AArch64 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {
using ABIKind = cir::AArch64ABIKind;

class AArch64ABIInfo : public ABIInfo {
private:
  ABIKind Kind;

public:
  AArch64ABIInfo(CIRGenTypes &CGT, ABIKind Kind) : ABIInfo(CGT), Kind(Kind) {}
  virtual bool allowBFloatArgsAndRet() const override {
    // TODO: Should query target info instead of hardcoding.
    assert(!cir::MissingFeatures::useTargetLoweringABIInfo());
    return true;
  }

private:
  ABIKind getABIKind() const { return Kind; }
  bool isDarwinPCS() const { return Kind == ABIKind::DarwinPCS; }

  cir::ABIArgInfo classifyReturnType(QualType RetTy, bool IsVariadic) const;
  cir::ABIArgInfo classifyArgumentType(QualType RetTy, bool IsVariadic,
                                       unsigned CallingConvention) const;
};

class AArch64TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  AArch64TargetCIRGenInfo(CIRGenTypes &CGT, ABIKind Kind)
      : TargetCIRGenInfo(std::make_unique<AArch64ABIInfo>(CGT, Kind)) {}
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createAArch64TargetCIRGenInfo(CIRGenTypes &CGT,
                                             cir::AArch64ABIKind Kind) {
  return std::make_unique<AArch64TargetCIRGenInfo>(CGT, Kind);
}

//===----------------------------------------------------------------------===//
// X86 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

/// The AVX ABI leel for X86 targets.
using X86AVXABILevel = cir::X86AVXABILevel;

class X86_64ABIInfo : public ABIInfo {
  using Class = cir::X86ArgClass;

  // X86AVXABILevel AVXLevel;
  // Some ABIs (e.g. X32 ABI and Native Client OS) use 32 bit pointers on 64-bit
  // hardware.
  // bool Has64BitPointers;

public:
  X86_64ABIInfo(CIRGenTypes &CGT, X86AVXABILevel AVXLevel)
      : ABIInfo(CGT)
  // , AVXLevel(AVXLevel)
  // , Has64BitPointers(CGT.getDataLayout().getPointeSize(0) == 8)
  {}

  /// classify - Determine the x86_64 register classes in which the given type T
  /// should be passed.
  ///
  /// \param Lo - The classification for the parts of the type residing in the
  /// low word of the containing object.
  ///
  /// \param Hi - The classification for the parts of the type residing in the
  /// high word of the containing object.
  ///
  /// \param OffsetBase - The bit offset of this type in the containing object.
  /// Some parameters are classified different depending on whether they
  /// straddle an eightbyte boundary.
  ///
  /// \param isNamedArg - Whether the argument in question is a "named"
  /// argument, as used in AMD64-ABI 3.5.7.
  ///
  /// If a word is unused its result will be NoClass; if a type should be passed
  /// in Memory then at least the classification of \arg Lo will be Memory.
  ///
  /// The \arg Lo class will be NoClass iff the argument is ignored.
  ///
  /// If the \arg Lo class is ComplexX87, then the \arg Hi class will also be
  /// ComplexX87.
  void classify(clang::QualType T, uint64_t OffsetBase, Class &Lo, Class &Hi,
                bool isNamedArg) const;

  mlir::Type GetSSETypeAtOffset(mlir::Type CIRType, unsigned CIROffset,
                                clang::QualType SourceTy,
                                unsigned SourceOffset) const;

  cir::ABIArgInfo classifyReturnType(QualType RetTy) const;

  cir::ABIArgInfo classifyArgumentType(clang::QualType Ty, unsigned freeIntRegs,
                                       unsigned &neededInt, unsigned &neededSSE,
                                       bool isNamedArg) const;

  mlir::Type GetINTEGERTypeAtOffset(mlir::Type CIRType, unsigned CIROffset,
                                    QualType SourceTy,
                                    unsigned SourceOffset) const;

  /// getIndirectResult - Give a source type \arg Ty, return a suitable result
  /// such that the argument will be passed in memory.
  ///
  /// \param freeIntRegs - The number of free integer registers remaining
  /// available.
  cir::ABIArgInfo getIndirectResult(QualType Ty, unsigned freeIntRegs) const;
};

class X86_64TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  X86_64TargetCIRGenInfo(CIRGenTypes &CGT, X86AVXABILevel AVXLevel)
      : TargetCIRGenInfo(std::make_unique<X86_64ABIInfo>(CGT, AVXLevel)) {}
};
} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createX86_64TargetCIRGenInfo(CIRGenTypes &CGT,
                                            X86AVXABILevel AVXLevel) {
  return std::make_unique<X86_64TargetCIRGenInfo>(CGT, AVXLevel);
}

//===----------------------------------------------------------------------===//
// Base ABI and target codegen info implementation common between SPIR and
// SPIR-V.
//===----------------------------------------------------------------------===//

namespace {
class CommonSPIRABIInfo : public DefaultABIInfo {
public:
  CommonSPIRABIInfo(CIRGenTypes &CGT) : DefaultABIInfo(CGT) {}
};

class SPIRVABIInfo : public CommonSPIRABIInfo {
public:
  SPIRVABIInfo(CIRGenTypes &CGT) : CommonSPIRABIInfo(CGT) {}

private:
  cir::ABIArgInfo classifyKernelArgumentType(QualType Ty) const {
    assert(!getContext().getLangOpts().CUDAIsDevice && "NYI");
    return classifyArgumentType(Ty);
  }
};

class CommonSPIRTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  CommonSPIRTargetCIRGenInfo(std::unique_ptr<ABIInfo> ABIInfo)
      : TargetCIRGenInfo(std::move(ABIInfo)) {}

  cir::AddressSpace getCIRAllocaAddressSpace() const override {
    return cir::AddressSpace::OffloadPrivate;
  }

  cir::CallingConv getOpenCLKernelCallingConv() const override {
    return cir::CallingConv::SpirKernel;
  }
};

class SPIRVTargetCIRGenInfo : public CommonSPIRTargetCIRGenInfo {
public:
  SPIRVTargetCIRGenInfo(CIRGenTypes &CGT)
      : CommonSPIRTargetCIRGenInfo(std::make_unique<SPIRVABIInfo>(CGT)) {}

  void setCUDAKernelCallingConvention(const FunctionType *&ft) const override {
    llvm_unreachable("NYI");
  }
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createSPIRVTargetCIRGenInfo(CIRGenTypes &CGT) {
  return std::make_unique<SPIRVTargetCIRGenInfo>(CGT);
}

//===----------------------------------------------------------------------===//
// NVPTX ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class NVPTXABIInfo : public ABIInfo {
public:
  NVPTXABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}

  cir::ABIArgInfo classifyReturnType(QualType retTy) const;
  cir::ABIArgInfo classifyArgumentType(QualType ty) const;
};

class NVPTXTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  NVPTXTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<NVPTXABIInfo>(cgt)) {}
  mlir::Type getCUDADeviceBuiltinSurfaceDeviceType() const override {
    // On the device side, surface reference is represented as an object handle
    // in 64-bit integer.
    return cir::IntType::get(&getABIInfo().CGT.getMLIRContext(), 64, true);
  }
  mlir::Type getCUDADeviceBuiltinTextureDeviceType() const override {
    // On the device side, texture reference is represented as an object handle
    // in 64-bit integer.
    return cir::IntType::get(&getABIInfo().CGT.getMLIRContext(), 64, true);
  }
  void setTargetAttributes(const clang::Decl *decl, mlir::Operation *global,
                           CIRGenModule &cgm) const override {
    if (clang::isa_and_nonnull<clang::VarDecl>(decl)) {
      assert(!cir::MissingFeatures::emitNVVMMetadata());
      return;
    }

    if (const auto *fd = clang::dyn_cast_or_null<clang::FunctionDecl>(decl)) {
      cir::FuncOp func = mlir::cast<cir::FuncOp>(global);
      if (func.isDeclaration())
        return;

      if (cgm.getLangOpts().CUDA) {
        if (fd->hasAttr<CUDAGlobalAttr>()) {
          func.setCallingConv(cir::CallingConv::PTXKernel);

          // In LLVM we should create metadata like:
          //    !{<func-ref>, metadata !"kernel", i32 1}
          assert(!cir::MissingFeatures::emitNVVMMetadata());
        }
      }

      if (fd->getAttr<CUDALaunchBoundsAttr>())
        llvm_unreachable("NYI");
    }
  }
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createNVPTXTargetCIRGenInfo(CIRGenTypes &CGT) {
  return std::make_unique<NVPTXTargetCIRGenInfo>(CGT);
}

//===----------------------------------------------------------------------===//
// AMDGPU ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AMDGPUABIInfo : public ABIInfo {
public:
  AMDGPUABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}

  cir::ABIArgInfo classifyReturnType(QualType retTy) const;
  cir::ABIArgInfo classifyArgumentType(QualType ty) const;
};

class AMDGPUTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  AMDGPUTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<AMDGPUABIInfo>(cgt)) {}

  void setCUDAKernelCallingConvention(const FunctionType *&ft) const override {
    llvm_unreachable("NYI");
  }
};

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createAMDGPUTargetCIRGenInfo(CIRGenTypes &CGT) {
  return std::make_unique<AMDGPUTargetCIRGenInfo>(CGT);
}

// TODO(cir): remove the attribute once this gets used.
LLVM_ATTRIBUTE_UNUSED
static bool classifyReturnType(const CIRGenCXXABI &CXXABI,
                               CIRGenFunctionInfo &FI, const ABIInfo &Info) {
  QualType Ty = FI.getReturnType();

  assert(!Ty->getAs<RecordType>() && "RecordType returns NYI");

  return CXXABI.classifyReturnType(FI);
}

CIRGenCXXABI &ABIInfo::getCXXABI() const { return CGT.getCXXABI(); }

cir::VectorType
ABIInfo::getOptimalVectorMemoryType(cir::VectorType T,
                                    const clang::LangOptions &Opt) const {
  if (T.getSize() == 3 && !Opt.PreserveVec3Type) {
    return cir::VectorType::get(T.getElementType(), 4);
  }
  return T;
}

clang::ASTContext &ABIInfo::getContext() const { return CGT.getContext(); }

cir::ABIArgInfo X86_64ABIInfo::getIndirectResult(QualType Ty,
                                                 unsigned freeIntRegs) const {
  assert(false && "NYI");
}

/// GetINTEGERTypeAtOffset - The ABI specifies that a value should be passed in
/// an 8-byte GPR. This means that we either have a scalar or we are talking
/// about the high or low part of an up-to-16-byte struct. This routine picks
/// the best CIR type to represent this, which may be i64 or may be anything
/// else that the backend will pass in a GPR that works better (e.g. i8, %foo*,
/// etc).
///
/// PrefType is a CIR type that corresponds to (part of) the IR type for the
/// source type. CIROffset is an offset in bytes into the CIR type taht the
/// 8-byte value references. PrefType may be null.
///
/// SourceTy is the source-level type for the entire argument. SourceOffset is
/// an offset into this that we're processing (which is always either 0 or 8).
///
mlir::Type X86_64ABIInfo::GetINTEGERTypeAtOffset(mlir::Type CIRType,
                                                 unsigned CIROffset,
                                                 QualType SourceTy,
                                                 unsigned SourceOffset) const {
  // TODO: entirely stubbed out
  assert(CIROffset == 0 && "NYI");
  assert(SourceOffset == 0 && "NYI");
  return CIRType;
}

cir::ABIArgInfo X86_64ABIInfo::classifyArgumentType(QualType Ty,
                                                    unsigned int freeIntRegs,
                                                    unsigned int &neededInt,
                                                    unsigned int &neededSSE,
                                                    bool isNamedArg) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  X86_64ABIInfo::Class Lo, Hi;
  classify(Ty, 0, Lo, Hi, isNamedArg);

  // Check some invariants
  // FIXME: Enforce these by construction.
  assert((Hi != Class::Memory || Lo == Class::Memory) &&
         "Invalid memory classification.");
  assert((Hi != Class::SSEUp || Lo == Class::SSE) &&
         "Invalid SSEUp classification.");

  neededInt = 0;
  neededSSE = 0;
  mlir::Type ResType = nullptr;
  switch (Lo) {
  default:
    assert(false && "NYI");

  // AMD64-ABI 3.2.3p3: Rule 2. If the class is INTEGER, the next available
  // register of the sequence %rdi, %rsi, %rdx, %rcx, %r8 and %r9 is used.
  case Class::Integer:
    ++neededInt;

    // Pick an 8-byte type based on the preferred type.
    ResType = GetINTEGERTypeAtOffset(CGT.convertType(Ty), 0, Ty, 0);

    // If we have a sign or zero extended integer, make sure to return Extend so
    // that the parameter gets the right LLVM IR attributes.
    if (Hi == Class::NoClass && mlir::isa<cir::IntType>(ResType)) {
      assert(!Ty->getAs<EnumType>() && "NYI");
      if (Ty->isSignedIntegerOrEnumerationType() &&
          isPromotableIntegerTypeForABI(Ty))
        return cir::ABIArgInfo::getExtend(Ty);
    }

    break;

    // AMD64-ABI 3.2.3p3: Rule 3. If the class is SSE, the next available SSE
    // register is used, the registers are taken in the order from %xmm0 to
    // %xmm7.
  case Class::SSE: {
    mlir::Type CIRType = CGT.convertType(Ty);
    ResType = GetSSETypeAtOffset(CIRType, 0, Ty, 0);
    ++neededSSE;
    break;
  }
  }

  mlir::Type HighPart = nullptr;
  switch (Hi) {
  default:
    assert(false && "NYI");
  case Class::NoClass:
    break;
  }

  assert(!HighPart && "NYI");

  return cir::ABIArgInfo::getDirect(ResType);
}

// Skeleton only. Implement when used in TargetLower stage.
cir::ABIArgInfo NVPTXABIInfo::classifyReturnType(QualType retTy) const {
  llvm_unreachable("not yet implemented");
}

cir::ABIArgInfo NVPTXABIInfo::classifyArgumentType(QualType ty) const {
  llvm_unreachable("not yet implemented");
}

// Skeleton only. Implement when used in TargetLower stage.
cir::ABIArgInfo AMDGPUABIInfo::classifyReturnType(QualType retTy) const {
  llvm_unreachable("not yet implemented");
}

cir::ABIArgInfo AMDGPUABIInfo::classifyArgumentType(QualType ty) const {
  llvm_unreachable("not yet implemented");
}

ABIInfo::~ABIInfo() {}

bool ABIInfo::isPromotableIntegerTypeForABI(QualType Ty) const {
  if (getContext().isPromotableIntegerType(Ty))
    return true;

  assert(!Ty->getAs<BitIntType>() && "NYI");

  return false;
}

void X86_64ABIInfo::classify(QualType Ty, uint64_t OffsetBase, Class &Lo,
                             Class &Hi, bool isNamedArg) const {
  // FIXME: This code can be simplified by introducing a simple value class for
  // Class pairs with appropriate constructor methods for the various
  // situations.

  // FIXME: Some of the split computations are wrong; unaligned vectors
  // shouldn't be passed in registers for example, so there is no chance they
  // can straddle an eightbyte. Verify & simplify.

  Lo = Hi = Class::NoClass;
  Class &Current = OffsetBase < 64 ? Lo : Hi;
  Current = Class::Memory;

  if (const auto *BT = Ty->getAs<BuiltinType>()) {
    BuiltinType::Kind k = BT->getKind();
    if (k == BuiltinType::Void) {
      Current = Class::NoClass;
    } else if (k == BuiltinType::Int128 || k == BuiltinType::UInt128) {
      assert(false && "NYI");
      Lo = Class::Integer;
      Hi = Class::Integer;
    } else if (k >= BuiltinType::Bool && k <= BuiltinType::LongLong) {
      Current = Class::Integer;
    } else if (k == BuiltinType::Float || k == BuiltinType::Double ||
               k == BuiltinType::Float16) {
      Current = Class::SSE;
    } else if (k == BuiltinType::LongDouble) {
      assert(false && "NYI");
    } else
      assert(false &&
             "Only void and Integer supported so far for builtin types");
    // FIXME: _Decimal32 and _Decimal64 are SSE.
    // FIXME: _float128 and _Decimal128 are (SSE, SSEUp).
    return;
  }

  assert(!Ty->getAs<EnumType>() && "Enums NYI");
  if (Ty->hasPointerRepresentation()) {
    Current = Class::Integer;
    return;
  }

  assert(false && "Nothing else implemented yet");
}

/// GetSSETypeAtOffset - Return a type that will be passed by the backend in the
/// low 8 bytes of an XMM register, corresponding to the SSE class.
mlir::Type X86_64ABIInfo::GetSSETypeAtOffset(mlir::Type CIRType,
                                             unsigned int CIROffset,
                                             clang::QualType SourceTy,
                                             unsigned int SourceOffset) const {
  // TODO: entirely stubbed out
  assert(CIROffset == 0 && "NYI");
  assert(SourceOffset == 0 && "NYI");
  return CIRType;
}

cir::ABIArgInfo X86_64ABIInfo::classifyReturnType(QualType RetTy) const {
  // AMD64-ABI 3.2.3p4: Rule 1. Classify the return type with the classification
  // algorithm.
  X86_64ABIInfo::Class Lo, Hi;
  classify(RetTy, 0, Lo, Hi, /*isNamedArg*/ true);

  // Check some invariants.
  assert((Hi != Class::Memory || Lo == Class::Memory) &&
         "Invalid memory classification.");
  assert((Hi != Class::SSEUp || Lo == Class::SSE) &&
         "Invalid SSEUp classification.");

  mlir::Type ResType = nullptr;
  assert(Lo == Class::NoClass || Lo == Class::Integer ||
         Lo == Class::SSE && "Only NoClass and Integer supported so far");

  switch (Lo) {
  case Class::NoClass:
    assert(Hi == Class::NoClass && "Only NoClass supported so far for Hi");
    return cir::ABIArgInfo::getIgnore();

  // AMD64-ABI 3.2.3p4: Rule 3. If the class is INTEGER, the next available
  // register of the sequence %rax, %rdx is used.
  case Class::Integer:
    ResType = GetINTEGERTypeAtOffset(CGT.convertType(RetTy), 0, RetTy, 0);

    // If we have a sign or zero extended integer, make sure to return Extend so
    // that the parameter gets the right LLVM IR attributes.
    // TODO: extend the above consideration to MLIR
    if (Hi == Class::NoClass && mlir::isa<cir::IntType>(ResType)) {
      // Treat an enum type as its underlying type.
      if (const auto *EnumTy = RetTy->getAs<EnumType>())
        RetTy = EnumTy->getDecl()->getIntegerType();

      if (RetTy->isIntegralOrEnumerationType() &&
          isPromotableIntegerTypeForABI(RetTy)) {
        return cir::ABIArgInfo::getExtend(RetTy);
      }
    }
    break;

    // AMD64-ABI 3.2.3p4: Rule 4. If the class is SSE, the next available SSE
    // register of the sequence %xmm0, %xmm1 is used.
  case Class::SSE:
    ResType = GetSSETypeAtOffset(CGT.convertType(RetTy), 0, RetTy, 0);
    break;

  default:
    llvm_unreachable("NYI");
  }

  mlir::Type HighPart = nullptr;

  if (HighPart)
    assert(false && "NYI");

  return cir::ABIArgInfo::getDirect(ResType);
}

clang::LangAS
TargetCIRGenInfo::getGlobalVarAddressSpace(CIRGenModule &CGM,
                                           const clang::VarDecl *D) const {
  assert(!CGM.getLangOpts().OpenCL &&
         !(CGM.getLangOpts().CUDA && CGM.getLangOpts().CUDAIsDevice) &&
         "Address space agnostic languages only");
  return D ? D->getType().getAddressSpace() : LangAS::Default;
}

mlir::Value TargetCIRGenInfo::performAddrSpaceCast(
    CIRGenFunction &CGF, mlir::Value Src, cir::AddressSpace SrcAddr,
    cir::AddressSpace DestAddr, mlir::Type DestTy, bool IsNonNull) const {
  // Since target may map different address spaces in AST to the same address
  // space, an address space conversion may end up as a bitcast.
  if (auto globalOp = Src.getDefiningOp<cir::GlobalOp>())
    llvm_unreachable("Global ops addrspace cast NYI");
  // Try to preserve the source's name to make IR more readable.
  return CGF.getBuilder().createAddrSpaceCast(Src, DestTy);
}
