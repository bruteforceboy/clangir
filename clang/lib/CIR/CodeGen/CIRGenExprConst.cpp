//===---- CIRGenExprCst.cpp - Emit LLVM Code from Constant Expressions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Constant Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//
#include "Address.h"
#include "CIRGenCXXABI.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Specifiers.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

using namespace clang;
using namespace clang::CIRGen;

//===----------------------------------------------------------------------===//
//                            ConstantAggregateBuilder
//===----------------------------------------------------------------------===//

namespace {
class ConstExprEmitter;

static mlir::TypedAttr computePadding(CIRGenModule &CGM, CharUnits size) {
  auto eltTy = CGM.UCharTy;
  auto arSize = size.getQuantity();
  auto &bld = CGM.getBuilder();
  if (size > CharUnits::One()) {
    SmallVector<mlir::Attribute, 4> elts(arSize, cir::ZeroAttr::get(eltTy));
    return bld.getConstArray(mlir::ArrayAttr::get(bld.getContext(), elts),
                             cir::ArrayType::get(eltTy, arSize));
  } else {
    return cir::ZeroAttr::get(eltTy);
  }
}

static mlir::Attribute
emitArrayConstant(CIRGenModule &CGM, mlir::Type DesiredType,
                  mlir::Type CommonElementType, unsigned ArrayBound,
                  SmallVectorImpl<mlir::TypedAttr> &Elements,
                  mlir::TypedAttr Filler);

struct ConstantAggregateBuilderUtils {
  CIRGenModule &CGM;
  cir::CIRDataLayout dataLayout;

  ConstantAggregateBuilderUtils(CIRGenModule &CGM)
      : CGM(CGM), dataLayout{CGM.getModule()} {}

  CharUnits getAlignment(const mlir::TypedAttr C) const {
    return CharUnits::fromQuantity(
        dataLayout.getAlignment(C.getType(), /*useABI=*/true));
  }

  CharUnits getSize(mlir::Type Ty) const {
    return CharUnits::fromQuantity(dataLayout.getTypeAllocSize(Ty));
  }

  CharUnits getSize(const mlir::TypedAttr C) const {
    return getSize(C.getType());
  }

  mlir::TypedAttr getPadding(CharUnits size) const {
    return computePadding(CGM, size);
  }

  mlir::Attribute getZeroes(CharUnits ZeroSize) const {
    llvm_unreachable("NYI");
  }
};

/// Incremental builder for an mlir::TypedAttr holding a record or array
/// constant.
class ConstantAggregateBuilder : private ConstantAggregateBuilderUtils {
  /// The elements of the constant. These two arrays must have the same size;
  /// Offsets[i] describes the offset of Elems[i] within the constant. The
  /// elements are kept in increasing offset order, and we ensure that there
  /// is no overlap: Offsets[i+1] >= Offsets[i] + getSize(Elemes[i]).
  ///
  /// This may contain explicit padding elements (in order to create a
  /// natural layout), but need not. Gaps between elements are implicitly
  /// considered to be filled with undef.
  llvm::SmallVector<mlir::Attribute, 32> Elems;
  llvm::SmallVector<CharUnits, 32> Offsets;

  /// The size of the constant (the maximum end offset of any added element).
  /// May be larger than the end of Elems.back() if we split the last element
  /// and removed some trailing undefs.
  CharUnits Size = CharUnits::Zero();

  /// This is true only if laying out Elems in order as the elements of a
  /// non-packed LLVM struct will give the correct layout.
  bool NaturalLayout = true;

  bool split(size_t Index, CharUnits Hint);
  std::optional<size_t> splitAt(CharUnits Pos);

  static mlir::Attribute
  buildFrom(CIRGenModule &CGM, ArrayRef<mlir::Attribute> Elems,
            ArrayRef<CharUnits> Offsets, CharUnits StartOffset, CharUnits Size,
            bool NaturalLayout, mlir::Type DesiredTy, bool AllowOversized);

public:
  ConstantAggregateBuilder(CIRGenModule &CGM)
      : ConstantAggregateBuilderUtils(CGM) {}

  /// Update or overwrite the value starting at \p Offset with \c C.
  ///
  /// \param AllowOverwrite If \c true, this constant might overwrite (part of)
  ///        a constant that has already been added. This flag is only used to
  ///        detect bugs.
  bool add(mlir::Attribute C, CharUnits Offset, bool AllowOverwrite);

  /// Update or overwrite the bits starting at \p OffsetInBits with \p Bits.
  bool addBits(llvm::APInt Bits, uint64_t OffsetInBits, bool AllowOverwrite);

  /// Attempt to condense the value starting at \p Offset to a constant of type
  /// \p DesiredTy.
  void condense(CharUnits Offset, mlir::Type DesiredTy);

  /// Produce a constant representing the entire accumulated value, ideally of
  /// the specified type. If \p AllowOversized, the constant might be larger
  /// than implied by \p DesiredTy (eg, if there is a flexible array member).
  /// Otherwise, the constant will be of exactly the same size as \p DesiredTy
  /// even if we can't represent it as that type.
  mlir::Attribute build(mlir::Type DesiredTy, bool AllowOversized) const {
    return buildFrom(CGM, Elems, Offsets, CharUnits::Zero(), Size,
                     NaturalLayout, DesiredTy, AllowOversized);
  }
};

template <typename Container, typename Range = std::initializer_list<
                                  typename Container::value_type>>
static void replace(Container &C, size_t BeginOff, size_t EndOff, Range Vals) {
  assert(BeginOff <= EndOff && "invalid replacement range");
  llvm::replace(C, C.begin() + BeginOff, C.begin() + EndOff, Vals);
}

bool ConstantAggregateBuilder::add(mlir::Attribute A, CharUnits Offset,
                                   bool AllowOverwrite) {
  // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
  mlir::TypedAttr C = mlir::dyn_cast<mlir::TypedAttr>(A);
  assert(C && "expected typed attribute");
  // Common case: appending to a layout.
  if (Offset >= Size) {
    CharUnits Align = getAlignment(C);
    CharUnits AlignedSize = Size.alignTo(Align);
    if (AlignedSize > Offset || Offset.alignTo(Align) != Offset)
      NaturalLayout = false;
    else if (AlignedSize < Offset) {
      Elems.push_back(getPadding(Offset - Size));
      Offsets.push_back(Size);
    }
    Elems.push_back(C);
    Offsets.push_back(Offset);
    Size = Offset + getSize(C);
    return true;
  }

  // Uncommon case: constant overlaps what we've already created.
  std::optional<size_t> FirstElemToReplace = splitAt(Offset);
  if (!FirstElemToReplace)
    return false;

  CharUnits CSize = getSize(C);
  std::optional<size_t> LastElemToReplace = splitAt(Offset + CSize);
  if (!LastElemToReplace)
    return false;

  assert((FirstElemToReplace == LastElemToReplace || AllowOverwrite) &&
         "unexpectedly overwriting field");

  replace(Elems, *FirstElemToReplace, *LastElemToReplace, {C});
  replace(Offsets, *FirstElemToReplace, *LastElemToReplace, {Offset});
  Size = std::max(Size, Offset + CSize);
  NaturalLayout = false;
  return true;
}

bool ConstantAggregateBuilder::addBits(llvm::APInt Bits, uint64_t OffsetInBits,
                                       bool AllowOverwrite) {
  const ASTContext &astContext = CGM.getASTContext();
  const uint64_t CharWidth = CGM.getASTContext().getCharWidth();
  auto charTy = CGM.getBuilder().getUIntNTy(CharWidth);
  // Offset of where we want the first bit to go within the bits of the
  // current char.
  unsigned OffsetWithinChar = OffsetInBits % CharWidth;

  // We split bit-fields up into individual bytes. Walk over the bytes and
  // update them.
  for (CharUnits OffsetInChars =
           astContext.toCharUnitsFromBits(OffsetInBits - OffsetWithinChar);
       /**/; ++OffsetInChars) {
    // Number of bits we want to fill in this char.
    unsigned WantedBits =
        std::min((uint64_t)Bits.getBitWidth(), CharWidth - OffsetWithinChar);

    // Get a char containing the bits we want in the right places. The other
    // bits have unspecified values.
    llvm::APInt BitsThisChar = Bits;
    if (BitsThisChar.getBitWidth() < CharWidth)
      BitsThisChar = BitsThisChar.zext(CharWidth);
    if (CGM.getDataLayout().isBigEndian()) {
      // Figure out how much to shift by. We may need to left-shift if we have
      // less than one byte of Bits left.
      int Shift = Bits.getBitWidth() - CharWidth + OffsetWithinChar;
      if (Shift > 0)
        BitsThisChar.lshrInPlace(Shift);
      else if (Shift < 0)
        BitsThisChar = BitsThisChar.shl(-Shift);
    } else {
      BitsThisChar = BitsThisChar.shl(OffsetWithinChar);
    }
    if (BitsThisChar.getBitWidth() > CharWidth)
      BitsThisChar = BitsThisChar.trunc(CharWidth);

    if (WantedBits == CharWidth) {
      // Got a full byte: just add it directly.
      add(cir::IntAttr::get(charTy, BitsThisChar), OffsetInChars,
          AllowOverwrite);
    } else {
      // Partial byte: update the existing integer if there is one. If we
      // can't split out a 1-CharUnit range to update, then we can't add
      // these bits and fail the entire constant emission.
      std::optional<size_t> FirstElemToUpdate = splitAt(OffsetInChars);
      if (!FirstElemToUpdate)
        return false;
      std::optional<size_t> LastElemToUpdate =
          splitAt(OffsetInChars + CharUnits::One());
      if (!LastElemToUpdate)
        return false;
      assert(*LastElemToUpdate - *FirstElemToUpdate < 2 &&
             "should have at most one element covering one byte");

      // Figure out which bits we want and discard the rest.
      llvm::APInt UpdateMask(CharWidth, 0);
      if (CGM.getDataLayout().isBigEndian())
        UpdateMask.setBits(CharWidth - OffsetWithinChar - WantedBits,
                           CharWidth - OffsetWithinChar);
      else
        UpdateMask.setBits(OffsetWithinChar, OffsetWithinChar + WantedBits);
      BitsThisChar &= UpdateMask;
      bool isNull = false;
      if (*FirstElemToUpdate < Elems.size()) {
        auto firstEltToUpdate =
            dyn_cast<cir::IntAttr>(Elems[*FirstElemToUpdate]);
        isNull = firstEltToUpdate && firstEltToUpdate.isNullValue();
      }

      if (*FirstElemToUpdate == *LastElemToUpdate || isNull) {
        // All existing bits are either zero or undef.
        add(cir::IntAttr::get(charTy, BitsThisChar), OffsetInChars,
            /*AllowOverwrite*/ true);
      } else {
        cir::IntAttr CI = dyn_cast<cir::IntAttr>(Elems[*FirstElemToUpdate]);
        // In order to perform a partial update, we need the existing bitwise
        // value, which we can only extract for a constant int.
        // auto *CI = dyn_cast<llvm::ConstantInt>(ToUpdate);
        if (!CI)
          return false;
        // Because this is a 1-CharUnit range, the constant occupying it must
        // be exactly one CharUnit wide.
        assert(CI.getBitWidth() == CharWidth && "splitAt failed");
        assert((!(CI.getValue() & UpdateMask) || AllowOverwrite) &&
               "unexpectedly overwriting bitfield");
        BitsThisChar |= (CI.getValue() & ~UpdateMask);
        Elems[*FirstElemToUpdate] = cir::IntAttr::get(charTy, BitsThisChar);
      }
    }

    // Stop if we've added all the bits.
    if (WantedBits == Bits.getBitWidth())
      break;

    // Remove the consumed bits from Bits.
    if (!CGM.getDataLayout().isBigEndian())
      Bits.lshrInPlace(WantedBits);
    Bits = Bits.trunc(Bits.getBitWidth() - WantedBits);

    // The remanining bits go at the start of the following bytes.
    OffsetWithinChar = 0;
  }

  return true;
}

/// Returns a position within Elems and Offsets such that all elements
/// before the returned index end before Pos and all elements at or after
/// the returned index begin at or after Pos. Splits elements as necessary
/// to ensure this. Returns None if we find something we can't split.
std::optional<size_t> ConstantAggregateBuilder::splitAt(CharUnits Pos) {
  if (Pos >= Size)
    return Offsets.size();

  while (true) {
    auto FirstAfterPos = llvm::upper_bound(Offsets, Pos);
    if (FirstAfterPos == Offsets.begin())
      return 0;

    // If we already have an element starting at Pos, we're done.
    size_t LastAtOrBeforePosIndex = FirstAfterPos - Offsets.begin() - 1;
    if (Offsets[LastAtOrBeforePosIndex] == Pos)
      return LastAtOrBeforePosIndex;

    // We found an element starting before Pos. Check for overlap.
    // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
    mlir::TypedAttr C =
        mlir::dyn_cast<mlir::TypedAttr>(Elems[LastAtOrBeforePosIndex]);
    assert(C && "expected typed attribute");
    if (Offsets[LastAtOrBeforePosIndex] + getSize(C) <= Pos)
      return LastAtOrBeforePosIndex + 1;

    // Try to decompose it into smaller constants.
    if (!split(LastAtOrBeforePosIndex, Pos))
      return std::nullopt;
  }
}

/// Split the constant at index Index, if possible. Return true if we did.
/// Hint indicates the location at which we'd like to split, but may be
/// ignored.
bool ConstantAggregateBuilder::split(size_t Index, CharUnits Hint) {
  llvm_unreachable("NYI");
}

mlir::Attribute ConstantAggregateBuilder::buildFrom(
    CIRGenModule &CGM, ArrayRef<mlir::Attribute> Elems,
    ArrayRef<CharUnits> Offsets, CharUnits StartOffset, CharUnits Size,
    bool NaturalLayout, mlir::Type DesiredTy, bool AllowOversized) {
  ConstantAggregateBuilderUtils Utils(CGM);

  if (Elems.empty())
    return cir::UndefAttr::get(DesiredTy);

  auto Offset = [&](size_t I) { return Offsets[I] - StartOffset; };

  // If we want an array type, see if all the elements are the same type and
  // appropriately spaced.
  if (auto aty = mlir::dyn_cast<cir::ArrayType>(DesiredTy)) {
    llvm_unreachable("NYI");
  }

  // The size of the constant we plan to generate. This is usually just the size
  // of the initialized type, but in AllowOversized mode (i.e. flexible array
  // init), it can be larger.
  CharUnits DesiredSize = Utils.getSize(DesiredTy);
  if (Size > DesiredSize) {
    assert(AllowOversized && "Elems are oversized");
    DesiredSize = Size;
  }

  // The natural alignment of an unpacked CIR record with the given elements.
  CharUnits Align = CharUnits::One();
  for (auto e : Elems) {
    // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
    auto C = mlir::dyn_cast<mlir::TypedAttr>(e);
    assert(C && "expected typed attribute");
    Align = std::max(Align, Utils.getAlignment(C));
  }

  // The natural size of an unpacked LLVM struct with the given elements.
  CharUnits AlignedSize = Size.alignTo(Align);

  bool Packed = false;
  bool Padded = false;
  ArrayRef<mlir::Attribute> UnpackedElems = Elems;

  llvm::SmallVector<mlir::Attribute, 32> UnpackedElemStorage;
  if (DesiredSize < AlignedSize || DesiredSize.alignTo(Align) != DesiredSize) {
    NaturalLayout = false;
    Packed = true;
  } else if (DesiredSize > AlignedSize) {
    // The natural layout would be too small. Add padding to fix it. (This
    // is ignored if we choose a packed layout.)
    UnpackedElemStorage.assign(UnpackedElems.begin(), UnpackedElems.end());
    UnpackedElemStorage.push_back(Utils.getPadding(DesiredSize - Size));
    UnpackedElems = UnpackedElemStorage;
  }

  // If we don't have a natural layout, insert padding as necessary.
  // As we go, double-check to see if we can actually just emit Elems
  // as a non-packed record and do so opportunistically if possible.
  llvm::SmallVector<mlir::Attribute, 32> PackedElems;
  if (!NaturalLayout) {
    CharUnits SizeSoFar = CharUnits::Zero();
    for (size_t I = 0; I != Elems.size(); ++I) {
      mlir::TypedAttr C = mlir::dyn_cast<mlir::TypedAttr>(Elems[I]);
      assert(C && "expected typed attribute");

      CharUnits Align = Utils.getAlignment(C);
      CharUnits NaturalOffset = SizeSoFar.alignTo(Align);
      CharUnits DesiredOffset = Offset(I);
      assert(DesiredOffset >= SizeSoFar && "elements out of order");

      if (DesiredOffset != NaturalOffset)
        Packed = true;
      if (DesiredOffset != SizeSoFar)
        PackedElems.push_back(Utils.getPadding(DesiredOffset - SizeSoFar));
      PackedElems.push_back(Elems[I]);
      SizeSoFar = DesiredOffset + Utils.getSize(C);
    }
    // If we're using the packed layout, pad it out to the desired size if
    // necessary.
    if (Packed) {
      assert(SizeSoFar <= DesiredSize &&
             "requested size is too small for contents");

      if (SizeSoFar < DesiredSize)
        PackedElems.push_back(Utils.getPadding(DesiredSize - SizeSoFar));
    }
  }

  auto &builder = CGM.getBuilder();
  auto arrAttr = mlir::ArrayAttr::get(builder.getContext(),
                                      Packed ? PackedElems : UnpackedElems);

  auto strType = builder.getCompleteRecordType(arrAttr, Packed);
  if (auto desired = dyn_cast<cir::RecordType>(DesiredTy))
    if (desired.isLayoutIdentical(strType))
      strType = desired;

  return builder.getConstRecordOrZeroAttr(arrAttr, Packed, Padded, strType);
}

void ConstantAggregateBuilder::condense(CharUnits Offset,
                                        mlir::Type DesiredTy) {
  CharUnits Size = getSize(DesiredTy);

  std::optional<size_t> FirstElemToReplace = splitAt(Offset);
  if (!FirstElemToReplace)
    return;
  size_t First = *FirstElemToReplace;

  std::optional<size_t> LastElemToReplace = splitAt(Offset + Size);
  if (!LastElemToReplace)
    return;
  size_t Last = *LastElemToReplace;

  size_t Length = Last - First;
  if (Length == 0)
    return;

  // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
  mlir::TypedAttr C = mlir::dyn_cast<mlir::TypedAttr>(Elems[First]);
  assert(C && "expected typed attribute");
  if (Length == 1 && Offsets[First] == Offset && getSize(C) == Size) {
    // Re-wrap single element records if necessary. Otherwise, leave any single
    // element constant of the right size alone even if it has the wrong type.
    llvm_unreachable("NYI");
  }

  mlir::Attribute Replacement = buildFrom(
      CGM, ArrayRef(Elems).slice(First, Length),
      ArrayRef(Offsets).slice(First, Length), Offset, getSize(DesiredTy),
      /*known to have natural layout=*/false, DesiredTy, false);
  replace(Elems, First, Last, {Replacement});
  replace(Offsets, First, Last, {Offset});
}

//===----------------------------------------------------------------------===//
//                            ConstRecordBuilder
//===----------------------------------------------------------------------===//

class ConstRecordBuilder {
  CIRGenModule &CGM;
  ConstantEmitter &Emitter;
  ConstantAggregateBuilder &Builder;
  CharUnits StartOffset;

public:
  static mlir::Attribute BuildRecord(ConstantEmitter &Emitter,
                                     InitListExpr *ILE, QualType RecordTy);
  static mlir::Attribute BuildRecord(ConstantEmitter &Emitter,
                                     const APValue &Value, QualType ValTy);
  static bool UpdateRecord(ConstantEmitter &Emitter,
                           ConstantAggregateBuilder &Const, CharUnits Offset,
                           InitListExpr *Updater);

private:
  ConstRecordBuilder(ConstantEmitter &Emitter,
                     ConstantAggregateBuilder &Builder, CharUnits StartOffset)
      : CGM(Emitter.CGM), Emitter(Emitter), Builder(Builder),
        StartOffset(StartOffset) {}

  bool AppendField(const FieldDecl *Field, uint64_t FieldOffset,
                   mlir::Attribute InitExpr, bool AllowOverwrite = false);

  bool AppendBytes(CharUnits FieldOffsetInChars, mlir::Attribute InitCst,
                   bool AllowOverwrite = false);

  bool AppendBitField(const FieldDecl *Field, uint64_t FieldOffset,
                      cir::IntAttr InitExpr, bool AllowOverwrite = false);

  bool Build(InitListExpr *ILE, bool AllowOverwrite);
  bool Build(const APValue &Val, const RecordDecl *RD, bool IsPrimaryBase,
             const CXXRecordDecl *VTableClass, CharUnits BaseOffset);

  bool ApplyZeroInitPadding(const ASTRecordLayout &Layout, unsigned FieldNo,
                            const FieldDecl &Field, bool AllowOverwrite,
                            CharUnits &SizeSoFar, bool &ZeroFieldSize);

  bool ApplyZeroInitPadding(const ASTRecordLayout &Layout, bool AllowOverwrite,
                            CharUnits SizeSoFar);

  mlir::Attribute Finalize(QualType Ty);
};

bool ConstRecordBuilder::AppendField(const FieldDecl *Field,
                                     uint64_t FieldOffset,
                                     mlir::Attribute InitCst,
                                     bool AllowOverwrite) {
  const ASTContext &astContext = CGM.getASTContext();

  CharUnits FieldOffsetInChars = astContext.toCharUnitsFromBits(FieldOffset);

  return AppendBytes(FieldOffsetInChars, InitCst, AllowOverwrite);
}

bool ConstRecordBuilder::AppendBytes(CharUnits FieldOffsetInChars,
                                     mlir::Attribute InitCst,
                                     bool AllowOverwrite) {
  return Builder.add(InitCst, StartOffset + FieldOffsetInChars, AllowOverwrite);
}

bool ConstRecordBuilder::AppendBitField(const FieldDecl *Field,
                                        uint64_t FieldOffset, cir::IntAttr CI,
                                        bool AllowOverwrite) {
  const auto &RL = CGM.getTypes().getCIRGenRecordLayout(Field->getParent());
  const auto &Info = RL.getBitFieldInfo(Field);
  llvm::APInt FieldValue = CI.getValue();

  // Promote the size of FieldValue if necessary
  // FIXME: This should never occur, but currently it can because initializer
  // constants are cast to bool, and because clang is not enforcing bitfield
  // width limits.
  if (Info.Size > FieldValue.getBitWidth())
    FieldValue = FieldValue.zext(Info.Size);

  // Truncate the size of FieldValue to the bit field size.
  if (Info.Size < FieldValue.getBitWidth())
    FieldValue = FieldValue.trunc(Info.Size);

  return Builder.addBits(FieldValue,
                         CGM.getASTContext().toBits(StartOffset) + FieldOffset,
                         AllowOverwrite);
}

static bool EmitDesignatedInitUpdater(ConstantEmitter &Emitter,
                                      ConstantAggregateBuilder &Const,
                                      CharUnits Offset, QualType Type,
                                      InitListExpr *Updater) {
  if (Type->isRecordType())
    return ConstRecordBuilder::UpdateRecord(Emitter, Const, Offset, Updater);

  auto CAT = Emitter.CGM.getASTContext().getAsConstantArrayType(Type);
  if (!CAT)
    return false;
  QualType ElemType = CAT->getElementType();
  CharUnits ElemSize = Emitter.CGM.getASTContext().getTypeSizeInChars(ElemType);
  mlir::Type ElemTy = Emitter.CGM.getTypes().convertTypeForMem(ElemType);

  mlir::Attribute FillC = nullptr;
  if (Expr *Filler = Updater->getArrayFiller()) {
    if (!isa<NoInitExpr>(Filler)) {
      llvm_unreachable("NYI");
    }
  }

  unsigned NumElementsToUpdate =
      FillC ? CAT->getSize().getZExtValue() : Updater->getNumInits();
  for (unsigned I = 0; I != NumElementsToUpdate; ++I, Offset += ElemSize) {
    Expr *Init = nullptr;
    if (I < Updater->getNumInits())
      Init = Updater->getInit(I);

    if (!Init && FillC) {
      if (!Const.add(FillC, Offset, true))
        return false;
    } else if (!Init || isa<NoInitExpr>(Init)) {
      continue;
    } else if (InitListExpr *ChildILE = dyn_cast<InitListExpr>(Init)) {
      if (!EmitDesignatedInitUpdater(Emitter, Const, Offset, ElemType,
                                     ChildILE))
        return false;
      // Attempt to reduce the array element to a single constant if necessary.
      Const.condense(Offset, ElemTy);
    } else {
      mlir::Attribute Val = Emitter.tryEmitPrivateForMemory(Init, ElemType);
      if (!Const.add(Val, Offset, true))
        return false;
    }
  }

  return true;
}

bool ConstRecordBuilder::Build(InitListExpr *ILE, bool AllowOverwrite) {
  RecordDecl *RD = ILE->getType()->castAs<clang::RecordType>()->getDecl();
  const ASTRecordLayout &Layout = CGM.getASTContext().getASTRecordLayout(RD);

  unsigned FieldNo = -1;
  unsigned ElementNo = 0;

  // Bail out if we have base classes. We could support these, but they only
  // arise in C++1z where we will have already constant folded most interesting
  // cases. FIXME: There are still a few more cases we can handle this way.
  if (auto *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    if (CXXRD->getNumBases())
      return false;

  const bool ZeroInitPadding = CGM.shouldZeroInitPadding();
  bool ZeroFieldSize = false;
  CharUnits SizeSoFar = CharUnits::Zero();

  for (FieldDecl *Field : RD->fields()) {
    ++FieldNo;

    // If this is a union, skip all the fields that aren't being initialized.
    if (RD->isUnion() &&
        !declaresSameEntity(ILE->getInitializedFieldInUnion(), Field))
      continue;

    // Don't emit anonymous bitfields.
    if (Field->isUnnamedBitField())
      continue;

    // Get the initializer.  A record can include fields without initializers,
    // we just use explicit null values for them.
    Expr *Init = nullptr;
    if (ElementNo < ILE->getNumInits())
      Init = ILE->getInit(ElementNo++);
    if (Init && isa<NoInitExpr>(Init))
      continue;

    // Zero-sized fields are not emitted, but their initializers may still
    // prevent emission of this record as a constant.
    if (Field->isZeroSize(CGM.getASTContext())) {
      if (Init->HasSideEffects(CGM.getASTContext()))
        return false;
      continue;
    }

    if (ZeroInitPadding &&
        !ApplyZeroInitPadding(Layout, FieldNo, *Field, AllowOverwrite,
                              SizeSoFar, ZeroFieldSize))
      return false;

    // When emitting a DesignatedInitUpdateExpr, a nested InitListExpr
    // represents additional overwriting of our current constant value, and not
    // a new constant to emit independently.
    if (AllowOverwrite &&
        (Field->getType()->isArrayType() || Field->getType()->isRecordType())) {
      if (auto *SubILE = dyn_cast<InitListExpr>(Init)) {
        CharUnits Offset = CGM.getASTContext().toCharUnitsFromBits(
            Layout.getFieldOffset(FieldNo));
        if (!EmitDesignatedInitUpdater(Emitter, Builder, StartOffset + Offset,
                                       Field->getType(), SubILE))
          return false;
        // If we split apart the field's value, try to collapse it down to a
        // single value now.
        llvm_unreachable("NYI");
        continue;
      }
    }

    mlir::Attribute EltInit;
    if (Init)
      EltInit = Emitter.tryEmitPrivateForMemory(Init, Field->getType());
    else
      EltInit = Emitter.emitNullForMemory(CGM.getLoc(ILE->getSourceRange()),
                                          Field->getType());

    if (!EltInit)
      return false;

    if (!Field->isBitField()) {
      // Handle non-bitfield members.
      if (!AppendField(Field, Layout.getFieldOffset(FieldNo), EltInit,
                       AllowOverwrite))
        return false;
      // After emitting a non-empty field with [[no_unique_address]], we may
      // need to overwrite its tail padding.
      if (Field->hasAttr<NoUniqueAddressAttr>())
        AllowOverwrite = true;
    } else {
      // Otherwise we have a bitfield.
      if (auto constInt = dyn_cast<cir::IntAttr>(EltInit)) {
        if (!AppendBitField(Field, Layout.getFieldOffset(FieldNo), constInt,
                            AllowOverwrite))
          return false;
      } else {
        // We are trying to initialize a bitfield with a non-trivial constant,
        // this must require run-time code.
        return false;
      }
    }
  }

  if (ZeroInitPadding &&
      !ApplyZeroInitPadding(Layout, AllowOverwrite, SizeSoFar))
    return false;

  return true;
}

namespace {
struct BaseInfo {
  BaseInfo(const CXXRecordDecl *Decl, CharUnits Offset, unsigned Index)
      : Decl(Decl), Offset(Offset), Index(Index) {}

  const CXXRecordDecl *Decl;
  CharUnits Offset;
  unsigned Index;

  bool operator<(const BaseInfo &O) const { return Offset < O.Offset; }
};
} // namespace

bool ConstRecordBuilder::Build(const APValue &Val, const RecordDecl *RD,
                               bool IsPrimaryBase,
                               const CXXRecordDecl *VTableClass,
                               CharUnits Offset) {
  const ASTRecordLayout &Layout = CGM.getASTContext().getASTRecordLayout(RD);
  if (const CXXRecordDecl *CD = dyn_cast<CXXRecordDecl>(RD)) {
    // Add a vtable pointer, if we need one and it hasn't already been added.
    if (Layout.hasOwnVFPtr()) {
      CIRGenBuilderTy &builder = CGM.getBuilder();
      cir::GlobalOp vtable =
          CGM.getCXXABI().getAddrOfVTable(VTableClass, CharUnits());
      clang::VTableLayout::AddressPointLocation addressPoint =
          CGM.getItaniumVTableContext()
              .getVTableLayout(VTableClass)
              .getAddressPoint(BaseSubobject(CD, Offset));
      assert(!cir::MissingFeatures::ptrAuth());
      mlir::ArrayAttr indices = builder.getArrayAttr({
          builder.getI32IntegerAttr(addressPoint.VTableIndex),
          builder.getI32IntegerAttr(addressPoint.AddressPointIndex),
      });
      cir::GlobalViewAttr vtableInit =
          CGM.getBuilder().getGlobalViewAttr(vtable, indices);
      if (!AppendBytes(Offset, vtableInit))
        return false;
    }
    // Accumulate and sort bases, in order to visit them in address order, which
    // may not be the same as declaration order.
    SmallVector<BaseInfo, 8> Bases;
    Bases.reserve(CD->getNumBases());
    unsigned BaseNo = 0;
    for (CXXRecordDecl::base_class_const_iterator Base = CD->bases_begin(),
                                                  BaseEnd = CD->bases_end();
         Base != BaseEnd; ++Base, ++BaseNo) {
      assert(!Base->isVirtual() && "should not have virtual bases here");
      const CXXRecordDecl *BD = Base->getType()->getAsCXXRecordDecl();
      CharUnits BaseOffset = Layout.getBaseClassOffset(BD);
      Bases.push_back(BaseInfo(BD, BaseOffset, BaseNo));
    }
    llvm::stable_sort(Bases);

    for (unsigned I = 0, N = Bases.size(); I != N; ++I) {
      BaseInfo &Base = Bases[I];

      bool IsPrimaryBase = Layout.getPrimaryBase() == Base.Decl;
      Build(Val.getStructBase(Base.Index), Base.Decl, IsPrimaryBase,
            VTableClass, Offset + Base.Offset);
    }
  }

  unsigned FieldNo = 0;
  uint64_t OffsetBits = CGM.getASTContext().toBits(Offset);

  bool AllowOverwrite = false;
  for (RecordDecl::field_iterator Field = RD->field_begin(),
                                  FieldEnd = RD->field_end();
       Field != FieldEnd; ++Field, ++FieldNo) {
    // If this is a union, skip all the fields that aren't being initialized.
    if (RD->isUnion() && !declaresSameEntity(Val.getUnionField(), *Field))
      continue;

    // Don't emit anonymous bitfields or zero-sized fields.
    if (Field->isUnnamedBitField() || Field->isZeroSize(CGM.getASTContext()))
      continue;

    // Emit the value of the initializer.
    const APValue &FieldValue =
        RD->isUnion() ? Val.getUnionValue() : Val.getStructField(FieldNo);
    mlir::Attribute EltInit =
        Emitter.tryEmitPrivateForMemory(FieldValue, Field->getType());
    if (!EltInit)
      return false;

    if (!Field->isBitField()) {
      // Handle non-bitfield members.
      if (!AppendField(*Field, Layout.getFieldOffset(FieldNo) + OffsetBits,
                       EltInit, AllowOverwrite))
        return false;
      // After emitting a non-empty field with [[no_unique_address]], we may
      // need to overwrite its tail padding.
      if (Field->hasAttr<NoUniqueAddressAttr>())
        AllowOverwrite = true;
    } else {
      llvm_unreachable("NYI");
    }
  }

  return true;
}

bool ConstRecordBuilder::ApplyZeroInitPadding(
    const ASTRecordLayout &Layout, unsigned FieldNo, const FieldDecl &Field,
    bool AllowOverwrite, CharUnits &SizeSoFar, bool &ZeroFieldSize) {

  uint64_t StartBitOffset = Layout.getFieldOffset(FieldNo);
  CharUnits StartOffset =
      CGM.getASTContext().toCharUnitsFromBits(StartBitOffset);
  if (SizeSoFar < StartOffset) {
    if (!AppendBytes(SizeSoFar, computePadding(CGM, StartOffset - SizeSoFar),
                     AllowOverwrite))
      return false;
  }

  if (!Field.isBitField()) {
    CharUnits FieldSize =
        CGM.getASTContext().getTypeSizeInChars(Field.getType());
    SizeSoFar = StartOffset + FieldSize;
    ZeroFieldSize = FieldSize.isZero();
  } else {
    const CIRGenRecordLayout &RL =
        CGM.getTypes().getCIRGenRecordLayout(Field.getParent());
    const CIRGenBitFieldInfo &Info = RL.getBitFieldInfo(&Field);
    uint64_t EndBitOffset = StartBitOffset + Info.Size;
    SizeSoFar = CGM.getASTContext().toCharUnitsFromBits(EndBitOffset);
    if (EndBitOffset % CGM.getASTContext().getCharWidth() != 0) {
      SizeSoFar++;
    }
    ZeroFieldSize = Info.Size == 0;
  }
  return true;
}

bool ConstRecordBuilder::ApplyZeroInitPadding(const ASTRecordLayout &Layout,
                                              bool AllowOverwrite,
                                              CharUnits SizeSoFar) {
  CharUnits TotalSize = Layout.getSize();
  if (SizeSoFar < TotalSize) {
    if (!AppendBytes(SizeSoFar, computePadding(CGM, TotalSize - SizeSoFar),
                     AllowOverwrite))
      return false;
  }
  SizeSoFar = TotalSize;
  return true;
}

mlir::Attribute ConstRecordBuilder::Finalize(QualType Type) {
  Type = Type.getNonReferenceType();
  RecordDecl *RD = Type->castAs<clang::RecordType>()->getDecl();
  mlir::Type ValTy = CGM.convertType(Type);
  return Builder.build(ValTy, RD->hasFlexibleArrayMember());
}

mlir::Attribute ConstRecordBuilder::BuildRecord(ConstantEmitter &Emitter,
                                                InitListExpr *ILE,
                                                QualType ValTy) {
  ConstantAggregateBuilder Const(Emitter.CGM);
  ConstRecordBuilder Builder(Emitter, Const, CharUnits::Zero());

  if (!Builder.Build(ILE, /*AllowOverwrite*/ false))
    return nullptr;

  return Builder.Finalize(ValTy);
}

mlir::Attribute ConstRecordBuilder::BuildRecord(ConstantEmitter &Emitter,
                                                const APValue &Val,
                                                QualType ValTy) {
  ConstantAggregateBuilder Const(Emitter.CGM);
  ConstRecordBuilder Builder(Emitter, Const, CharUnits::Zero());

  const RecordDecl *RD = ValTy->castAs<clang::RecordType>()->getDecl();
  const CXXRecordDecl *CD = dyn_cast<CXXRecordDecl>(RD);
  if (!Builder.Build(Val, RD, false, CD, CharUnits::Zero()))
    return nullptr;

  return Builder.Finalize(ValTy);
}

bool ConstRecordBuilder::UpdateRecord(ConstantEmitter &Emitter,
                                      ConstantAggregateBuilder &Const,
                                      CharUnits Offset, InitListExpr *Updater) {
  return ConstRecordBuilder(Emitter, Const, Offset)
      .Build(Updater, /*AllowOverwrite*/ true);
}

//===----------------------------------------------------------------------===//
//                             ConstExprEmitter
//===----------------------------------------------------------------------===//

// This class only needs to handle arrays, structs and unions.
//
// In LLVM codegen, when outside C++11 mode, those types are not constant
// folded, while all other types are handled by constant folding.
//
// In CIR codegen, instead of folding things here, we should defer that work
// to MLIR: do not attempt to do much here.
class ConstExprEmitter
    : public StmtVisitor<ConstExprEmitter, mlir::Attribute, QualType> {
  CIRGenModule &CGM;
  LLVM_ATTRIBUTE_UNUSED ConstantEmitter &Emitter;

public:
  ConstExprEmitter(ConstantEmitter &emitter)
      : CGM(emitter.CGM), Emitter(emitter) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Attribute VisitStmt(Stmt *S, QualType T) { return nullptr; }

  mlir::Attribute VisitConstantExpr(ConstantExpr *CE, QualType T) {
    if (mlir::Attribute Result = Emitter.tryEmitConstantExpr(CE))
      return Result;
    return Visit(CE->getSubExpr(), T);
  }

  mlir::Attribute VisitParenExpr(ParenExpr *PE, QualType T) {
    return Visit(PE->getSubExpr(), T);
  }

  mlir::Attribute
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *PE,
                                    QualType T) {
    return Visit(PE->getReplacement(), T);
  }

  mlir::Attribute VisitGenericSelectionExpr(GenericSelectionExpr *GE,
                                            QualType T) {
    return Visit(GE->getResultExpr(), T);
  }

  mlir::Attribute VisitChooseExpr(ChooseExpr *CE, QualType T) {
    return Visit(CE->getChosenSubExpr(), T);
  }

  mlir::Attribute VisitCompoundLiteralExpr(CompoundLiteralExpr *E, QualType T) {
    return Visit(E->getInitializer(), T);
  }

  mlir::Attribute VisitCastExpr(CastExpr *E, QualType destType) {
    if (const auto *ECE = dyn_cast<ExplicitCastExpr>(E))
      CGM.emitExplicitCastExprType(ECE, Emitter.CGF);
    Expr *subExpr = E->getSubExpr();

    switch (E->getCastKind()) {
    case CK_HLSLArrayRValue:
    case CK_HLSLVectorTruncation:
    case CK_HLSLElementwiseCast:
    case CK_HLSLAggregateSplatCast:
    case CK_ToUnion:
      llvm_unreachable("not implemented");

    case CK_AddressSpaceConversion: {
      llvm_unreachable("not implemented");
    }

    case CK_LValueToRValue:
    case CK_AtomicToNonAtomic:
    case CK_NonAtomicToAtomic:
    case CK_NoOp:
    case CK_ConstructorConversion:
      return Visit(subExpr, destType);

    case CK_IntToOCLSampler:
      llvm_unreachable("global sampler variables are not generated");

    case CK_Dependent:
      llvm_unreachable("saw dependent cast!");

    case CK_BuiltinFnToFnPtr:
      llvm_unreachable("builtin functions are handled elsewhere");

    case CK_ReinterpretMemberPointer:
    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer: {
      llvm_unreachable("not implemented");
    }

    // These will never be supported.
    case CK_ObjCObjectLValueCast:
    case CK_ARCProduceObject:
    case CK_ARCConsumeObject:
    case CK_ARCReclaimReturnedObject:
    case CK_ARCExtendBlockObject:
    case CK_CopyAndAutoreleaseBlockObject:
      return nullptr;

    // These don't need to be handled here because Evaluate knows how to
    // evaluate them in the cases where they can be folded.
    case CK_BitCast:
    case CK_ToVoid:
    case CK_Dynamic:
    case CK_LValueBitCast:
    case CK_LValueToRValueBitCast:
    case CK_NullToMemberPointer:
    case CK_UserDefinedConversion:
    case CK_CPointerToObjCPointerCast:
    case CK_BlockPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
    case CK_ArrayToPointerDecay:
    case CK_FunctionToPointerDecay:
    case CK_BaseToDerived:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_MemberPointerToBoolean:
    case CK_VectorSplat:
    case CK_FloatingRealToComplex:
    case CK_FloatingComplexToReal:
    case CK_FloatingComplexToBoolean:
    case CK_FloatingComplexCast:
    case CK_FloatingComplexToIntegralComplex:
    case CK_IntegralRealToComplex:
    case CK_IntegralComplexToReal:
    case CK_IntegralComplexToBoolean:
    case CK_IntegralComplexCast:
    case CK_IntegralComplexToFloatingComplex:
    case CK_PointerToIntegral:
    case CK_PointerToBoolean:
    case CK_NullToPointer:
    case CK_IntegralCast:
    case CK_BooleanToSignedIntegral:
    case CK_IntegralToPointer:
    case CK_IntegralToBoolean:
    case CK_IntegralToFloating:
    case CK_FloatingToIntegral:
    case CK_FloatingToBoolean:
    case CK_FloatingCast:
    case CK_FloatingToFixedPoint:
    case CK_FixedPointToFloating:
    case CK_FixedPointCast:
    case CK_FixedPointToBoolean:
    case CK_FixedPointToIntegral:
    case CK_IntegralToFixedPoint:
    case CK_ZeroToOCLOpaqueType:
    case CK_MatrixCast:
      return nullptr;
    }
    llvm_unreachable("Invalid CastKind");
  }

  mlir::Attribute VisitCXXDefaultInitExpr(CXXDefaultInitExpr *DIE, QualType T) {
    // TODO(cir): figure out CIR story here...
    // No need for a DefaultInitExprScope: we don't handle 'this' in a
    // constant expression.
    return Visit(DIE->getExpr(), T);
  }

  mlir::Attribute VisitExprWithCleanups(ExprWithCleanups *E, QualType T) {
    // Since this about constant emission no need to wrap this under a scope.
    return Visit(E->getSubExpr(), T);
  }

  mlir::Attribute VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *E,
                                                QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  mlir::Attribute EmitArrayInitialization(InitListExpr *ILE, QualType T) {
    auto *CAT = CGM.getASTContext().getAsConstantArrayType(ILE->getType());
    assert(CAT && "can't emit array init for non-constant-bound array");
    unsigned NumInitElements = ILE->getNumInits();        // init list size
    unsigned NumElements = CAT->getSize().getZExtValue(); // array size
    unsigned NumInitableElts = std::min(NumInitElements, NumElements);

    QualType EltTy = CAT->getElementType();
    SmallVector<mlir::TypedAttr, 16> Elts;
    Elts.reserve(NumElements);

    // Emit array filler, if there is one.
    mlir::Attribute Filler;
    if (ILE->hasArrayFiller()) {
      auto *aux = ILE->getArrayFiller();
      Filler = Emitter.tryEmitAbstractForMemory(aux, CAT->getElementType());
      if (!Filler)
        return {};
    }

    auto desiredType = CGM.convertType(T);
    // FIXME(cir): A hack to handle the emission of arrays of unions directly.
    // See clang/test/CIR/CodeGen/union-array.c and
    // clang/test/CIR/Lowering/nested-union-array.c for example. The root
    // cause of these problems is CIR handles union differently than LLVM IR.
    // So we can't fix the problem fundamentally by mocking LLVM's handling for
    // unions. In LLVM, the union is basically a struct with the largest member
    // of the union and consumers cast the union arbitrarily according to their
    // needs. But in CIR, we tried to express union semantics properly. This is
    // a fundamental difference.
    //
    // Concretely, for the problem here, if we're constructing the initializer
    // for the array of unions, we can't even assume the type of the elements in
    // the initializer are the same! It is odd that we can have an array with
    // different element types. Here we just pretend it is fine by checking if
    // we're constructing an array for an array of unions. If we didn't do so,
    // we may meet problems during lowering to LLVM. To solve the problem, we
    // may need to introduce 2 type systems for CIR: one for the CIR itself and
    // one for lowering. e.g., we can compare the type of CIR during CIRGen,
    // analysis and transformations without worrying the concerns here. And
    // lower to LLVM IR (or anyother dialects) with the proper type.
    //
    // (Although the idea to make CIR's type system self contained and generate
    // LLVM's
    //  types in later passes look fine, it has engineering level concern that
    //  it will make the skeleton of CIRGen to be diverged from the traditional
    //  CodeGen.)
    //
    // Besides union, there are other differences between CIR and LLVM's type
    // system. e.g., LLVM's pointer types are opaque while CIR has concrete
    // pointer types.
    bool isDesiredArrayOfUnionDirectly = [&]() {
      auto desiredArrayType = dyn_cast<cir::ArrayType>(desiredType);
      if (!desiredArrayType)
        return false;

      auto elementRecordType =
          dyn_cast<cir::RecordType>(desiredArrayType.getElementType());
      if (!elementRecordType)
        return false;

      return elementRecordType.isUnion();
    }();

    // Emit initializer elements as MLIR attributes and check for common type.
    mlir::Type CommonElementType;
    for (unsigned i = 0; i != NumInitableElts; ++i) {
      Expr *Init = ILE->getInit(i);
      auto C = Emitter.tryEmitPrivateForMemory(Init, EltTy);
      if (!C)
        return {};
      if (i == 0)
        CommonElementType = C.getType();
      else if (isDesiredArrayOfUnionDirectly &&
               C.getType() != CommonElementType)
        CommonElementType = nullptr;
      Elts.push_back(std::move(C));
    }

    auto typedFiller = llvm::dyn_cast_or_null<mlir::TypedAttr>(Filler);
    if (Filler && !typedFiller)
      llvm_unreachable("We shouldn't be receiving untyped attrs here");
    return emitArrayConstant(CGM, desiredType, CommonElementType, NumElements,
                             Elts, typedFiller);
  }

  mlir::Attribute EmitRecordInitialization(InitListExpr *ILE, QualType T) {
    return ConstRecordBuilder::BuildRecord(Emitter, ILE, T);
  }

  mlir::Attribute EmitVectorInitialization(InitListExpr *ILE, QualType T) {
    cir::VectorType VecTy = mlir::cast<cir::VectorType>(CGM.convertType(T));
    unsigned NumElements = VecTy.getSize();
    unsigned NumInits = ILE->getNumInits();
    assert(NumElements >= NumInits && "Too many initializers for a vector");
    QualType EltTy = T->castAs<VectorType>()->getElementType();
    SmallVector<mlir::Attribute, 8> Elts;
    // Process the explicit initializers
    for (unsigned i = 0; i < NumInits; ++i) {
      auto Value = Emitter.tryEmitPrivateForMemory(ILE->getInit(i), EltTy);
      if (!Value)
        return {};
      Elts.push_back(std::move(Value));
    }
    // Zero-fill the rest of the vector
    for (unsigned i = NumInits; i < NumElements; ++i) {
      Elts.push_back(CGM.getBuilder().getZeroInitAttr(VecTy.getElementType()));
    }
    return cir::ConstVectorAttr::get(
        VecTy, mlir::ArrayAttr::get(CGM.getBuilder().getContext(), Elts));
  }

  mlir::Attribute VisitImplicitValueInitExpr(ImplicitValueInitExpr *E,
                                             QualType T) {
    return CGM.getBuilder().getZeroInitAttr(CGM.convertType(T));
  }

  mlir::Attribute VisitInitListExpr(InitListExpr *ILE, QualType T) {
    if (ILE->isTransparent())
      return Visit(ILE->getInit(0), T);

    if (ILE->getType()->isArrayType())
      return EmitArrayInitialization(ILE, T);

    if (ILE->getType()->isRecordType())
      return EmitRecordInitialization(ILE, T);

    if (ILE->getType()->isVectorType())
      return EmitVectorInitialization(ILE, T);

    return nullptr;
  }

  mlir::Attribute VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *E,
                                                QualType destType) {
    auto C = Visit(E->getBase(), destType);
    if (!C)
      return nullptr;

    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitCXXConstructExpr(CXXConstructExpr *E, QualType Ty) {
    if (!E->getConstructor()->isTrivial())
      return nullptr;

    // Only default and copy/move constructors can be trivial.
    if (E->getNumArgs()) {
      assert(E->getNumArgs() == 1 && "trivial ctor with > 1 argument");
      assert(E->getConstructor()->isCopyOrMoveConstructor() &&
             "trivial ctor has argument but isn't a copy/move ctor");

      Expr *Arg = E->getArg(0);
      assert(CGM.getASTContext().hasSameUnqualifiedType(Ty, Arg->getType()) &&
             "argument to copy ctor is of wrong type");

      // Look through the temporary; it's just converting the value to an lvalue
      // to pass it to the constructor.
      if (auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Arg))
        return Visit(MTE->getSubExpr(), Ty);
      // Don't try to support arbitrary lvalue-to-rvalue conversions for now.
      return nullptr;
    }

    return CGM.getBuilder().getZeroInitAttr(CGM.convertType(Ty));
  }

  mlir::Attribute VisitStringLiteral(StringLiteral *E, QualType T) {
    // This is a string literal initializing an array in an initializer.
    return CGM.getConstantArrayFromStringLiteral(E);
  }

  mlir::Attribute VisitObjCEncodeExpr(ObjCEncodeExpr *E, QualType T) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitUnaryExtension(const UnaryOperator *E, QualType T) {
    return Visit(E->getSubExpr(), T);
  }

  // Utility methods
  mlir::Type convertType(QualType T) { return CGM.convertType(T); }
};

static mlir::Attribute
emitArrayConstant(CIRGenModule &CGM, mlir::Type DesiredType,
                  mlir::Type CommonElementType, unsigned ArrayBound,
                  SmallVectorImpl<mlir::TypedAttr> &Elements,
                  mlir::TypedAttr Filler) {
  auto &builder = CGM.getBuilder();

  // Figure out how long the initial prefix of non-zero elements is.
  unsigned NonzeroLength = ArrayBound;
  if (Elements.size() < NonzeroLength && builder.isNullValue(Filler))
    NonzeroLength = Elements.size();
  if (NonzeroLength == Elements.size()) {
    while (NonzeroLength > 0 &&
           builder.isNullValue(Elements[NonzeroLength - 1]))
      --NonzeroLength;
  }

  if (NonzeroLength == 0)
    return builder.getZeroInitAttr(DesiredType);

  // Add a zeroinitializer array filler if we have lots of trailing zeroes.
  unsigned TrailingZeroes = ArrayBound - NonzeroLength;
  if (TrailingZeroes >= 8) {
    assert(Elements.size() >= NonzeroLength &&
           "missing initializer for non-zero element");

    SmallVector<mlir::Attribute, 4> Eles;
    Eles.reserve(Elements.size());
    for (auto const &Element : Elements)
      Eles.push_back(Element);

    return builder.getConstArray(
        mlir::ArrayAttr::get(builder.getContext(), Eles),
        cir::ArrayType::get(CommonElementType, ArrayBound));
    // TODO(cir): If all the elements had the same type up to the trailing
    // zeroes, emit a record of two arrays (the nonzero data and the
    // zeroinitializer). Use DesiredType to get the element type.
  } else if (Elements.size() != ArrayBound) {
    // Otherwise pad to the right size with the filler if necessary.
    Elements.resize(ArrayBound, Filler);
    if (Filler.getType() != CommonElementType)
      CommonElementType = {};
  }

  // If all elements have the same type, just emit an array constant.
  if (CommonElementType) {
    SmallVector<mlir::Attribute, 4> Eles;
    Eles.reserve(Elements.size());
    for (auto const &Element : Elements)
      Eles.push_back(Element);

    return builder.getConstArray(
        mlir::ArrayAttr::get(builder.getContext(), Eles),
        cir::ArrayType::get(CommonElementType, ArrayBound));
  }

  SmallVector<mlir::Attribute, 4> Eles;
  Eles.reserve(Elements.size());
  for (auto const &Element : Elements)
    Eles.push_back(Element);

  auto arrAttr = mlir::ArrayAttr::get(builder.getContext(), Eles);
  return builder.getAnonConstRecord(arrAttr, false);
}

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                          ConstantLValueEmitter
//===----------------------------------------------------------------------===//

namespace {
/// A struct which can be used to peephole certain kinds of finalization
/// that normally happen during l-value emission.
struct ConstantLValue {
  llvm::PointerUnion<mlir::Value, mlir::Attribute> Value;
  bool HasOffsetApplied;

  /*implicit*/ ConstantLValue(mlir::Value value, bool hasOffsetApplied = false)
      : Value(value), HasOffsetApplied(hasOffsetApplied) {}

  /*implicit*/ ConstantLValue(cir::GlobalViewAttr address)
      : Value(address), HasOffsetApplied(false) {}

  ConstantLValue(std::nullptr_t) : ConstantLValue({}, false) {}
};

/// A helper class for emitting constant l-values.
class ConstantLValueEmitter
    : public ConstStmtVisitor<ConstantLValueEmitter, ConstantLValue> {
  CIRGenModule &CGM;
  ConstantEmitter &Emitter;
  const APValue &Value;
  QualType DestType;

  // Befriend StmtVisitorBase so that we don't have to expose Visit*.
  friend StmtVisitorBase;

public:
  ConstantLValueEmitter(ConstantEmitter &emitter, const APValue &value,
                        QualType destType)
      : CGM(emitter.CGM), Emitter(emitter), Value(value), DestType(destType) {}

  mlir::Attribute tryEmit();

private:
  mlir::Attribute tryEmitAbsolute(mlir::Type destTy);
  ConstantLValue tryEmitBase(const APValue::LValueBase &base);

  ConstantLValue VisitStmt(const Stmt *S) { return nullptr; }
  ConstantLValue VisitConstantExpr(const ConstantExpr *E);
  ConstantLValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *E);
  ConstantLValue VisitStringLiteral(const StringLiteral *E);
  ConstantLValue VisitObjCBoxedExpr(const ObjCBoxedExpr *E);
  ConstantLValue VisitObjCEncodeExpr(const ObjCEncodeExpr *E);
  ConstantLValue VisitObjCStringLiteral(const ObjCStringLiteral *E);
  ConstantLValue VisitPredefinedExpr(const PredefinedExpr *E);
  ConstantLValue VisitAddrLabelExpr(const AddrLabelExpr *E);
  ConstantLValue VisitCallExpr(const CallExpr *E);
  ConstantLValue VisitBlockExpr(const BlockExpr *E);
  ConstantLValue VisitCXXTypeidExpr(const CXXTypeidExpr *E);
  ConstantLValue
  VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *expr);

  bool hasNonZeroOffset() const { return !Value.getLValueOffset().isZero(); }

  /// Return GEP-like value offset
  mlir::ArrayAttr getOffset(mlir::Type Ty) {
    auto Offset = Value.getLValueOffset().getQuantity();
    cir::CIRDataLayout Layout(CGM.getModule());
    SmallVector<int64_t, 3> Idx;
    CGM.getBuilder().computeGlobalViewIndicesFromFlatOffset(Offset, Ty, Layout,
                                                            Idx);

    llvm::SmallVector<mlir::Attribute, 3> Indices;
    for (auto I : Idx) {
      auto Attr = CGM.getBuilder().getI32IntegerAttr(I);
      Indices.push_back(Attr);
    }

    if (Indices.empty())
      return {};
    return CGM.getBuilder().getArrayAttr(Indices);
  }

  // TODO(cir): create a proper interface to absctract CIR constant values.

  /// Apply the value offset to the given constant.
  ConstantLValue applyOffset(ConstantLValue &C) {

    // Handle attribute constant LValues.
    if (auto Attr = mlir::dyn_cast<mlir::Attribute>(C.Value)) {
      if (auto GV = mlir::dyn_cast<cir::GlobalViewAttr>(Attr)) {
        auto baseTy = mlir::cast<cir::PointerType>(GV.getType()).getPointee();
        auto destTy = CGM.getTypes().convertTypeForMem(DestType);
        assert(!GV.getIndices() && "Global view is already indexed");
        return cir::GlobalViewAttr::get(destTy, GV.getSymbol(),
                                        getOffset(baseTy));
      }
      llvm_unreachable("Unsupported attribute type to offset");
    }

    // TODO(cir): use ptr_stride, or something...
    llvm_unreachable("NYI");
  }
};

} // namespace

mlir::Attribute ConstantLValueEmitter::tryEmit() {
  const APValue::LValueBase &base = Value.getLValueBase();

  // The destination type should be a pointer or reference
  // type, but it might also be a cast thereof.
  //
  // FIXME: the chain of casts required should be reflected in the APValue.
  // We need this in order to correctly handle things like a ptrtoint of a
  // non-zero null pointer and addrspace casts that aren't trivially
  // represented in LLVM IR.
  auto destTy = CGM.getTypes().convertTypeForMem(DestType);
  assert(mlir::isa<cir::PointerType>(destTy));

  // If there's no base at all, this is a null or absolute pointer,
  // possibly cast back to an integer type.
  if (!base) {
    return tryEmitAbsolute(destTy);
  }

  // Otherwise, try to emit the base.
  ConstantLValue result = tryEmitBase(base);

  // If that failed, we're done.
  auto &value = result.Value;
  if (!value)
    return {};

  // Apply the offset if necessary and not already done.
  if (!result.HasOffsetApplied) {
    value = applyOffset(result).Value;
  }

  // Convert to the appropriate type; this could be an lvalue for
  // an integer. FIXME: performAddrSpaceCast
  if (mlir::isa<cir::PointerType>(destTy)) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(value))
      return attr;
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
}

/// Try to emit an absolute l-value, such as a null pointer or an integer
/// bitcast to pointer type.
mlir::Attribute ConstantLValueEmitter::tryEmitAbsolute(mlir::Type destTy) {
  // If we're producing a pointer, this is easy.
  auto destPtrTy = mlir::dyn_cast<cir::PointerType>(destTy);
  assert(destPtrTy && "expected !cir.ptr type");
  return CGM.getBuilder().getConstPtrAttr(
      destPtrTy, Value.getLValueOffset().getQuantity());
}

ConstantLValue
ConstantLValueEmitter::tryEmitBase(const APValue::LValueBase &base) {
  // Handle values.
  if (const ValueDecl *D = base.dyn_cast<const ValueDecl *>()) {
    // The constant always points to the canonical declaration. We want to look
    // at properties of the most recent declaration at the point of emission.
    D = cast<ValueDecl>(D->getMostRecentDecl());

    if (D->hasAttr<WeakRefAttr>())
      llvm_unreachable("emit pointer base for weakref is NYI");

    if (auto *FD = dyn_cast<FunctionDecl>(D)) {
      auto fop = CGM.GetAddrOfFunction(FD);
      auto builder = CGM.getBuilder();
      mlir::MLIRContext *mlirContext = builder.getContext();
      return cir::GlobalViewAttr::get(
          builder.getPointerTo(fop.getFunctionType()),
          mlir::FlatSymbolRefAttr::get(mlirContext, fop.getSymNameAttr()));
    }

    if (auto *VD = dyn_cast<VarDecl>(D)) {
      // We can never refer to a variable with local storage.
      if (!VD->hasLocalStorage()) {
        if (VD->isFileVarDecl() || VD->hasExternalStorage())
          return CGM.getAddrOfGlobalVarAttr(VD);

        if (VD->isLocalVarDecl()) {
          auto linkage =
              CGM.getCIRLinkageVarDefinition(VD, /*IsConstant=*/false);
          return CGM.getBuilder().getGlobalViewAttr(
              CGM.getOrCreateStaticVarDecl(*VD, linkage));
        }
      }
    }
  }

  // Handle typeid(T).
  if (TypeInfoLValue TI = base.dyn_cast<TypeInfoLValue>()) {
    assert(0 && "NYI");
  }

  // Otherwise, it must be an expression.
  return Visit(base.get<const Expr *>());
}

static ConstantLValue
tryEmitGlobalCompoundLiteral(ConstantEmitter &emitter,
                             const CompoundLiteralExpr *E) {
  CIRGenModule &CGM = emitter.CGM;

  LangAS addressSpace = E->getType().getAddressSpace();
  mlir::Attribute C = emitter.tryEmitForInitializer(E->getInitializer(),
                                                    addressSpace, E->getType());
  if (!C) {
    assert(!E->isFileScope() &&
           "file-scope compound literal did not have constant initializer!");
    return nullptr;
  }

  auto GV = CIRGenModule::createGlobalOp(
      CGM, CGM.getLoc(E->getSourceRange()),
      CGM.createGlobalCompoundLiteralName(),
      CGM.getTypes().convertTypeForMem(E->getType()),
      E->getType().isConstantStorage(CGM.getASTContext(), false, false));
  GV.setInitialValueAttr(C);
  GV.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
  CharUnits Align = CGM.getASTContext().getTypeAlignInChars(E->getType());
  GV.setAlignment(Align.getAsAlign().value());

  emitter.finalize(GV);
  return CGM.getBuilder().getGlobalViewAttr(GV);
}

ConstantLValue ConstantLValueEmitter::VisitConstantExpr(const ConstantExpr *E) {
  assert(0 && "NYI");
  return Visit(E->getSubExpr());
}

ConstantLValue
ConstantLValueEmitter::VisitCompoundLiteralExpr(const CompoundLiteralExpr *E) {
  ConstantEmitter CompoundLiteralEmitter(CGM, Emitter.CGF);
  CompoundLiteralEmitter.setInConstantContext(Emitter.isInConstantContext());
  return tryEmitGlobalCompoundLiteral(CompoundLiteralEmitter, E);
}

ConstantLValue
ConstantLValueEmitter::VisitStringLiteral(const StringLiteral *E) {
  return CGM.getAddrOfConstantStringFromLiteral(E);
}

ConstantLValue
ConstantLValueEmitter::VisitObjCEncodeExpr(const ObjCEncodeExpr *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitObjCStringLiteral(const ObjCStringLiteral *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitObjCBoxedExpr(const ObjCBoxedExpr *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitPredefinedExpr(const PredefinedExpr *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitAddrLabelExpr(const AddrLabelExpr *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue ConstantLValueEmitter::VisitCallExpr(const CallExpr *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue ConstantLValueEmitter::VisitBlockExpr(const BlockExpr *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitCXXTypeidExpr(const CXXTypeidExpr *E) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue ConstantLValueEmitter::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *expr) {
  assert(expr->getStorageDuration() == SD_Static);
  const Expr *inner = expr->getSubExpr()->skipRValueSubobjectAdjustments();
  mlir::Operation *globalTemp = CGM.getAddrOfGlobalTemporary(expr, inner);
  CIRGenBuilderTy builder = CGM.getBuilder();
  return ConstantLValue(
      builder.getGlobalViewAttr(mlir::cast<cir::GlobalOp>(globalTemp)));
}

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Attribute ConstantEmitter::validateAndPopAbstract(mlir::Attribute C,
                                                        AbstractState saved) {
  Abstract = saved.OldValue;

  assert(saved.OldPlaceholdersSize == PlaceholderAddresses.size() &&
         "created a placeholder while doing an abstract emission?");

  // No validation necessary for now.
  // No cleanup to do for now.
  return C;
}

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const VarDecl &D) {
  initializeNonAbstract(D.getType().getAddressSpace());
  return markIfFailed(tryEmitPrivateForVarInit(D));
}

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const Expr *E,
                                                       LangAS destAddrSpace,
                                                       QualType destType) {
  initializeNonAbstract(destAddrSpace);
  return markIfFailed(tryEmitPrivateForMemory(E, destType));
}

mlir::Attribute ConstantEmitter::emitForInitializer(const APValue &value,
                                                    LangAS destAddrSpace,
                                                    QualType destType) {
  initializeNonAbstract(destAddrSpace);
  auto c = tryEmitPrivateForMemory(value, destType);
  assert(c && "couldn't emit constant value non-abstractly?");
  return c;
}

void ConstantEmitter::finalize(cir::GlobalOp global) {
  assert(InitializedNonAbstract &&
         "finalizing emitter that was used for abstract emission?");
  assert(!Finalized && "finalizing emitter multiple times");
  assert(!global.isDeclaration());

  // Note that we might also be Failed.
  Finalized = true;

  if (!PlaceholderAddresses.empty()) {
    assert(0 && "not implemented");
  }
}

ConstantEmitter::~ConstantEmitter() {
  assert((!InitializedNonAbstract || Finalized || Failed) &&
         "not finalized after being initialized for non-abstract emission");
  assert(PlaceholderAddresses.empty() && "unhandled placeholders");
}

// TODO(cir): this can be shared with LLVM's codegen
static QualType getNonMemoryType(CIRGenModule &CGM, QualType type) {
  if (auto AT = type->getAs<AtomicType>()) {
    return CGM.getASTContext().getQualifiedType(AT->getValueType(),
                                                type.getQualifiers());
  }
  return type;
}

mlir::Attribute
ConstantEmitter::tryEmitAbstractForInitializer(const VarDecl &D) {
  auto state = pushAbstract();
  auto C = tryEmitPrivateForVarInit(D);
  return validateAndPopAbstract(C, state);
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForVarInit(const VarDecl &D) {
  // Make a quick check if variable can be default NULL initialized
  // and avoid going through rest of code which may do, for c++11,
  // initialization of memory to all NULLs.
  if (!D.hasLocalStorage()) {
    QualType Ty = CGM.getASTContext().getBaseElementType(D.getType());
    if (Ty->isRecordType()) {
      if (const CXXConstructExpr *E =
              dyn_cast_or_null<CXXConstructExpr>(D.getInit())) {
        const CXXConstructorDecl *CD = E->getConstructor();
        // FIXME: we should probably model this more closely to C++ than
        // just emitting a global with zero init (mimic what we do for trivial
        // assignments and whatnots). Since this is for globals shouldn't
        // be a problem for the near future.
        if (CD->isTrivial() && CD->isDefaultConstructor()) {
          const auto *cxxrd =
              cast<CXXRecordDecl>(Ty->getAs<RecordType>()->getDecl());
          // Some cases, such as member pointer members, can't be zero
          // initialized. These are "zero-initialized" in the language standard
          // sense, but the target ABI may require that a literal value other
          // than zero be used in the initializer to make clear that a pointer
          // with the value zero is not what is intended. The classic codegen
          // goes through emitNullConstant for those cases but generates a
          // non-zero constant. We can't quite do that here because we need an
          // attribute and not a value, but something like that can be
          // implemented.
          if (!CGM.getTypes().isZeroInitializable(cxxrd)) {
            llvm_unreachable("NYI");
          }
          return cir::ZeroAttr::get(CGM.convertType(D.getType()));
        }
      }
    }
  }
  InConstantContext = D.hasConstantInitialization();

  const Expr *E = D.getInit();
  assert(E && "No initializer to emit");

  QualType destType = D.getType();

  if (!destType->isReferenceType()) {
    QualType nonMemoryDestType = getNonMemoryType(CGM, destType);
    if (auto C = ConstExprEmitter(*this).Visit(const_cast<Expr *>(E),
                                               nonMemoryDestType))
      return emitForMemory(C, destType);
  }

  // Try to emit the initializer.  Note that this can allow some things that
  // are not allowed by tryEmitPrivateForMemory alone.
  if (auto value = D.evaluateValue())
    return tryEmitPrivateForMemory(*value, destType);

  return nullptr;
}

mlir::Attribute ConstantEmitter::tryEmitAbstract(const Expr *E,
                                                 QualType destType) {
  auto state = pushAbstract();
  auto C = tryEmitPrivate(E, destType);
  return validateAndPopAbstract(C, state);
}

mlir::Attribute ConstantEmitter::tryEmitAbstract(const APValue &value,
                                                 QualType destType) {
  auto state = pushAbstract();
  auto C = tryEmitPrivate(value, destType);
  return validateAndPopAbstract(C, state);
}

mlir::Attribute ConstantEmitter::tryEmitConstantExpr(const ConstantExpr *CE) {
  if (!CE->hasAPValueResult())
    return nullptr;

  QualType RetType = CE->getType();
  if (CE->isGLValue())
    RetType = CGM.getASTContext().getLValueReferenceType(RetType);

  return emitAbstract(CE->getBeginLoc(), CE->getAPValueResult(), RetType);
}

mlir::Attribute ConstantEmitter::tryEmitAbstractForMemory(const Expr *E,
                                                          QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto C = tryEmitAbstract(E, nonMemoryDestType);
  return (C ? emitForMemory(C, destType) : nullptr);
}

mlir::Attribute ConstantEmitter::tryEmitAbstractForMemory(const APValue &value,
                                                          QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto C = tryEmitAbstract(value, nonMemoryDestType);
  return (C ? emitForMemory(C, destType) : nullptr);
}

mlir::TypedAttr ConstantEmitter::tryEmitPrivateForMemory(const Expr *E,
                                                         QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto C = tryEmitPrivate(E, nonMemoryDestType);
  if (C) {
    auto attr = emitForMemory(C, destType);
    auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(attr);
    if (!typedAttr)
      llvm_unreachable("this should always be typed");
    return typedAttr;
  } else {
    return nullptr;
  }
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForMemory(const APValue &value,
                                                         QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto C = tryEmitPrivate(value, nonMemoryDestType);
  return (C ? emitForMemory(C, destType) : nullptr);
}

mlir::Attribute ConstantEmitter::emitForMemory(CIRGenModule &CGM,
                                               mlir::Attribute C,
                                               QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (auto AT = destType->getAs<AtomicType>()) {
    QualType destValueType = AT->getValueType();
    C = emitForMemory(CGM, C, destValueType);

    uint64_t innerSize = CGM.getASTContext().getTypeSize(destValueType);
    uint64_t outerSize = CGM.getASTContext().getTypeSize(destType);
    if (innerSize == outerSize)
      return C;

    assert(innerSize < outerSize && "emitted over-large constant for atomic");
    auto &builder = CGM.getBuilder();
    auto zeroArray = builder.getZeroInitAttr(
        cir::ArrayType::get(builder.getUInt8Ty(), (outerSize - innerSize) / 8));
    SmallVector<mlir::Attribute, 4> anonElts = {C, zeroArray};
    auto arrAttr = mlir::ArrayAttr::get(builder.getContext(), anonElts);
    return builder.getAnonConstRecord(arrAttr, false);
  }

  // Zero-extend bool.
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(C);
  if (typed && mlir::isa<cir::BoolType>(typed.getType())) {
    // Already taken care given that bool values coming from
    // integers only carry true/false.
  }

  return C;
}

mlir::TypedAttr ConstantEmitter::tryEmitPrivate(const Expr *E,
                                                QualType destType) {
  assert(!destType->isVoidType() && "can't emit a void constant");

  if (auto C = ConstExprEmitter(*this).Visit(const_cast<Expr *>(E), destType)) {
    if (auto TypedC = mlir::dyn_cast_if_present<mlir::TypedAttr>(C))
      return TypedC;
    llvm_unreachable("this should always be typed");
  }

  Expr::EvalResult Result;

  bool Success;

  if (destType->isReferenceType())
    Success = E->EvaluateAsLValue(Result, CGM.getASTContext());
  else
    Success =
        E->EvaluateAsRValue(Result, CGM.getASTContext(), InConstantContext);

  if (Success && !Result.hasSideEffects()) {
    auto C = tryEmitPrivate(Result.Val, destType);
    if (auto TypedC = mlir::dyn_cast_if_present<mlir::TypedAttr>(C))
      return TypedC;
    llvm_unreachable("this should always be typed");
  }

  return nullptr;
}

mlir::Attribute ConstantEmitter::tryEmitPrivate(const APValue &Value,
                                                QualType DestType) {
  auto &builder = CGM.getBuilder();
  switch (Value.getKind()) {
  case APValue::None:
  case APValue::Indeterminate:
    // TODO(cir): LLVM models out-of-lifetime and indeterminate values as
    // 'undef'. Find out what's better for CIR.
    assert(0 && "not implemented");
  case APValue::Int: {
    mlir::Type ty = CGM.convertType(DestType);
    if (mlir::isa<cir::BoolType>(ty))
      return builder.getCIRBoolAttr(Value.getInt().getZExtValue());
    assert(mlir::isa<cir::IntType>(ty) && "expected integral type");
    return cir::IntAttr::get(ty, Value.getInt());
  }
  case APValue::Float: {
    const llvm::APFloat &Init = Value.getFloat();
    if (&Init.getSemantics() == &llvm::APFloat::IEEEhalf() &&
        !CGM.getASTContext().getLangOpts().NativeHalfType &&
        CGM.getASTContext().getTargetInfo().useFP16ConversionIntrinsics())
      assert(0 && "not implemented");
    else {
      mlir::Type ty = CGM.convertType(DestType);
      assert(mlir::isa<cir::FPTypeInterface>(ty) &&
             "expected floating-point type");
      return cir::FPAttr::get(ty, Init);
    }
  }
  case APValue::Array: {
    const ArrayType *ArrayTy = CGM.getASTContext().getAsArrayType(DestType);
    unsigned NumElements = Value.getArraySize();
    unsigned NumInitElts = Value.getArrayInitializedElts();

    // Emit array filler, if there is one.
    mlir::Attribute Filler;
    if (Value.hasArrayFiller()) {
      Filler = tryEmitAbstractForMemory(Value.getArrayFiller(),
                                        ArrayTy->getElementType());
      if (!Filler)
        return {};
    }

    // Emit initializer elements.
    SmallVector<mlir::TypedAttr, 16> Elts;
    if (Filler && builder.isNullValue(Filler))
      Elts.reserve(NumInitElts + 1);
    else
      Elts.reserve(NumElements);

    mlir::Type CommonElementType;
    for (unsigned I = 0; I < NumInitElts; ++I) {
      auto C = tryEmitPrivateForMemory(Value.getArrayInitializedElt(I),
                                       ArrayTy->getElementType());
      if (!C)
        return {};

      assert(mlir::isa<mlir::TypedAttr>(C) &&
             "This should always be a TypedAttr.");
      auto CTyped = mlir::cast<mlir::TypedAttr>(C);

      if (I == 0)
        CommonElementType = CTyped.getType();
      else if (CTyped.getType() != CommonElementType)
        CommonElementType = {};
      auto typedC = llvm::dyn_cast<mlir::TypedAttr>(C);
      if (!typedC)
        llvm_unreachable("this should always be typed");
      Elts.push_back(typedC);
    }

    auto Desired = CGM.convertType(DestType);

    auto typedFiller = llvm::dyn_cast_or_null<mlir::TypedAttr>(Filler);
    if (Filler && !typedFiller)
      llvm_unreachable("this should always be typed");

    return emitArrayConstant(CGM, Desired, CommonElementType, NumElements, Elts,
                             typedFiller);
  }
  case APValue::Vector: {
    const QualType ElementType =
        DestType->castAs<VectorType>()->getElementType();
    unsigned NumElements = Value.getVectorLength();
    SmallVector<mlir::Attribute, 16> Elts;
    Elts.reserve(NumElements);
    for (unsigned i = 0; i < NumElements; ++i) {
      auto C = tryEmitPrivateForMemory(Value.getVectorElt(i), ElementType);
      if (!C)
        return {};
      Elts.push_back(C);
    }
    auto Desired = mlir::cast<cir::VectorType>(CGM.convertType(DestType));
    return cir::ConstVectorAttr::get(
        Desired, mlir::ArrayAttr::get(CGM.getBuilder().getContext(), Elts));
  }
  case APValue::MemberPointer: {
    assert(!cir::MissingFeatures::cxxABI());

    const ValueDecl *memberDecl = Value.getMemberPointerDecl();
    assert(!Value.isMemberPointerToDerivedMember() && "NYI");

    if (isa<CXXMethodDecl>(memberDecl))
      assert(0 && "not implemented");

    auto cirTy = mlir::cast<cir::DataMemberType>(CGM.convertType(DestType));

    const auto *fieldDecl = cast<FieldDecl>(memberDecl);
    return builder.getDataMemberAttr(cirTy, fieldDecl->getFieldIndex());
  }
  case APValue::LValue:
    return ConstantLValueEmitter(*this, Value, DestType).tryEmit();
  case APValue::Struct:
  case APValue::Union:
    return ConstRecordBuilder::BuildRecord(*this, Value, DestType);
  case APValue::ComplexFloat:
  case APValue::ComplexInt: {
    mlir::Type desiredType = CGM.convertType(DestType);
    cir::ComplexType complexType =
        mlir::dyn_cast<cir::ComplexType>(desiredType);

    mlir::Type complexElemTy = complexType.getElementType();
    if (isa<cir::IntType>(complexElemTy)) {
      llvm::APSInt real = Value.getComplexIntReal();
      llvm::APSInt imag = Value.getComplexIntImag();
      return builder.getAttr<cir::ComplexAttr>(
          complexType, cir::IntAttr::get(complexElemTy, real),
          cir::IntAttr::get(complexElemTy, imag));
    }

    assert(isa<cir::FPTypeInterface>(complexElemTy) &&
           "expected floating-point type");
    llvm::APFloat real = Value.getComplexFloatReal();
    llvm::APFloat imag = Value.getComplexFloatImag();
    return builder.getAttr<cir::ComplexAttr>(
        complexType, cir::FPAttr::get(complexElemTy, real),
        cir::FPAttr::get(complexElemTy, imag));
  }
  case APValue::FixedPoint:
  case APValue::AddrLabelDiff:
    assert(0 && "not implemented");
  }
  llvm_unreachable("Unknown APValue kind");
}

mlir::Value CIRGenModule::emitNullConstant(QualType T, mlir::Location loc) {
  if (T->getAs<PointerType>()) {
    return builder.getNullPtr(getTypes().convertTypeForMem(T), loc);
  }

  if (getTypes().isZeroInitializable(T))
    return builder.getNullValue(getTypes().convertTypeForMem(T), loc);

  if (getASTContext().getAsConstantArrayType(T)) {
    llvm_unreachable("NYI");
  }

  if (T->getAs<clang::RecordType>())
    llvm_unreachable("NYI");

  assert(T->isMemberDataPointerType() &&
         "Should only see pointers to data members here!");

  llvm_unreachable("NYI");
  return {};
}

mlir::Value CIRGenModule::emitMemberPointerConstant(const UnaryOperator *E) {
  assert(!cir::MissingFeatures::cxxABI());

  auto loc = getLoc(E->getSourceRange());

  const auto *decl = cast<DeclRefExpr>(E->getSubExpr())->getDecl();

  // A member function pointer.
  if (const auto *methodDecl = dyn_cast<CXXMethodDecl>(decl)) {
    auto ty = mlir::cast<cir::MethodType>(convertType(E->getType()));
    if (methodDecl->isVirtual())
      return builder.create<cir::ConstantOp>(
          loc, getCXXABI().buildVirtualMethodAttr(ty, methodDecl));

    auto methodFuncOp = GetAddrOfFunction(methodDecl);
    return builder.create<cir::ConstantOp>(
        loc, builder.getMethodAttr(ty, methodFuncOp));
  }

  auto ty = mlir::cast<cir::DataMemberType>(convertType(E->getType()));

  // Otherwise, a member data pointer.
  const auto *fieldDecl = cast<FieldDecl>(decl);
  return builder.create<cir::ConstantOp>(
      loc, builder.getDataMemberAttr(ty, fieldDecl->getFieldIndex()));
}

mlir::Attribute ConstantEmitter::emitAbstract(const Expr *E,
                                              QualType destType) {
  auto state = pushAbstract();
  auto C = mlir::cast<mlir::Attribute>(tryEmitPrivate(E, destType));
  C = validateAndPopAbstract(C, state);
  if (!C) {
    llvm_unreachable("NYI");
  }
  return C;
}

mlir::Attribute ConstantEmitter::emitAbstract(SourceLocation loc,
                                              const APValue &value,
                                              QualType destType) {
  auto state = pushAbstract();
  auto C = tryEmitPrivate(value, destType);
  C = validateAndPopAbstract(C, state);
  if (!C) {
    CGM.Error(loc,
              "internal error: could not emit constant value \"abstractly\"");
    llvm_unreachable("NYI");
  }
  return C;
}

mlir::Attribute ConstantEmitter::emitNullForMemory(mlir::Location loc,
                                                   CIRGenModule &CGM,
                                                   QualType T) {
  auto cstOp = CGM.emitNullConstant(T, loc).getDefiningOp<cir::ConstantOp>();
  assert(cstOp && "expected cir.const op");
  return emitForMemory(CGM, cstOp.getValue(), T);
}

static mlir::TypedAttr emitNullConstant(CIRGenModule &CGM, const RecordDecl *rd,
                                        bool asCompleteObject) {
  const CIRGenRecordLayout &layout = CGM.getTypes().getCIRGenRecordLayout(rd);
  mlir::Type ty = (asCompleteObject ? layout.getCIRType()
                                    : layout.getBaseSubobjectCIRType());
  auto record = dyn_cast<cir::RecordType>(ty);
  assert(record && "expected");

  unsigned numElements = record.getNumElements();
  SmallVector<mlir::Attribute, 4> elements(numElements);

  auto CXXR = dyn_cast<CXXRecordDecl>(rd);
  // Fill in all the bases.
  if (CXXR) {
    for (const auto &I : CXXR->bases()) {
      if (I.isVirtual()) {
        // Ignore virtual bases; if we're laying out for a complete
        // object, we'll lay these out later.
        continue;
      }
      llvm_unreachable("NYI");
    }
  }

  // Fill in all the fields.
  for (const auto *Field : rd->fields()) {
    // Fill in non-bitfields. (Bitfields always use a zero pattern, which we
    // will fill in later.)
    if (!Field->isBitField()) {
      // TODO(cir) check for !isEmptyFieldForLayout(CGM.getContext(), Field))
      llvm_unreachable("NYI");
    }

    // For unions, stop after the first named field.
    if (rd->isUnion()) {
      if (Field->getIdentifier())
        break;
      if (const auto *FieldRD = Field->getType()->getAsRecordDecl())
        if (FieldRD->findFirstNamedDataMember())
          break;
    }
  }

  // Fill in the virtual bases, if we're working with the complete object.
  if (CXXR && asCompleteObject) {
    for ([[maybe_unused]] const auto &I : CXXR->vbases()) {
      llvm_unreachable("NYI");
    }
  }

  // Now go through all other fields and zero them out.
  for (unsigned i = 0; i != numElements; ++i) {
    if (!elements[i]) {
      llvm_unreachable("NYI");
    }
  }

  mlir::MLIRContext *mlirContext = record.getContext();
  return cir::ConstRecordAttr::get(record,
                                   mlir::ArrayAttr::get(mlirContext, elements));
}

mlir::TypedAttr
CIRGenModule::emitNullConstantForBase(const CXXRecordDecl *Record) {
  return ::emitNullConstant(*this, Record, false);
}
