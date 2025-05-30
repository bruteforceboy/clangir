//===- ConstantInitBuilder.h - Builder for CIR attributes -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class provides a convenient interface for building complex
// global initializers of the sort that are frequently required for
// language ABIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CODEGEN_CONSTANTINITBUILDER_H
#define LLVM_CLANG_CIR_CODEGEN_CONSTANTINITBUILDER_H

#include "clang/AST/CharUnits.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "CIRGenBuilder.h"
#include "ConstantInitFuture.h"

#include <cstdint>
#include <vector>

namespace clang::CIRGen {

class CIRGenModule;

/// A convenience builder class for complex constant initializers,
/// especially for anonymous global structures used by various language
/// runtimes.
///
/// The basic usage pattern is expected to be something like:
///    ConstantInitBuilder builder(CGM);
///    auto toplevel = builder.beginRecord();
///    toplevel.addInt(CGM.SizeTy, widgets.size());
///    auto widgetArray = builder.beginArray();
///    for (auto &widget : widgets) {
///      auto widgetDesc = widgetArray.beginRecord();
///      widgetDesc.addInt(CGM.SizeTy, widget.getPower());
///      widgetDesc.add(CGM.GetAddrOfConstantString(widget.getName()));
///      widgetDesc.add(CGM.GetAddrOfGlobal(widget.getInitializerDecl()));
///      widgetDesc.finishAndAddTo(widgetArray);
///    }
///    widgetArray.finishAndAddTo(toplevel);
///    auto global = toplevel.finishAndCreateGlobal("WIDGET_LIST", Align,
///                                                 /*constant*/ true);
class ConstantInitBuilderBase {
  struct SelfReference {
    cir::GlobalOp Dummy;
    llvm::SmallVector<mlir::Attribute, 4> Indices;

    SelfReference(cir::GlobalOp dummy) : Dummy(dummy) {}
  };
  CIRGenModule &CGM;
  CIRGenBuilderTy &builder;
  llvm::SmallVector<mlir::Attribute, 16> Buffer;
  std::vector<SelfReference> SelfReferences;
  bool Frozen = false;

  friend class ConstantInitFuture;
  friend class ConstantAggregateBuilderBase;
  template <class, class> friend class ConstantAggregateBuilderTemplateBase;

protected:
  explicit ConstantInitBuilderBase(CIRGenModule &CGM);

  ~ConstantInitBuilderBase() {
    assert(Buffer.empty() && "didn't claim all values out of buffer");
    assert(SelfReferences.empty() && "didn't apply all self-references");
  }

private:
  cir::GlobalOp createGlobal(
      mlir::Attribute initializer, const llvm::Twine &name, CharUnits alignment,
      bool constant = false,
      cir::GlobalLinkageKind linkage = cir::GlobalLinkageKind::InternalLinkage,
      unsigned addressSpace = 0);

  ConstantInitFuture createFuture(mlir::Attribute initializer);

  void setGlobalInitializer(cir::GlobalOp GV, mlir::Attribute initializer);

  void resolveSelfReferences(cir::GlobalOp GV);

  void abandon(size_t newEnd);
};

/// A concrete base class for record and array aggregate
/// initializer builders.
class ConstantAggregateBuilderBase {
protected:
  ConstantInitBuilderBase &Builder;
  ConstantAggregateBuilderBase *Parent;
  size_t Begin;
  mutable size_t CachedOffsetEnd = 0;
  bool Finished = false;
  bool Frozen = false;
  bool Packed = false;
  mutable CharUnits CachedOffsetFromGlobal;

  llvm::SmallVectorImpl<mlir::Attribute> &getBuffer() { return Builder.Buffer; }

  const llvm::SmallVectorImpl<mlir::Attribute> &getBuffer() const {
    return Builder.Buffer;
  }

  ConstantAggregateBuilderBase(ConstantInitBuilderBase &builder,
                               ConstantAggregateBuilderBase *parent)
      : Builder(builder), Parent(parent), Begin(builder.Buffer.size()) {
    if (parent) {
      assert(!parent->Frozen && "parent already has child builder active");
      parent->Frozen = true;
    } else {
      assert(!builder.Frozen && "builder already has child builder active");
      builder.Frozen = true;
    }
  }

  ~ConstantAggregateBuilderBase() {
    assert(Finished && "didn't finish aggregate builder");
  }

  void markFinished() {
    assert(!Frozen && "child builder still active");
    assert(!Finished && "builder already finished");
    Finished = true;
    if (Parent) {
      assert(Parent->Frozen && "parent not frozen while child builder active");
      Parent->Frozen = false;
    } else {
      assert(Builder.Frozen && "builder not frozen while child builder active");
      Builder.Frozen = false;
    }
  }

public:
  // Not copyable.
  ConstantAggregateBuilderBase(const ConstantAggregateBuilderBase &) = delete;
  ConstantAggregateBuilderBase &
  operator=(const ConstantAggregateBuilderBase &) = delete;

  // Movable, mostly to allow returning.  But we have to write this out
  // properly to satisfy the assert in the destructor.
  ConstantAggregateBuilderBase(ConstantAggregateBuilderBase &&other)
      : Builder(other.Builder), Parent(other.Parent), Begin(other.Begin),
        CachedOffsetEnd(other.CachedOffsetEnd), Finished(other.Finished),
        Frozen(other.Frozen), Packed(other.Packed),
        CachedOffsetFromGlobal(other.CachedOffsetFromGlobal) {
    other.Finished = true;
  }
  ConstantAggregateBuilderBase &
  operator=(ConstantAggregateBuilderBase &&other) = delete;

  /// Return the number of elements that have been added to
  /// this record or array.
  size_t size() const {
    assert(!this->Finished && "cannot query after finishing builder");
    assert(!this->Frozen && "cannot query while sub-builder is active");
    assert(this->Begin <= this->getBuffer().size());
    return this->getBuffer().size() - this->Begin;
  }

  /// Return true if no elements have yet been added to this record or array.
  bool empty() const { return size() == 0; }

  /// Abandon this builder completely.
  void abandon() {
    markFinished();
    Builder.abandon(Begin);
  }

  /// Add a new value to this initializer.
  void add(mlir::Attribute value) {
    assert(value && "adding null value to constant initializer");
    assert(!Finished && "cannot add more values after finishing builder");
    assert(!Frozen && "cannot add values while subbuilder is active");
    Builder.Buffer.push_back(value);
  }

  /// Add an integer value of type size_t.
  void addSize(CharUnits size);

  /// Add an integer value of a specific type.
  void addInt(cir::IntType intTy, uint64_t value, bool isSigned = false) {
    llvm_unreachable("NYI");
  }

  /// Add a pointer of a specific type.
  void addPointer(cir::PointerType ptrTy, uint64_t value) {
    mlir::IntegerAttr val = Builder.builder.getI64IntegerAttr(value);
    add(cir::ConstPtrAttr::get(ptrTy, val));
  }

  /// Add a bitcast of a value to a specific type.
  void addBitCast(mlir::Attribute value, mlir::Type type) {
    llvm_unreachable("NYI");
    // add(llvm::ConstantExpr::getBitCast(value, type));
  }

  /// Add a bunch of new values to this initializer.
  void addAll(llvm::ArrayRef<mlir::Attribute> values) {
    assert(!Finished && "cannot add more values after finishing builder");
    assert(!Frozen && "cannot add values while subbuilder is active");
    Builder.Buffer.append(values.begin(), values.end());
  }

  /// Add a relative offset to the given target address, i.e. the
  /// static difference between the target address and the address
  /// of the relative offset.  The target must be known to be defined
  /// in the current linkage unit.  The offset will have the given
  /// integer type, which must be no wider than intptr_t.  Some
  /// targets may not fully support this operation.
  void addRelativeOffset(cir::IntType type, mlir::Attribute target) {
    llvm_unreachable("NYI");
    // add(getRelativeOffset(type, target));
  }

  /// Same as addRelativeOffset(), but instead relative to an element in this
  /// aggregate, identified by its index.
  void addRelativeOffsetToPosition(cir::IntType type, mlir::Attribute target,
                                   size_t position) {
    llvm_unreachable("NYI");
    // add(getRelativeOffsetToPosition(type, target, position));
  }

  /// Add a relative offset to the target address, plus a small
  /// constant offset.  This is primarily useful when the relative
  /// offset is known to be a multiple of (say) four and therefore
  /// the tag can be used to express an extra two bits of information.
  void addTaggedRelativeOffset(cir::IntType type, mlir::Attribute address,
                               unsigned tag) {
    llvm_unreachable("NYI");
    // mlir::Attribute offset =
    // getRelativeOffset(type, address); if
    // (tag) {
    //   offset =
    //       llvm::ConstantExpr::getAdd(offset,
    //       llvm::ConstantInt::get(type, tag));
    // }
    // add(offset);
  }

  /// Return the offset from the start of the initializer to the
  /// next position, assuming no padding is required prior to it.
  ///
  /// This operation will not succeed if any unsized placeholders are
  /// currently in place in the initializer.
  CharUnits getNextOffsetFromGlobal() const {
    assert(!Finished && "cannot add more values after finishing builder");
    assert(!Frozen && "cannot add values while subbuilder is active");
    return getOffsetFromGlobalTo(Builder.Buffer.size());
  }

  /// An opaque class to hold the abstract position of a placeholder.
  class PlaceholderPosition {
    size_t Index;
    friend class ConstantAggregateBuilderBase;
    PlaceholderPosition(size_t index) : Index(index) {}
  };

  /// Add a placeholder value to the record.  The returned position
  /// can be used to set the value later; it will not be invalidated by
  /// any intermediate operations except (1) filling the same position or
  /// (2) finishing the entire builder.
  ///
  /// This is useful for emitting certain kinds of record which
  /// contain some sort of summary field, generally a count, before any
  /// of the data.  By emitting a placeholder first, the record can
  /// be emitted eagerly.
  PlaceholderPosition addPlaceholder() {
    assert(!Finished && "cannot add more values after finishing builder");
    assert(!Frozen && "cannot add values while subbuilder is active");
    Builder.Buffer.push_back(nullptr);
    return Builder.Buffer.size() - 1;
  }

  /// Add a placeholder, giving the expected type that will be filled in.
  PlaceholderPosition addPlaceholderWithSize(mlir::Type expectedType);

  /// Fill a previously-added placeholder.
  void fillPlaceholderWithInt(PlaceholderPosition position, cir::IntType type,
                              uint64_t value, bool isSigned = false) {
    llvm_unreachable("NYI");
    // fillPlaceholder(position, llvm::ConstantInt::get(type, value, isSigned));
  }

  /// Fill a previously-added placeholder.
  void fillPlaceholder(PlaceholderPosition position, mlir::Attribute value) {
    assert(!Finished && "cannot change values after finishing builder");
    assert(!Frozen && "cannot add values while subbuilder is active");
    mlir::Attribute &slot = Builder.Buffer[position.Index];
    assert(slot == nullptr && "placeholder already filled");
    slot = value;
  }

  /// Produce an address which will eventually point to the next
  /// position to be filled.  This is computed with an indexed
  /// getelementptr rather than by computing offsets.
  ///
  /// The returned pointer will have type T*, where T is the given type. This
  /// type can differ from the type of the actual element.
  mlir::Attribute getAddrOfCurrentPosition(mlir::Type type);

  /// Produce an address which points to a position in the aggregate being
  /// constructed. This is computed with an indexed getelementptr rather than by
  /// computing offsets.
  ///
  /// The returned pointer will have type T*, where T is the given type. This
  /// type can differ from the type of the actual element.
  mlir::Attribute getAddrOfPosition(mlir::Type type, size_t position);

  llvm::ArrayRef<mlir::Attribute> getGEPIndicesToCurrentPosition(
      llvm::SmallVectorImpl<mlir::Attribute> &indices) {
    getGEPIndicesTo(indices, Builder.Buffer.size());
    return indices;
  }

protected:
  mlir::Attribute finishArray(mlir::Type eltTy);
  mlir::Attribute finishRecord(mlir::MLIRContext *mlirContext,
                               cir::RecordType recordTy);

private:
  void getGEPIndicesTo(llvm::SmallVectorImpl<mlir::Attribute> &indices,
                       size_t position) const;

  mlir::Attribute getRelativeOffset(cir::IntType offsetType,
                                    mlir::Attribute target);

  mlir::Attribute getRelativeOffsetToPosition(cir::IntType offsetType,
                                              mlir::Attribute target,
                                              size_t position);

  CharUnits getOffsetFromGlobalTo(size_t index) const;
};

template <class Impl, class Traits>
class ConstantAggregateBuilderTemplateBase
    : public Traits::AggregateBuilderBase {
  using super = typename Traits::AggregateBuilderBase;

public:
  using InitBuilder = typename Traits::InitBuilder;
  using ArrayBuilder = typename Traits::ArrayBuilder;
  using RecordBuilder = typename Traits::RecordBuilder;
  using AggregateBuilderBase = typename Traits::AggregateBuilderBase;

protected:
  ConstantAggregateBuilderTemplateBase(InitBuilder &builder,
                                       AggregateBuilderBase *parent)
      : super(builder, parent) {}

  Impl &asImpl() { return *static_cast<Impl *>(this); }

public:
  ArrayBuilder beginArray(mlir::Type eltTy = nullptr) {
    return ArrayBuilder(static_cast<InitBuilder &>(this->Builder), this, eltTy);
  }

  RecordBuilder beginRecord(cir::RecordType ty = nullptr) {
    return RecordBuilder(static_cast<InitBuilder &>(this->Builder), this, ty);
  }

  /// Given that this builder was created by beginning an array or record
  /// component on the given parent builder, finish the array/record
  /// component and add it to the parent.
  ///
  /// It is an intentional choice that the parent is passed in explicitly
  /// despite it being redundant with information already kept in the
  /// builder.  This aids in readability by making it easier to find the
  /// places that add components to a builder, as well as "bookending"
  /// the sub-builder more explicitly.
  void finishAndAddTo(mlir::MLIRContext *mlirContext,
                      AggregateBuilderBase &parent) {
    assert(this->Parent == &parent && "adding to non-parent builder");
    parent.add(asImpl().finishImpl(mlirContext));
  }

  /// Given that this builder was created by beginning an array or record
  /// directly on a ConstantInitBuilder, finish the array/record and
  /// create a global variable with it as the initializer.
  template <class... As>
  cir::GlobalOp finishAndCreateGlobal(mlir::MLIRContext *mlirContext,
                                      As &&...args) {
    assert(!this->Parent && "finishing non-root builder");
    return this->Builder.createGlobal(asImpl().finishImpl(mlirContext),
                                      std::forward<As>(args)...);
  }

  /// Given that this builder was created by beginning an array or record
  /// directly on a ConstantInitBuilder, finish the array/record and
  /// set it as the initializer of the given global variable.
  void finishAndSetAsInitializer(cir::GlobalOp global, bool forVTable = false) {
    assert(!this->Parent && "finishing non-root builder");
    mlir::Attribute init = asImpl().finishImpl(global.getContext());
    auto initCSA = mlir::dyn_cast<cir::ConstRecordAttr>(init);
    assert(initCSA &&
           "expected #cir.const_record attribute to represent vtable data");
    return this->Builder.setGlobalInitializer(
        global, forVTable ? cir::VTableAttr::get(initCSA.getType(),
                                                 initCSA.getMembers())
                          : init);
  }

  /// Given that this builder was created by beginning an array or record
  /// directly on a ConstantInitBuilder, finish the array/record and
  /// return a future which can be used to install the initializer in
  /// a global later.
  ///
  /// This is useful for allowing a finished initializer to passed to
  /// an API which will build the global.  However, the "future" preserves
  /// a dependency on the original builder; it is an error to pass it aside.
  ConstantInitFuture finishAndCreateFuture(mlir::MLIRContext *mlirContext) {
    assert(!this->Parent && "finishing non-root builder");
    return this->Builder.createFuture(asImpl().finishImpl(mlirContext));
  }
};

template <class Traits>
class ConstantArrayBuilderTemplateBase
    : public ConstantAggregateBuilderTemplateBase<typename Traits::ArrayBuilder,
                                                  Traits> {
  using super =
      ConstantAggregateBuilderTemplateBase<typename Traits::ArrayBuilder,
                                           Traits>;

public:
  using InitBuilder = typename Traits::InitBuilder;
  using AggregateBuilderBase = typename Traits::AggregateBuilderBase;

private:
  mlir::Type EltTy;

  template <class, class> friend class ConstantAggregateBuilderTemplateBase;

protected:
  ConstantArrayBuilderTemplateBase(InitBuilder &builder,
                                   AggregateBuilderBase *parent,
                                   mlir::Type eltTy)
      : super(builder, parent), EltTy(eltTy) {}

private:
  /// Form an array constant from the values that have been added to this
  /// builder.
  mlir::Attribute finishImpl([[maybe_unused]] mlir::MLIRContext *mlirContext) {
    return AggregateBuilderBase::finishArray(EltTy);
  }
};

/// A template class designed to allow other frontends to
/// easily customize the builder classes used by ConstantInitBuilder,
/// and thus to extend the API to work with the abstractions they
/// prefer.  This would probably not be necessary if C++ just
/// supported extension methods.
template <class Traits>
class ConstantRecordBuilderTemplateBase
    : public ConstantAggregateBuilderTemplateBase<
          typename Traits::RecordBuilder, Traits> {
  using super =
      ConstantAggregateBuilderTemplateBase<typename Traits::RecordBuilder,
                                           Traits>;

public:
  using InitBuilder = typename Traits::InitBuilder;
  using AggregateBuilderBase = typename Traits::AggregateBuilderBase;

private:
  cir::RecordType RecordTy;

  template <class, class> friend class ConstantAggregateBuilderTemplateBase;

protected:
  ConstantRecordBuilderTemplateBase(InitBuilder &builder,
                                    AggregateBuilderBase *parent,
                                    cir::RecordType recordTy)
      : super(builder, parent), RecordTy(recordTy) {
    if (recordTy) {
      llvm_unreachable("NYI");
      // this->Packed = recordTy->isPacked();
    }
  }

public:
  void setPacked(bool packed) { this->Packed = packed; }

  /// Use the given type for the record if its element count is correct.
  /// Don't add more elements after calling this.
  void suggestType(cir::RecordType recordTy) {
    if (this->size() == recordTy.getNumElements()) {
      RecordTy = recordTy;
    }
  }

private:
  /// Form an array constant from the values that have been added to this
  /// builder.
  mlir::Attribute finishImpl(mlir::MLIRContext *mlirContext) {
    return AggregateBuilderBase::finishRecord(mlirContext, RecordTy);
  }
};

/// A template class designed to allow other frontends to
/// easily customize the builder classes used by ConstantInitBuilder,
/// and thus to extend the API to work with the abstractions they
/// prefer.  This would probably not be necessary if C++ just
/// supported extension methods.
template <class Traits>
class ConstantInitBuilderTemplateBase : public ConstantInitBuilderBase {
protected:
  ConstantInitBuilderTemplateBase(CIRGenModule &CGM)
      : ConstantInitBuilderBase(CGM) {}

public:
  using InitBuilder = typename Traits::InitBuilder;
  using ArrayBuilder = typename Traits::ArrayBuilder;
  using RecordBuilder = typename Traits::RecordBuilder;

  ArrayBuilder beginArray(mlir::Type eltTy = nullptr) {
    return ArrayBuilder(static_cast<InitBuilder &>(*this), nullptr, eltTy);
  }

  RecordBuilder beginRecord(cir::RecordType recordTy = nullptr) {
    return RecordBuilder(static_cast<InitBuilder &>(*this), nullptr, recordTy);
  }
};

class ConstantInitBuilder;
class ConstantRecordBuilder;
class ConstantArrayBuilder;

struct ConstantInitBuilderTraits {
  using InitBuilder = ConstantInitBuilder;
  using AggregateBuilderBase = ConstantAggregateBuilderBase;
  using ArrayBuilder = ConstantArrayBuilder;
  using RecordBuilder = ConstantRecordBuilder;
};

/// The standard implementation of ConstantInitBuilder used in Clang.
class ConstantInitBuilder
    : public ConstantInitBuilderTemplateBase<ConstantInitBuilderTraits> {
public:
  explicit ConstantInitBuilder(CIRGenModule &CGM)
      : ConstantInitBuilderTemplateBase(CGM) {}
};

/// A helper class of ConstantInitBuilder, used for building constant
/// array initializers.
class ConstantArrayBuilder
    : public ConstantArrayBuilderTemplateBase<ConstantInitBuilderTraits> {
  template <class Traits> friend class ConstantInitBuilderTemplateBase;

  // The use of explicit qualification is a GCC workaround.
  template <class Impl, class Traits>
  friend class ConstantAggregateBuilderTemplateBase;

  ConstantArrayBuilder(ConstantInitBuilder &builder,
                       ConstantAggregateBuilderBase *parent, mlir::Type eltTy)
      : ConstantArrayBuilderTemplateBase(builder, parent, eltTy) {}
};

/// A helper class of ConstantInitBuilder, used for building constant
/// record initializers.
class ConstantRecordBuilder
    : public ConstantRecordBuilderTemplateBase<ConstantInitBuilderTraits> {
  template <class Traits> friend class ConstantInitBuilderTemplateBase;

  // The use of explicit qualification is a GCC workaround.
  template <class Impl, class Traits>
  friend class ConstantAggregateBuilderTemplateBase;

  ConstantRecordBuilder(ConstantInitBuilder &builder,
                        ConstantAggregateBuilderBase *parent,
                        cir::RecordType recordTy)
      : ConstantRecordBuilderTemplateBase(builder, parent, recordTy) {}
};

} // namespace clang::CIRGen

#endif
