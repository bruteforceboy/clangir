//===- CIRSimplify.cpp - performs CIR canonicalization --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

namespace {

/// Removes branches between two blocks if it is the only branch.
///
/// From:
///   ^bb0:
///     cir.br ^bb1
///   ^bb1:  // pred: ^bb0
///     cir.return
///
/// To:
///   ^bb0:
///     cir.return
struct RemoveRedundantBranches : public OpRewritePattern<BrOp> {
  using OpRewritePattern<BrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrOp op,
                                PatternRewriter &rewriter) const final {
    Block *block = op.getOperation()->getBlock();
    Block *dest = op.getDest();

    if (isa<cir::LabelOp>(dest->front()))
      return failure();

    // Single edge between blocks: merge it.
    if (block->getNumSuccessors() == 1 &&
        dest->getSinglePredecessor() == block) {
      rewriter.eraseOp(op);
      rewriter.mergeBlocks(dest, block);
      return success();
    }

    return failure();
  }
};

struct RemoveEmptyScope : public OpRewritePattern<ScopeOp> {
  using OpRewritePattern<ScopeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScopeOp op,
                                PatternRewriter &rewriter) const final {
    // TODO: Remove this logic once CIR uses MLIR infrastructure to remove
    // trivially dead operations
    if (!op.isEmpty())
      return failure();

    Region *region = op.getRegions().front();
    if ((region && region->getBlocks().front().getOperations().size() == 1) &&
        !isa<YieldOp>(region->getBlocks().front().front()))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct RemoveEmptySwitch : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern<SwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SwitchOp op,
                                PatternRewriter &rewriter) const final {
    if (!(op.getBody().empty() || isa<YieldOp>(op.getBody().front().front())))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct RemoveTrivialTry : public OpRewritePattern<TryOp> {
  using OpRewritePattern<TryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TryOp op,
                                PatternRewriter &rewriter) const final {
    // FIXME: also check all catch regions are empty
    // return success(op.getTryRegion().hasOneBlock());
    return mlir::failure();

    // Move try body to the parent.
    assert(op.getTryRegion().hasOneBlock());

    Block *parentBlock = op.getOperation()->getBlock();
    mlir::Block *tryBody = &op.getTryRegion().getBlocks().front();
    YieldOp y = dyn_cast<YieldOp>(tryBody->getTerminator());
    assert(y && "expected well wrapped up try block");
    y->erase();

    rewriter.inlineBlockBefore(tryBody, parentBlock, Block::iterator(op));
    rewriter.eraseOp(op);
    return success();
  }
};

// Remove call exception with empty cleanups
struct SimplifyCallOp : public OpRewritePattern<CallOp> {
  using OpRewritePattern<CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CallOp op,
                                PatternRewriter &rewriter) const final {
    // Applicable to cir.call exception ... clean { cir.yield }
    mlir::Region *r = &op.getCleanup();
    if (r->empty() || !r->hasOneBlock())
      return failure();

    mlir::Block *b = &r->getBlocks().back();
    if (&b->back() != &b->front())
      return failure();

    if (!(isa<YieldOp>(&b->getOperations().back())))
      return failure();

    b = &op.getCleanup().back();
    rewriter.eraseOp(&b->back());
    rewriter.eraseBlock(b);
    return success();
  }
};

struct SimplifyPtrStrideOp : public OpRewritePattern<PtrStrideOp> {
  using OpRewritePattern<PtrStrideOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PtrStrideOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Value base = op.getBase(), index = op.getStride();
    if (auto castOp = base.getDefiningOp<cir::CastOp>()) {
      // REWRITE ptr_stride(cast array_to_ptrdecay %base), %index)
      //      => get_element %base[%index]
      if (castOp.getKind() != cir::CastKind::array_to_ptrdecay)
        return mlir::failure();

      base = castOp.getOperand();

    } else if (auto getElemOp = base.getDefiningOp<cir::GetElementOp>()) {
      // REWRITE ptr_stride(get_element %base[%index]), %stride)
      //      => get_element %base[%index + %stride]
      auto elemIndex = getElemOp.getIndex();
      if (elemIndex.getType() != index.getType())
        return mlir::failure();

      base = getElemOp.getBase();
      index = rewriter.create<cir::BinOp>(op->getLoc(), cir::BinOpKind::Add,
                                          elemIndex, index);

    } else {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<cir::GetElementOp>(op, op.getType(), base,
                                                   index);
    return mlir::success();
  }
};

struct SimplifyCastOp : public OpRewritePattern<cir::CastOp> {
  using OpRewritePattern<cir::CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(cir::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != cir::CastKind::array_to_ptrdecay)
      return mlir::failure();

    auto elemTy = cast<cir::PointerType>(op.getType()).getPointee();
    for (auto *user : op->getUsers()) {
      if (auto loadOp = dyn_cast<cir::LoadOp>(user)) {
        if (elemTy != loadOp.getType())
          return mlir::failure();
      } else if (auto storeOp = dyn_cast<cir::StoreOp>(user)) {
        if (storeOp.getAddr() != op.getResult() ||
            elemTy != storeOp.getValue().getType())
          return mlir::failure();
      } else if (isa<cir::GetMemberOp>(user)) {
        continue;
      } else {
        return mlir::failure();
      }
    }

    // REWRITE cast array_to_ptrdecay %base
    //      => get_element %base[0]
    auto *context = elemTy.getContext();
    CIRDataLayout dataLayout(op->getParentOfType<ModuleOp>());
    auto intType = cast<cir::IntType>(dataLayout.getIntType(context));
    auto zeroAttr = rewriter.getAttr<cir::IntAttr>(
        intType, llvm::APInt(intType.getWidth(), 0));
    auto index = rewriter.create<cir::ConstantOp>(op->getLoc(), zeroAttr);

    rewriter.replaceOpWithNewOp<cir::GetElementOp>(op, op.getType(),
                                                   op.getOperand(), index);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// CIRCanonicalizePass
//===----------------------------------------------------------------------===//

struct CIRCanonicalizePass : public CIRCanonicalizeBase<CIRCanonicalizePass> {
  using CIRCanonicalizeBase::CIRCanonicalizeBase;

  // The same operation rewriting done here could have been performed
  // by CanonicalizerPass (adding hasCanonicalizer for target Ops and
  // implementing the same from above in CIRDialects.cpp). However, it's
  // currently too aggressive for static analysis purposes, since it might
  // remove things where a diagnostic can be generated.
  //
  // FIXME: perhaps we can add one more mode to GreedyRewriteConfig to
  // disable this behavior.
  void runOnOperation() override;
};

void populateCIRCanonicalizePatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    RemoveRedundantBranches,
    RemoveEmptyScope,
    RemoveEmptySwitch,
    RemoveTrivialTry,
    SimplifyCallOp,
    SimplifyPtrStrideOp,
    SimplifyCastOp
  >(patterns.getContext());
  // clang-format on
}

void CIRCanonicalizePass::runOnOperation() {
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateCIRCanonicalizePatterns(patterns);

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    // CastOp, UnaryOp and VecExtractOp are here to perform a manual `fold` in
    // applyOpPatternsGreedily.
    if (isa<BrOp, BrCondOp, ScopeOp, SwitchOp, CastOp, TryOp, UnaryOp, SelectOp,
            ComplexCreateOp, ComplexRealOp, ComplexImagOp, CallOp, VecCmpOp,
            VecCreateOp, VecExtractOp, VecShuffleOp, VecShuffleDynamicOp,
            VecTernaryOp, PtrStrideOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRCanonicalizePass() {
  return std::make_unique<CIRCanonicalizePass>();
}
