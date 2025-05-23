//===- HoistVectorAllocas.cpp - performs CIR vector alloca opt ------------===//
//
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
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace cir;

/*
** for (...) {
**   std::vector<T> v;
**   ...
** }
**
** -->
**
** {
** std::vector<T> v;
** for (...) {
**   v.clear();
**   ...
** }
** } // extra scope for the vector destructor
*/

namespace {

struct HoistVectorAllocasPass
    : public HoistVectorAllocasBase<HoistVectorAllocasPass> {
  HoistVectorAllocasPass() = default;
  void runOnOperation() override;

  static llvm::StringRef vectorCtor, vectorDtor, vectorClear;
};

llvm::StringRef HoistVectorAllocasPass::vectorCtor =
    llvm::StringRef("_ZNSt6vectorIiSaIiEEC2Ev");
llvm::StringRef HoistVectorAllocasPass::vectorDtor =
    llvm::StringRef("_ZNSt6vectorIiSaIiEED2Ev");
llvm::StringRef HoistVectorAllocasPass::vectorClear =
    llvm::StringRef("_ZNSt6vectorIiSaIiEE5clearEv");

bool isVectorConstructor(cir::CallOp callOp) {
  if (auto callee = callOp.getCallee())
    return callee->str() == HoistVectorAllocasPass::vectorCtor;
  return false;
}

std::optional<cir::ScopeOp> getOuterScope(cir::CallOp callOp) {
  assert(isVectorConstructor(callOp) && "expected a vector constructor call");

  if (auto loopBody = callOp.getOperation()->getParentOfType<cir::ScopeOp>())
    return loopBody.getOperation()->getParentOfType<cir::ScopeOp>();

  return std::nullopt;
}

std::optional<cir::CallOp>
getMatchingDestructorCall(cir::CallOp constructorCall, mlir::Value alloca) {
  for (auto user : alloca.getUsers())
    if (auto call = dyn_cast<cir::CallOp>(user))
      if (call.getOperation()->getParentOp() ==
              constructorCall.getOperation()->getParentOp() &&
          call.getCallee()->str() == HoistVectorAllocasPass::vectorDtor)
        return call;

  return std::nullopt;
}

// Rewrite only std::vector alloca's that are used
bool allocaHasNoUse(mlir::Value alloca) {
  for (auto user : alloca.getUsers()) {
    if (!isa<cir::CallOp>(user))
      return false;

    auto call = dyn_cast<cir::CallOp>(user);
    if (call.getCallee()->str() != HoistVectorAllocasPass::vectorCtor &&
        call.getCallee()->str() != HoistVectorAllocasPass::vectorDtor)
      return false;
  }

  return true;
}

bool hoisted(mlir::Value alloca) {
  for (auto user : alloca.getUsers())
    if (auto call = dyn_cast<cir::CallOp>(user))
      if (call.getCallee()->str() == HoistVectorAllocasPass::vectorClear)
        return true;

  return false;
}

void insertVectorClear(mlir::OpBuilder &rewriter, mlir::Value alloca,
                       cir::FuncOp clearFn) {
  rewriter.setInsertionPointAfter(alloca.getDefiningOp());

  cir::CallOp callOp = rewriter.create<cir::CallOp>(
      alloca.getLoc(), mlir::SymbolRefAttr::get(clearFn), cir::VoidType(),
      alloca, cir::CallingConv::C, cir::SideEffect::All);

  mlir::NamedAttrList empty;
  callOp->setAttr("extra_attrs", cir::ExtraFuncAttributesAttr::get(
                                     empty.getDictionary(alloca.getContext())));
}

void moveToOuterScope(mlir::OpBuilder &rewriter, cir::CallOp constructorCall,
                      cir::FuncOp clearFn) {
  auto alloca = constructorCall.getArgOperand(0);
  auto destructorCall = getMatchingDestructorCall(constructorCall, alloca);
  auto outerScope = getOuterScope(constructorCall);

  if (hoisted(alloca) || allocaHasNoUse(alloca) || !outerScope ||
      !destructorCall)
    return;

  insertVectorClear(rewriter, alloca, clearFn);

  alloca.getDefiningOp()->moveBefore(*outerScope);
  constructorCall.getOperation()->moveBefore(*outerScope);
  destructorCall->getOperation()->moveAfter(*outerScope);
}

// Attempts to reconstruct the original .clear() function as a runtime function.
// Not sure if this links properly
FuncOp buildRuntimeFunction(mlir::OpBuilder &builder, llvm::StringRef name,
                            mlir::Location loc, cir::FuncType type,
                            cir::GlobalLinkageKind linkage,
                            mlir::ModuleOp theModule) {
  builder.setInsertionPointToStart(theModule.getBody());
  FuncOp f = dyn_cast_or_null<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      theModule, StringAttr::get(theModule->getContext(), name)));
  if (!f) {
    f = builder.create<cir::FuncOp>(loc, name, type);
    f.setLinkageAttr(
        cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);
    mlir::NamedAttrList attrs;
    f.setExtraAttrsAttr(cir::ExtraFuncAttributesAttr::get(
        attrs.getDictionary(builder.getContext())));
  }
  return f;
}

void HoistVectorAllocasPass::runOnOperation() {
  llvm::SmallVector<cir::CallOp> callsToRewrite;
  getOperation()->walk([&](cir::CallOp callOp) {
    if (isVectorConstructor(callOp))
      callsToRewrite.push_back(callOp);
  });

  if (callsToRewrite.empty())
    return;

  if (auto theModule = dyn_cast<ModuleOp>(getOperation())) {
    mlir::OpBuilder rewriter(callsToRewrite[0].getContext());

    auto fnType =
        cir::FuncType::get({callsToRewrite[0].getArgOperand(0).getType()},
                           cir::VoidType::get(rewriter.getContext()));
    FuncOp clearFn = buildRuntimeFunction(
        rewriter, HoistVectorAllocasPass::vectorClear, theModule.getLoc(),
        fnType, cir::GlobalLinkageKind::ExternalLinkage, theModule);

    for (auto &call : callsToRewrite)
      moveToOuterScope(rewriter, call, clearFn);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createHoistVectorAllocasPass() {
  return std::make_unique<HoistVectorAllocasPass>();
}
