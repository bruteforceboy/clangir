//===---- CIRGenAction.h - CIR Code Generation Frontend Action -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CIRGENACTION_H
#define LLVM_CLANG_CIR_CIRGENACTION_H

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/FrontendAction.h"
#include <memory>

namespace llvm {
class LLVMIRContext;
class Module;
} // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T> class OwningOpRef;
} // namespace mlir

namespace cir {
class CIRGenConsumer;
class CIRGenerator;

class CIRGenAction : public clang::ASTFrontendAction {
public:
  enum class OutputType {
    EmitAssembly,
    EmitMLIR,
    EmitLLVM,
    EmitBC,
    EmitObj,
    None
  };

private:
  friend class CIRGenConsumer;

  // TODO: this is redundant but just using the OwningModuleRef requires more of
  // clang against MLIR. Hide this somewhere else.
  std::unique_ptr<mlir::OwningOpRef<mlir::ModuleOp>> mlirModule;
  std::unique_ptr<llvm::Module> llvmModule;

  mlir::MLIRContext *mlirContext;

  mlir::OwningOpRef<mlir::ModuleOp> loadModule(llvm::MemoryBufferRef mbRef);

protected:
  CIRGenAction(OutputType action, mlir::MLIRContext *_MLIRContext = nullptr);

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override;

  void ExecuteAction() override;

  void EndSourceFileAction() override;

public:
  ~CIRGenAction() override;

  virtual bool hasCIRSupport() const override { return true; }

  CIRGenConsumer *cgConsumer;
  OutputType action;
};

class EmitCIROnlyAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitCIROnlyAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitMLIRAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitMLIRAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitLLVMAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitLLVMAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitBCAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitBCAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitAssemblyAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitAssemblyAction(mlir::MLIRContext *mlirCtx = nullptr);
};

class EmitObjAction : public CIRGenAction {
  virtual void anchor();

public:
  EmitObjAction(mlir::MLIRContext *mlirCtx = nullptr);
};

// Used for -fclangir-analysis-only: use CIR analysis but still use original
// LLVM codegen path
class AnalysisOnlyActionBase : public clang::CodeGenAction {
  virtual void anchor();

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override;

  AnalysisOnlyActionBase(unsigned _Act,
                         llvm::LLVMContext *_VMContext = nullptr);
};

class AnalysisOnlyAndEmitAssemblyAction : public AnalysisOnlyActionBase {
  virtual void anchor() override;

public:
  AnalysisOnlyAndEmitAssemblyAction(llvm::LLVMContext *_VMContext = nullptr);
};

class AnalysisOnlyAndEmitBCAction : public AnalysisOnlyActionBase {
  virtual void anchor() override;

public:
  AnalysisOnlyAndEmitBCAction(llvm::LLVMContext *_VMContext = nullptr);
};

class AnalysisOnlyAndEmitLLVMAction : public AnalysisOnlyActionBase {
  virtual void anchor() override;

public:
  AnalysisOnlyAndEmitLLVMAction(llvm::LLVMContext *_VMContext = nullptr);
};

class AnalysisOnlyAndEmitLLVMOnlyAction : public AnalysisOnlyActionBase {
  virtual void anchor() override;

public:
  AnalysisOnlyAndEmitLLVMOnlyAction(llvm::LLVMContext *_VMContext = nullptr);
};

class AnalysisOnlyAndEmitCodeGenOnlyAction : public AnalysisOnlyActionBase {
  virtual void anchor() override;

public:
  AnalysisOnlyAndEmitCodeGenOnlyAction(llvm::LLVMContext *_VMContext = nullptr);
};

class AnalysisOnlyAndEmitObjAction : public AnalysisOnlyActionBase {
  virtual void anchor() override;

public:
  AnalysisOnlyAndEmitObjAction(llvm::LLVMContext *_VMContext = nullptr);
};
} // namespace cir

#endif
