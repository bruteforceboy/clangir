// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIRFLAT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir=core -mmlir --mlir-print-ir-after-all %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIRMLIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -mmlir --mlir-print-ir-after-all -mllvm -print-after-all  %s -o %t.ll 2>&1 | FileCheck %s -check-prefix=CIR -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-drop-ast %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIRPASS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat -mmlir --mlir-print-ir-before=cir-flatten-cfg %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CFGPASS

int foo(void) {
  int i = 3;
  return i;
}


// CIR:  IR Dump After CIRCanonicalize (cir-canonicalize)
// CIR:  cir.func dso_local @foo() -> !s32i
// CIR:  IR Dump After LoweringPrepare (cir-lowering-prepare)
// CIR:  cir.func dso_local @foo() -> !s32i
// CIR-NOT: IR Dump After FlattenCFG
// CIR-NOT: IR Dump After SCFPrepare
// CIR:  IR Dump After DropAST (cir-drop-ast)
// CIR:  cir.func dso_local @foo() -> !s32i
// CIRFLAT:  IR Dump After CIRCanonicalize (cir-canonicalize)
// CIRFLAT:  cir.func dso_local @foo() -> !s32i
// CIRFLAT:  IR Dump After LoweringPrepare (cir-lowering-prepare)
// CIRFLAT:  cir.func dso_local @foo() -> !s32i
// CIRFLAT:  IR Dump After FlattenCFG (cir-flatten-cfg)
// CIRFLAT:  IR Dump After DropAST (cir-drop-ast)
// CIRFLAT:  cir.func dso_local @foo() -> !s32i
// CIRMLIR:  IR Dump After CIRCanonicalize (cir-canonicalize)
// CIRMLIR:  IR Dump After LoweringPrepare (cir-lowering-prepare)
// CIRMLIR:  IR Dump After SCFPrepare (cir-mlir-scf-prepare
// CIRMLIR:  IR Dump After DropAST (cir-drop-ast)
// LLVM: IR Dump After cir::direct::ConvertCIRToLLVMPass (cir-flat-to-llvm)
// LLVM: llvm.func @foo() -> i32
// LLVM: IR Dump After
// LLVM: define dso_local i32 @foo()

// CIRPASS-NOT:  IR Dump After CIRCanonicalize
// CIRPASS:      IR Dump After DropAST

// CFGPASS: IR Dump Before FlattenCFG (cir-flatten-cfg)
