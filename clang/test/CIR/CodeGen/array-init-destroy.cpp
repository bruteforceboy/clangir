// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t1.cir 2>&1 | FileCheck -check-prefix=BEFORE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t2.cir 2>&1 | FileCheck -check-prefix=AFTER %s

void foo() noexcept;

class xpto {
public:
  xpto() {
    foo();
  }
  int i;
  float f;
  ~xpto() {
    foo();
  }
};

void x() {
  xpto array[2];
}

// BEFORE: cir.func dso_local @_Z1xv()
// BEFORE:   %[[ArrayAddr:.*]] = cir.alloca !cir.array<!rec_xpto x 2>

// BEFORE:   cir.array.ctor(%[[ArrayAddr]] : !cir.ptr<!cir.array<!rec_xpto x 2>>) {
// BEFORE:   ^bb0(%arg0: !cir.ptr<!rec_xpto>
// BEFORE:     cir.call @_ZN4xptoC1Ev(%arg0) : (!cir.ptr<!rec_xpto>) -> ()
// BEFORE:     cir.yield
// BEFORE:   }

// BEFORE:   cir.array.dtor(%[[ArrayAddr]] : !cir.ptr<!cir.array<!rec_xpto x 2>>) {
// BEFORE:   ^bb0(%arg0: !cir.ptr<!rec_xpto>
// BEFORE:     cir.call @_ZN4xptoD1Ev(%arg0) : (!cir.ptr<!rec_xpto>) -> ()
// BEFORE:     cir.yield
// BEFORE:   }

// AFTER: cir.func dso_local @_Z1xv()
// AFTER: %[[ArrayAddr0:.*]] = cir.alloca !cir.array<!rec_xpto x 2>
// AFTER: %[[ConstTwo:.*]] = cir.const #cir.int<2> : !u64i
// AFTER: %[[ArrayBegin:.*]] = cir.cast(array_to_ptrdecay, %[[ArrayAddr0]] : !cir.ptr<!cir.array<!rec_xpto x 2>>), !cir.ptr<!rec_xpto>
// AFTER: %[[ArrayPastEnd:.*]] = cir.ptr_stride(%[[ArrayBegin]] : !cir.ptr<!rec_xpto>, %[[ConstTwo]] : !u64i), !cir.ptr<!rec_xpto>
// AFTER: %[[TmpIdx:.*]] = cir.alloca !cir.ptr<!rec_xpto>, !cir.ptr<!cir.ptr<!rec_xpto>>, ["__array_idx"] {alignment = 1 : i64}
// AFTER: cir.store %[[ArrayBegin]], %[[TmpIdx]] : !cir.ptr<!rec_xpto>, !cir.ptr<!cir.ptr<!rec_xpto>>
// AFTER: cir.do {
// AFTER:   %[[ArrayElt:.*]] = cir.load %[[TmpIdx]] : !cir.ptr<!cir.ptr<!rec_xpto>>, !cir.ptr<!rec_xpto>
// AFTER:   %[[ConstOne:.*]] = cir.const #cir.int<1> : !u64i
// AFTER:   cir.call @_ZN4xptoC1Ev(%[[ArrayElt]]) : (!cir.ptr<!rec_xpto>) -> ()
// AFTER:   %[[NextElt:.*]] = cir.ptr_stride(%[[ArrayElt]] : !cir.ptr<!rec_xpto>, %[[ConstOne]] : !u64i), !cir.ptr<!rec_xpto>
// AFTER:   cir.store %[[NextElt]], %[[TmpIdx]] : !cir.ptr<!rec_xpto>, !cir.ptr<!cir.ptr<!rec_xpto>>
// AFTER:   cir.yield
// AFTER: } while {
// AFTER:   %[[ArrayElt:.*]] = cir.load %[[TmpIdx]] : !cir.ptr<!cir.ptr<!rec_xpto>>, !cir.ptr<!rec_xpto>
// AFTER:   %[[ExitCond:.*]] = cir.cmp(ne, %[[ArrayElt]], %[[ArrayPastEnd]]) : !cir.ptr<!rec_xpto>, !cir.bool
// AFTER:   cir.condition(%[[ExitCond]])
// AFTER: }

// AFTER: cir.do {
// AFTER:   cir.call @_ZN4xptoD1Ev({{.*}}) : (!cir.ptr<!rec_xpto>) -> ()
// AFTER: } while {
// AFTER: }

// AFTER: cir.return
