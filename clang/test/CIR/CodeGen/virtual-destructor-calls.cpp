
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O1 -fclangir -emit-cir %s -o %t-o1.cir
// RUN: FileCheck --check-prefix=CIR_O1 --input-file=%t-o1.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -std=c++20 -mconstructor-aliases -O0 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// FIXME: LLVM IR dialect does not yet support function ptr globals, which precludes
// a lot of the proper semantics for properly representing alias functions in LLVM
// (see the note on LLVM_O1 below).

struct Member {
  ~Member();
};

struct A {
  virtual ~A();
};

struct B : A {
  Member m;
  virtual ~B();
};

// Aliases are inserted before the function definitions in LLVM IR
// FIXME: These should have unnamed_addr set.
// LLVM: @_ZN1BD1Ev = alias void (ptr), ptr @_ZN1BD2Ev
// LLVM: @_ZN1CD1Ev = alias void (ptr), ptr @_ZN1CD2Ev

// Base dtor: actually calls A's base dtor.
// CIR: cir.func dso_local @_ZN1BD2Ev
// CIR:   cir.call @_ZN6MemberD1Ev
// CIR:   cir.call @_ZN1AD2Ev
// LLVM: define{{.*}} void @_ZN1BD2Ev(ptr
// LLVM: call void @_ZN6MemberD1Ev
// LLVM: call void @_ZN1AD2Ev

// Complete dtor: just an alias because there are no virtual bases.
// CIR: cir.func private dso_local @_ZN1BD1Ev(!cir.ptr<!rec_B>) special_member<#cir.cxx_dtor<!rec_B>> alias(@_ZN1BD2Ev)

// Deleting dtor: defers to the complete dtor.
// LLVM: define{{.*}} void @_ZN1BD0Ev(ptr
// LLVM: call void @_ZN1BD1Ev
// LLVM: call void @_ZdlPv

// (aliases from C)
// CIR: cir.func dso_local @_ZN1CD2Ev(%arg0: !cir.ptr<!rec_C>{{.*}})) {{.*}} {
// CIR: cir.func private dso_local @_ZN1CD1Ev(!cir.ptr<!rec_C>) special_member<#cir.cxx_dtor<!rec_C>> alias(@_ZN1CD2Ev)

// CIR_O1-NOT: cir.func dso_local @_ZN1CD2Ev(%arg0: !cir.ptr<!rec_C>{{.*}})) {{.*}} {
// CIR_O1: cir.func private dso_local @_ZN1CD2Ev(!cir.ptr<!rec_C>) special_member<#cir.cxx_dtor<!rec_C>> alias(@_ZN1BD2Ev)
// CIR_O1: cir.func private dso_local @_ZN1CD1Ev(!cir.ptr<!rec_C>) special_member<#cir.cxx_dtor<!rec_C>> alias(@_ZN1CD2Ev)

// FIXME: LLVM output should be: @_ZN1CD2Ev ={{.*}} unnamed_addr alias {{.*}} @_ZN1BD2Ev
// LLVM: define dso_local void @_ZN1CD2Ev(ptr
// FIXME: note that LLVM_O1 cannot be tested because the canocalizers running
// on top of LLVM IR dialect delete _ZN1CD2Ev in its current form (a function
// declaration) since its not used in the TU.

B::~B() { }

struct C : B {
  ~C();
};

C::~C() { }

// Complete dtor: just an alias (checked above).

// Deleting dtor: defers to the complete dtor.
// CIR: cir.func dso_local @_ZN1CD0Ev
// CIR: cir.call @_ZN1CD1Ev
// CIR: cir.call @_ZdlPvm
// LLVM: define{{.*}} void @_ZN1CD0Ev(ptr
// LLVM: call void @_ZN1CD1Ev
// LLVM: call void @_ZdlPv

// Base dtor: just an alias to B's base dtor.

namespace PR12798 {
  // A qualified call to a base class destructor should not undergo virtual
  // dispatch. Template instantiation used to lose the qualifier.
  struct A { virtual ~A(); };
  template<typename T> void f(T *p) { p->A::~A(); }

  // CIR: cir.func weak_odr @_ZN7PR127981fINS_1AEEEvPT_
  // CIR: cir.call @_ZN7PR127981AD1Ev
  // LLVM: define {{.*}} @_ZN7PR127981fINS_1AEEEvPT_(
  // LLVM: call void @_ZN7PR127981AD1Ev(
  template void f(A*);
}
