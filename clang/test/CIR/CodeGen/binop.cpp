// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O1 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void b0(int a, int b) {
  int x = a * b;
  x = x / b;
  x = x % b;
  x = x + b;
  x = x - b;
  x = x >> b;
  x = x << b;
  x = x & b;
  x = x ^ b;
  x = x | b;
}

// CHECK: = cir.binop(mul, %3, %4) nsw : !s32i
// CHECK: = cir.binop(div, %6, %7) : !s32i
// CHECK: = cir.binop(rem, %9, %10) : !s32i
// CHECK: = cir.binop(add, %12, %13) nsw : !s32i
// CHECK: = cir.binop(sub, %15, %16) nsw : !s32i
// CHECK: = cir.shift(right, %18 : !s32i, %19 : !s32i) -> !s32i
// CHECK: = cir.shift(left, %21 : !s32i, %22 : !s32i) -> !s32i
// CHECK: = cir.binop(and, %24, %25) : !s32i
// CHECK: = cir.binop(xor, %27, %28) : !s32i
// CHECK: = cir.binop(or, %30, %31) : !s32i

void b1(bool a, bool b) {
  bool x = a && b;
  x = x || b;
}

// CHECK: cir.ternary(%3, true
// CHECK-NEXT: %7 = cir.load{{.*}} %1
// CHECK-NEXT: cir.yield %7
// CHECK-NEXT: false {
// CHECK-NEXT: cir.const #false
// CHECK-NEXT: cir.yield

// CHECK: cir.ternary(%5, true
// CHECK-NEXT: cir.const #true
// CHECK-NEXT: cir.yield
// CHECK-NEXT: false {
// CHECK-NEXT: %7 = cir.load{{.*}} %1
// CHECK-NEXT: cir.yield

void b2(bool a) {
 bool x = 0 && a;
 x = 1 && a;
 x = 0 || a;
 x = 1 || a;
}

// CHECK: %0 = cir.alloca {{.*}} ["a", init]
// CHECK: %1 = cir.alloca {{.*}} ["x", init]
// CHECK: %2 = cir.const #false
// CHECK-NEXT: cir.store{{.*}} %2, %1
// CHECK-NEXT: %3 = cir.load{{.*}} %0
// CHECK-NEXT: cir.store{{.*}} %3, %1
// CHECK-NEXT: %4 = cir.load{{.*}} %0
// CHECK-NEXT: cir.store{{.*}} %4, %1
// CHECK-NEXT: %5 = cir.const #true
// CHECK-NEXT: cir.store{{.*}} %5, %1

void b3(int a, int b, int c, int d) {
  bool x = (a == b) && (c == d);
  x = (a == b) || (c == d);
}

// CHECK: %0 = cir.alloca {{.*}} ["a", init]
// CHECK-NEXT: %1 = cir.alloca {{.*}} ["b", init]
// CHECK-NEXT: %2 = cir.alloca {{.*}} ["c", init]
// CHECK-NEXT: %3 = cir.alloca {{.*}} ["d", init]
// CHECK-NEXT: %4 = cir.alloca {{.*}} ["x", init]
// CHECK: %5 = cir.load{{.*}} %0
// CHECK-NEXT: %6 = cir.load{{.*}} %1
// CHECK-NEXT: %7 = cir.cmp(eq, %5, %6)
// CHECK-NEXT: cir.ternary(%7, true
// CHECK-NEXT: %13 = cir.load{{.*}} %2
// CHECK-NEXT: %14 = cir.load{{.*}} %3
// CHECK-NEXT: %15 = cir.cmp(eq, %13, %14)
// CHECK-NEXT: cir.yield %15
// CHECK-NEXT: }, false {
// CHECK-NEXT: %13 = cir.const #false
// CHECK-NEXT: cir.yield %13

void testFloatingPointBinOps(float a, float b) {
  a * b;
  // CHECK: cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.float
  a / b;
  // CHECK: cir.binop(div, %{{.+}}, %{{.+}}) : !cir.float
  a + b;
  // CHECK: cir.binop(add, %{{.+}}, %{{.+}}) : !cir.float
  a - b;
  // CHECK: cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.float
}

struct S {};

struct HasOpEq
{
  bool operator==(const S& other);
};

void rewritten_binop()
{
  HasOpEq s1;
  S s2;
  if (s1 != s2)
    return;
}

// CHECK-LABEL: _Z15rewritten_binopv
// CHECK:   cir.scope {
// CHECK:     cir.call @_ZN7HasOpEqeqERK1S
// CHECK:     %[[COND:.*]] = cir.unary(not
// CHECK:     cir.if %[[COND]]
// CHECK:       cir.return
