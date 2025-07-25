// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Bar {
  int a;
  char b;
} bar;

struct Foo {
  int a;
  char b;
  struct Bar z;
};

// Recursive type
typedef struct Node {
  struct Node* next;
} NodeStru;

void baz(void) {
  struct Bar b;
  struct Foo f;
}

// CHECK-DAG: !rec_Node = !cir.record<struct "Node" {!cir.ptr<!cir.record<struct "Node">>} #cir.record.decl.ast>
// CHECK-DAG: !rec_Bar = !cir.record<struct "Bar" {!s32i, !s8i}>
// CHECK-DAG: !rec_Foo = !cir.record<struct "Foo" {!s32i, !s8i, !rec_Bar}>
// CHECK-DAG: !rec_SLocal = !cir.record<struct "SLocal" {!s32i}>
// CHECK-DAG: !rec_SLocal2E0 = !cir.record<struct "SLocal.0" {!cir.float}>
//  CHECK-DAG: module {{.*}} {
     // CHECK:   cir.func dso_local @baz()
// CHECK-NEXT:     %0 = cir.alloca !rec_Bar, !cir.ptr<!rec_Bar>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.alloca !rec_Foo, !cir.ptr<!rec_Foo>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }

void shouldConstInitStructs(void) {
// CHECK: cir.func dso_local @shouldConstInitStructs
  struct Foo f = {1, 2, {3, 4}};
  // CHECK: %[[#V0:]] = cir.alloca !rec_Foo, !cir.ptr<!rec_Foo>, ["f"] {alignment = 4 : i64}
  // CHECK: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!rec_Foo>), !cir.ptr<!rec_anon_struct1>
  // CHECK: %[[#V2:]] = cir.const #cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s8i,
  // CHECK-SAME:        #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>,
  // CHECK-SAME:        #cir.const_record<{#cir.int<3> : !s32i, #cir.int<4> : !s8i,
  // CHECK-SAME:        #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>}>
  // CHECK-SAME:        : !rec_anon_struct}> : !rec_anon_struct1
  // CHECK: cir.store{{.*}} %[[#V2]], %[[#V1]] : !rec_anon_struct1, !cir.ptr<!rec_anon_struct1>
}

// Should zero-initialize uninitialized global structs.
struct S {
  int a,b;
} s;
// CHECK-DAG: cir.global external @s = #cir.zero : !rec_S

// Should initialize basic global structs.
struct S1 {
  int a;
  float f;
  int *p;
} s1 = {1, .1, 0};
// CHECK-DAG: cir.global external @s1 = #cir.const_record<{#cir.int<1> : !s32i, #cir.fp<1.000000e-01> : !cir.float, #cir.ptr<null> : !cir.ptr<!s32i>}> : !rec_S1

// Should initialize global nested structs.
struct S2 {
  struct S2A {
    int a;
  } s2a;
} s2 = {{1}};
// CHECK-DAG: cir.global external @s2 = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i}> : !rec_S2A}> : !rec_S2

// Should initialize global arrays of structs.
struct S3 {
  int a;
} s3[3] = {{1}, {2}, {3}};
// CHECK-DAG: cir.global external @s3 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i}> : !rec_S3, #cir.const_record<{#cir.int<2> : !s32i}> : !rec_S3, #cir.const_record<{#cir.int<3> : !s32i}> : !rec_S3]> : !cir.array<!rec_S3 x 3>

void shouldCopyStructAsCallArg(struct S1 s) {
// CHECK-DAG: cir.func dso_local @shouldCopyStructAsCallArg
  shouldCopyStructAsCallArg(s);
  // CHECK-DAG: %[[#LV:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!rec_S1>, !rec_S1
  // CHECK-DAG: cir.call @shouldCopyStructAsCallArg(%[[#LV]]) : (!rec_S1) -> ()
}

struct Bar shouldGenerateAndAccessStructArrays(void) {
  struct Bar s[1] = {{3, 4}};
  return s[0];
}
// CHECK-DAG: cir.func dso_local @shouldGenerateAndAccessStructArrays
// CHECK-DAG: %[[#STRIDE:]] = cir.const #cir.int<0> : !s32i
// CHECK-DAG: %[[#ELT:]] = cir.get_element %{{.+}}[%[[#STRIDE]]] : (!cir.ptr<!cir.array<!rec_Bar x 1>>, !s32i) -> !cir.ptr<!rec_Bar>
// CHECK-DAG: cir.copy %[[#ELT]] to %{{.+}} : !cir.ptr<!rec_Bar>

// CHECK-DAG: cir.func dso_local @local_decl
// CHECK-DAG: {{%.}} = cir.alloca !rec_Local, !cir.ptr<!rec_Local>, ["a"]
void local_decl(void) {
  struct Local {
    int i;
  };
  struct Local a;
}

// CHECK-DAG: cir.func dso_local @useRecursiveType
// CHECK-DAG: cir.get_member {{%.}}[0] {name = "next"} : !cir.ptr<!rec_Node> -> !cir.ptr<!cir.ptr<!rec_Node>>
void useRecursiveType(NodeStru* a) {
  a->next = 0;
}

// CHECK-DAG: cir.alloca !rec_SLocal, !cir.ptr<!rec_SLocal>, ["loc", init] {alignment = 4 : i64}
// CHECK-DAG: cir.scope {
// CHECK-DAG:   cir.alloca !rec_SLocal2E0, !cir.ptr<!rec_SLocal2E0>, ["loc", init] {alignment = 4 : i64}
void local_structs(int a, float b) {
  struct SLocal { int x; };
  struct SLocal loc = {a};
  {
    struct SLocal { float y; };
    struct SLocal loc = {b};
  }
}
