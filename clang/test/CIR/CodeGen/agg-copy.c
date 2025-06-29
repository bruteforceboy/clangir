// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct {} S;

typedef struct {
    int a;
    int b;
    S s;
} A;

// CHECK: cir.func dso_local @foo1
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["a1", init]
// CHECK:   [[TMP1:%.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["a2", init]
// CHECK:   cir.store{{.*}} %arg0, [[TMP0]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CHECK:   cir.store{{.*}} %arg1, [[TMP1]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CHECK:   [[TMP2:%.*]] = cir.load{{.*}} [[TMP0]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CHECK:   [[TMP3:%.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   [[TMP4:%.*]] = cir.ptr_stride([[TMP2]] : !cir.ptr<!rec_A>, [[TMP3]] : !s32i), !cir.ptr<!rec_A>
// CHECK:   [[TMP5:%.*]] = cir.load{{.*}} [[TMP1]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CHECK:   [[TMP6:%.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   [[TMP7:%.*]] = cir.ptr_stride([[TMP5]] : !cir.ptr<!rec_A>, [[TMP6]] : !s32i), !cir.ptr<!rec_A>
// CHECK:   cir.copy [[TMP7]] to [[TMP4]] : !cir.ptr<!rec_A>
void foo1(A* a1, A* a2) {
    a1[1] = a2[1];
}

// CHECK: cir.func dso_local @foo2
// CHECK:    [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["a1", init]
// CHECK:    [[TMP1:%.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["a2", init]
// CHECK:    cir.store{{.*}} %arg0, [[TMP0]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CHECK:    cir.store{{.*}} %arg1, [[TMP1]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CHECK:    [[TMP2:%.*]] = cir.load{{.*}} [[TMP0]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CHECK:    [[TMP3:%.*]] = cir.get_member [[TMP2]][2] {name = "s"} : !cir.ptr<!rec_A> -> !cir.ptr<!rec_S>
// CHECK:    [[TMP4:%.*]] = cir.load{{.*}} [[TMP1]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CHECK:    [[TMP5:%.*]] = cir.get_member [[TMP4]][2] {name = "s"} : !cir.ptr<!rec_A> -> !cir.ptr<!rec_S>
// CHECK:    cir.copy [[TMP5]] to [[TMP3]] : !cir.ptr<!rec_S>
void foo2(A* a1, A* a2) {
    a1->s = a2->s;
}

// CHECK: cir.global external @a = #cir.zero : !rec_A
// CHECK: cir.func dso_local @foo3
// CHECK:    [[TMP0]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["__retval"] {alignment = 4 : i64}
// CHECK:    [[TMP1]] = cir.get_global @a : !cir.ptr<!rec_A>
// CHECK:    cir.copy [[TMP1]] to [[TMP0]] : !cir.ptr<!rec_A>
// CHECK:    [[TMP2]] = cir.load{{.*}} [[TMP0]] : !cir.ptr<!rec_A>, !rec_A
// CHECK:    cir.return [[TMP2]] : !rec_A
A a;
A foo3(void) {
    return a;
}

// CHECK: cir.func dso_local @foo4
// CHECK:    [[TMP0]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["a1", init]
// CHECK:    [[TMP1]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a2", init]
// CHECK:    cir.store{{.*}} %arg0, [[TMP0]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CHECK:    [[TMP2]] = cir.load deref{{.*}}  [[TMP0]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CHECK:    cir.copy [[TMP2]] to [[TMP1]] : !cir.ptr<!rec_A>
void foo4(A* a1) {
    A a2 = *a1;
}

A create() { A a; return a; }

// CHECK: cir.func {{.*@foo5}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>,
// CHECK:   [[TMP1:%.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["tmp"] {alignment = 4 : i64}
// CHECK:   [[TMP2:%.*]] = cir.call @create() : () -> !rec_A
// CHECK:   cir.store{{.*}} [[TMP2]], [[TMP1]] : !rec_A, !cir.ptr<!rec_A>
// CHECK:   cir.copy [[TMP1]] to [[TMP0]] : !cir.ptr<!rec_A>
void foo5() {
    A a;
    a = create();
}

void foo6(A* a1) {
  A a2 = (*a1);
// CHECK: cir.func {{.*@foo6}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["a1", init] {alignment = 8 : i64}
// CHECK:   [[TMP1:%.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a2", init] {alignment = 4 : i64}
// CHECK:   cir.store{{.*}} %arg0, [[TMP0]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CHECK:   [[TMP2:%.*]] = cir.load deref{{.*}}  [[TMP0]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CHECK:   cir.copy [[TMP2]] to [[TMP1]] : !cir.ptr<!rec_A>
}

volatile A vol_a;
A foo7() {
  return vol_a;
}
// CHECK: cir.func {{.*@foo7}}
// CHECK:   %0 = cir.alloca
// CHECK:   %1 = cir.get_global @vol_a
// CHECK:   cir.copy %1 to %0 volatile
