// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int x(int y) {
  return y > 0 ? 3 : 5;
}

// CHECK: cir.func dso_local @_Z1xi
// CHECK:     %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:     cir.store %arg0, %0 : !s32i, !cir.ptr<!s32i>
// CHECK:     %2 = cir.load{{.*}} %0 : !cir.ptr<!s32i>, !s32i
// CHECK:     %3 = cir.const #cir.int<0> : !s32i
// CHECK:     %4 = cir.cmp(gt, %2, %3) : !s32i, !cir.bool
// CHECK:     %5 = cir.const #cir.int<3> : !s32i
// CHECK:     %6 = cir.const #cir.int<5> : !s32i
// CHECK:     %7 = cir.select if %4 then %5 else %6 : (!cir.bool, !s32i, !s32i) -> !s32i
// CHECK:     cir.store %7, %1 : !s32i, !cir.ptr<!s32i>
// CHECK:     %8 = cir.load{{.*}} %1 : !cir.ptr<!s32i>, !s32i
// CHECK:     cir.return %8 : !s32i
// CHECK:   }

typedef enum {
  API_A,
  API_EnumSize = 0x7fffffff
} APIType;

void oba(const char *);

void m(APIType api) {
  ((api == API_A) ? (static_cast<void>(0)) : oba("yo.cpp"));
}

// CHECK:  cir.func dso_local @_Z1m7APIType
// CHECK:    %0 = cir.alloca !u32i, !cir.ptr<!u32i>, ["api", init] {alignment = 4 : i64}
// CHECK:    cir.store %arg0, %0 : !u32i, !cir.ptr<!u32i>
// CHECK:    %1 = cir.load{{.*}} %0 : !cir.ptr<!u32i>, !u32i
// CHECK:    %2 = cir.cast(integral, %1 : !u32i), !s32i
// CHECK:    %3 = cir.const #cir.int<0> : !u32i
// CHECK:    %4 = cir.cast(integral, %3 : !u32i), !s32i
// CHECK:    %5 = cir.cmp(eq, %2, %4) : !s32i, !cir.bool
// CHECK:    cir.ternary(%5, true {
// CHECK:      %6 = cir.const #cir.int<0> : !s32i
// CHECK:      cir.yield
// CHECK:    }, false {
// CHECK:      %6 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 7>>
// CHECK:      %7 = cir.cast(array_to_ptrdecay, %6 : !cir.ptr<!cir.array<!s8i x 7>>), !cir.ptr<!s8i>
// CHECK:      cir.call @_Z3obaPKc(%7) : (!cir.ptr<!s8i>) -> ()
// CHECK:      cir.yield
// CHECK:    }) : (!cir.bool) -> ()
// CHECK:    cir.return
// CHECK:  }

int foo(int a, int b) {
  if (a < b ? 0 : a)
    return -1;
  return 0;
}

// CHECK:  cir.func dso_local @_Z3fooii
// CHECK:   [[A0:%.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK:   [[B0:%.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK:   [[CMP:%.*]] = cir.cmp(lt, [[A0]], [[B0]]) : !s32i, !cir.bool
// CHECK:   [[RES:%.*]] = cir.ternary([[CMP]], true {
// CHECK:     [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CHECK:     cir.yield [[ZERO]] : !s32i
// CHECK:   }, false {
// CHECK:     [[A1:%.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!s32i>, !s32i
// CHECK:     cir.yield [[A1]] : !s32i
// CHECK:   }) : (!cir.bool) -> !s32i
// CHECK:   [[RES_CAST:%.*]] = cir.cast(int_to_bool, [[RES]] : !s32i), !cir.bool
// CHECK:   cir.if [[RES_CAST]]

void maybe_has_side_effects();

bool func(bool a, bool b) {
    return (maybe_has_side_effects(), a) ?: b;
}

// CHECK:  cir.func dso_local @_Z4funcbb([[ARG_A:%.*]]: !cir.bool {{.*}}, [[ARG_B:%.*]]: !cir.bool {{.*}}
// CHECK:    [[ALLOC_A:%.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a", init]
// CHECK:    [[ALLOC_B:%.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CHECK:    [[ALLOC_RET:%.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["__retval"]
// CHECK:    cir.store [[ARG_A]], [[ALLOC_A]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:    cir.store [[ARG_B]], [[ALLOC_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:    cir.call @_Z22maybe_has_side_effectsv() : () -> ()
// CHECK:    [[A0:%.*]] = cir.load{{.*}} [[ALLOC_A]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:    [[RES:%.*]] = cir.ternary([[A0]], true {
// CHECK:      [[A1:%.*]] = cir.load{{.*}} [[ALLOC_A]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:      cir.yield [[A1]] : !cir.bool
// CHECK:    }, false {
// CHECK:      [[B0:%.*]] = cir.load{{.*}} [[ALLOC_B]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:      cir.yield [[B0]] : !cir.bool
// CHECK:    }) : (!cir.bool) -> !cir.bool
// CHECK:    cir.store [[RES]], [[ALLOC_RET]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:    [[R:%.*]] = cir.load{{.*}} [[ALLOC_RET]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:    cir.return [[R]] : !cir.bool
