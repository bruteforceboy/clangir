// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Point {
  int x;
  int y;
  int z;
};
// CHECK-DAG: !rec_Point = !cir.record<struct "Point" {!s32i, !s32i, !s32i}

struct Incomplete;
// CHECK-DAG: !rec_Incomplete = !cir.record<struct "Incomplete" incomplete>

int Point::*pt_member = &Point::x;
// CHECK: cir.global external @pt_member = #cir.data_member<0> : !cir.data_member<!s32i in !rec_Point>

auto test1() -> int Point::* {
  return &Point::y;
}
// CHECK: cir.func dso_local @_Z5test1v() -> !cir.data_member<!s32i in !rec_Point>
// CHECK:   %{{.+}} = cir.const #cir.data_member<1> : !cir.data_member<!s32i in !rec_Point>
// CHECK: }

int test2(const Point &pt, int Point::*member) {
  return pt.*member;
}
// CHECK: cir.func dso_local @_Z5test2RK5PointMS_i
// CHECK:   %{{.+}} = cir.get_runtime_member %{{.+}}[%{{.+}} : !cir.data_member<!s32i in !rec_Point>] : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CHECK: }

int test3(const Point *pt, int Point::*member) {
  return pt->*member;
}
// CHECK: cir.func dso_local @_Z5test3PK5PointMS_i
// CHECK:   %{{.+}} = cir.get_runtime_member %{{.+}}[%{{.+}} : !cir.data_member<!s32i in !rec_Point>] : !cir.ptr<!rec_Point> -> !cir.ptr<!s32i>
// CHECK: }

auto test4(int Incomplete::*member) -> int Incomplete::* {
  return member;
}
// CHECK: cir.func dso_local @_Z5test4M10Incompletei(%arg0: !cir.data_member<!s32i in !rec_Incomplete> loc({{.+}})) -> !cir.data_member<!s32i in !rec_Incomplete>

int test5(Incomplete *ic, int Incomplete::*member) {
  return ic->*member;
}
// CHECK: cir.func dso_local @_Z5test5P10IncompleteMS_i
// CHECK: %{{.+}} = cir.get_runtime_member %{{.+}}[%{{.+}} : !cir.data_member<!s32i in !rec_Incomplete>] : !cir.ptr<!rec_Incomplete> -> !cir.ptr<!s32i>
// CHECK: }

auto test_null() -> int Point::* {
  return nullptr;
}
// CHECK: cir.func dso_local @_Z9test_nullv
// CHECK:   %{{.+}} = cir.const #cir.data_member<null> : !cir.data_member<!s32i in !rec_Point>
// CHECK: }

auto test_null_incomplete() -> int Incomplete::* {
  return nullptr;
}
// CHECK: cir.func dso_local @_Z20test_null_incompletev
// CHECK:   %{{.+}} = cir.const #cir.data_member<null> : !cir.data_member<!s32i in !rec_Incomplete>
// CHECK: }
