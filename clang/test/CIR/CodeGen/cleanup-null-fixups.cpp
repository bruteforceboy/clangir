// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// CIR-LABEL: cir.func {{.*}} @_ZNK1d1eEv
// CIR:  cir.scope {
// CIR-NEXT:    %[[V5:.*]] = cir.alloca !rec_aj, !cir.ptr<!rec_aj>
// CIR-NEXT:    %[[V6:.*]] = cir.get_global @an : !cir.ptr<!rec_aj>
// CIR-NEXT:    cir.copy %[[V6]] to %[[V5]] : !cir.ptr<!rec_aj>
// CIR-NEXT:    %[[V7:.*]] = cir.load align(1) %[[V5]] : !cir.ptr<!rec_aj>, !rec_aj
// CIR-NEXT:    cir.call @_ZN1a1cC1I2ajEET_({{.*}}, %[[V7]]) : (!cir.ptr<!rec_a3A3Ac>, !rec_aj) -> ()
// CIR-NEXT:    cir.call @_ZN2ajD1Ev(%[[V5]]) : (!cir.ptr<!rec_aj>) -> ()
// CIR-NEXT:  }
// CIR-NEXT:  cir.call @_ZN2ajD1Ev({{.*}}) : (!cir.ptr<!rec_aj>) -> ()
// CIR-NEXT:  %[[V4:.*]] = cir.load {{.*}} : !cir.ptr<!rec_a3A3Ac>, !rec_a3A3Ac
// CIR-NEXT:  cir.return %[[V4]] : !rec_a3A3Ac
// CIR-NEXT: ^bb1:  // no predecessors
// CIR-NEXT:  cir.scope {
// CIR-NEXT:    %[[V5:.*]] = cir.alloca !rec_a3A3Ac, !cir.ptr<!rec_a3A3Ac>
// CIR-NEXT:    %[[V6:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:    cir.call @_ZN1a1cC1IiEET_(%[[V5]], %[[V6]]) : (!cir.ptr<!rec_a3A3Ac>, !s32i) -> ()
// CIR-NEXT:    cir.call @_ZN1a1cD1Ev(%[[V5]]) : (!cir.ptr<!rec_a3A3Ac>) -> ()
// CIR-NEXT:  }
// CIR-NEXT:  cir.trap

// LLVM-LABEL: {{.*}} @_ZNK1d1eEv(ptr {{.*}})
// LLVM-NEXT:   %[[V2:.*]] = alloca %class.aj, i64 1, align 1
// LLVM-NEXT:   %[[V3:.*]] = alloca %"class.a::c", i64 1, align 1
// LLVM-NEXT:   %[[V4:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   %[[V5:.*]] = alloca %"class.a::c", i64 1, align 1
// LLVM-NEXT:   %[[V6:.*]] = alloca %class.aj, i64 1, align 1
// LLVM-NEXT:   store ptr {{.*}}, ptr %[[V4]], align 8
// LLVM-NEXT:   %[[V7:.*]] = load ptr, ptr %[[V4]], align 8
// LLVM-NEXT:   br label %[[B8:.*]]
// LLVM: [[B8]]:
// LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i32(ptr %[[V2]], ptr @an, i32 1, i1 false)
// LLVM-NEXT:   %[[V9:.*]] = load %class.aj, ptr %[[V2]], align 1
// LLVM-NEXT:   call void @_ZN1a1cC1I2ajEET_(ptr %[[V5]], %class.aj %[[V9]])
// LLVM-NEXT:   call void @_ZN2ajD1Ev(ptr %[[V2]])
// LLVM-NEXT:   br label %[[B10:.*]]
// LLVM: [[B10]]:
// LLVM-NEXT:   call void @_ZN2ajD1Ev(ptr %[[V6]])
// LLVM-NEXT:   %[[V11:.*]] = load %"class.a::c", ptr %[[V5]], align 1
// LLVM-NEXT:   ret %"class.a::c" %[[V11]]
// LLVM: [[B12:.*]]:                                               ; No predecessors!
// LLVM-NEXT:   br label %[[B13:.*]]
// LLVM: [[B13]]:
// LLVM-NEXT:   call void @_ZN1a1cC1IiEET_(ptr %[[V3]], i32 0)
// LLVM-NEXT:   call void @_ZN1a1cD1Ev(ptr %[[V3]])
// LLVM-NEXT:   br label %[[B14:.*]]
// LLVM: [[B14]]:
// LLVM-NEXT:   call void @llvm.trap()

// OGCG-LABEL: {{.*}} @_ZNK1d1eEv(ptr {{.*}})
// OGCG:  %{{.*}} = alloca ptr, align 8
// OGCG-NEXT:  %{{.*}} = alloca ptr, align 8
// OGCG-NEXT:  %{{.*}} = alloca %{{.*}}, align 1
// OGCG-NEXT:  %{{.*}} = alloca %{{.*}}, align 1
// OGCG-NEXT:  store ptr %{{.*}}, ptr %{{.*}}, align 8
// OGCG-NEXT:  store ptr %{{.*}}, ptr %{{.*}}, align 8
// OGCG-NEXT:  %{{.*}} = load ptr, ptr %{{.*}}, align 8
// OGCG-NEXT:  call void @_ZN1a1cC1I2ajEET_(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}}, ptr noundef %{{.*}})
// OGCG-NEXT:  call void @_ZN2ajD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}})
// OGCG-NEXT:  call void @_ZN2ajD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}})
// OGCG-NEXT:  ret void
// OGCG-NEXT: }

inline namespace a {
class c {
public:
  template <typename b> c(b);
  ~c();
};
} // namespace a
class d {
  c e() const;
};
class aj {
public:
  ~aj();
} an;
c d::e() const {
  aj ao;
  return an;
  c(0);
}
