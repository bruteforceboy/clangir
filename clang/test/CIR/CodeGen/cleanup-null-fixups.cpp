// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// CIR-LABEL: cir.func {{.*}} @{{.*}}ZNK1d1eEv
// CIR:  cir.scope {
// CIR:    %[[V5:.*]] = cir.alloca !rec_aj, !cir.ptr<!rec_aj>
// CIR:    %[[V6:.*]] = cir.get_global @an : !cir.ptr<!rec_aj>
// CIR:    cir.copy %[[V6]] to %[[V5]] : !cir.ptr<!rec_aj>
// CIR:    %[[V7:.*]] = cir.load align(1) %[[V5]] : !cir.ptr<!rec_aj>, !rec_aj
// CIR:    cir.call @_ZN1a1cC1I2ajEET_({{.*}}, %[[V7]]) : (!cir.ptr<!rec_a3A3Ac>, !rec_aj) -> ()
// CIR:    cir.call @_ZN2ajD1Ev(%[[V5]]) : (!cir.ptr<!rec_aj>) -> ()
// CIR:  }
// CIR:  cir.call @_ZN2ajD1Ev({{.*}}) : (!cir.ptr<!rec_aj>) -> ()
// CIR:  %[[V4:.*]] = cir.load {{.*}} : !cir.ptr<!rec_a3A3Ac>, !rec_a3A3Ac
// CIR:  cir.return %[[V4]] : !rec_a3A3Ac
// CIR: ^bb1:  // no predecessors
// CIR:  cir.scope {
// CIR:    %[[V5:.*]] = cir.alloca !rec_a3A3Ac, !cir.ptr<!rec_a3A3Ac>
// CIR:    %[[V6:.*]] = cir.const #cir.int<0> : !s32i
// CIR:    cir.call @_ZN1a1cC1IiEET_(%[[V5]], %[[V6]]) : (!cir.ptr<!rec_a3A3Ac>, !s32i) -> ()
// CIR:    cir.call @_ZN1a1cD1Ev(%[[V5]]) : (!cir.ptr<!rec_a3A3Ac>) -> ()
// CIR:  }
// CIR:  cir.trap

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
