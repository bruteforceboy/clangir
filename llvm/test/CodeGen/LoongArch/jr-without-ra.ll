; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc --mtriple=loongarch64 -mattr=+d < %s | FileCheck %s

;; Check the `jr` instruction does not use `ra` register.
;; Ensure that this function has only one `ret` instruction.
;; ret = jr $ra

define void @jr_without_ra(ptr %rtwdev, ptr %chan, ptr %h2c, i8 %.pre, i1 %cmp.i, ptr %tssi_trim.i, i64 %indvars.iv, ptr %arrayidx14.i, i8 %0, ptr %curr_tssi_trim_de, ptr %arrayidx, ptr %switch.gep, ptr %tssi_cck, i64 %switch.load, ptr %curr_tssi_cck_de, ptr %arrayidx14, ptr %curr_tssi_cck_de_20m, ptr %tssi_trim_6g.i, i64 %indvars.iv14, ptr %tssi_mcs.i) nounwind {
; CHECK-LABEL: jr_without_ra:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addi.d $sp, $sp, -96
; CHECK-NEXT:    st.d $ra, $sp, 88 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $fp, $sp, 80 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s0, $sp, 72 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s1, $sp, 64 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s2, $sp, 56 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s3, $sp, 48 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s4, $sp, 40 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s5, $sp, 32 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s6, $sp, 24 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s7, $sp, 16 # 8-byte Folded Spill
; CHECK-NEXT:    st.d $s8, $sp, 8 # 8-byte Folded Spill
; CHECK-NEXT:    move $s7, $zero
; CHECK-NEXT:    move $s0, $zero
; CHECK-NEXT:    ld.d $t0, $sp, 184
; CHECK-NEXT:    ld.d $s2, $sp, 176
; CHECK-NEXT:    ld.d $s1, $sp, 168
; CHECK-NEXT:    ld.d $t1, $sp, 160
; CHECK-NEXT:    ld.d $t2, $sp, 152
; CHECK-NEXT:    ld.d $t3, $sp, 144
; CHECK-NEXT:    ld.d $t4, $sp, 136
; CHECK-NEXT:    ld.d $t5, $sp, 128
; CHECK-NEXT:    ld.d $t6, $sp, 120
; CHECK-NEXT:    ld.d $t7, $sp, 112
; CHECK-NEXT:    ld.d $t8, $sp, 104
; CHECK-NEXT:    ld.d $fp, $sp, 96
; CHECK-NEXT:    andi $a4, $a4, 1
; CHECK-NEXT:    alsl.d $a6, $a6, $s1, 4
; CHECK-NEXT:    pcalau12i $s1, %pc_hi20(.LJTI0_0)
; CHECK-NEXT:    addi.d $s1, $s1, %pc_lo12(.LJTI0_0)
; CHECK-NEXT:    slli.d $s3, $s2, 2
; CHECK-NEXT:    alsl.d $s2, $s2, $s3, 1
; CHECK-NEXT:    add.d $s2, $t5, $s2
; CHECK-NEXT:    addi.w $s4, $zero, -41
; CHECK-NEXT:    ori $s3, $zero, 1
; CHECK-NEXT:    slli.d $s4, $s4, 3
; CHECK-NEXT:    ori $s6, $zero, 3
; CHECK-NEXT:    lu32i.d $s6, 262144
; CHECK-NEXT:    b .LBB0_4
; CHECK-NEXT:    .p2align 4, , 16
; CHECK-NEXT:  .LBB0_1: # %sw.bb27.i.i
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    ori $s8, $zero, 1
; CHECK-NEXT:  .LBB0_2: # %if.else.i106
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    alsl.d $s5, $s0, $s0, 3
; CHECK-NEXT:    alsl.d $s0, $s5, $s0, 1
; CHECK-NEXT:    add.d $s0, $t0, $s0
; CHECK-NEXT:    ldx.bu $s8, $s0, $s8
; CHECK-NEXT:  .LBB0_3: # %phy_tssi_get_ofdm_de.exit
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    st.b $zero, $t5, 0
; CHECK-NEXT:    st.b $s7, $t3, 0
; CHECK-NEXT:    st.b $zero, $t8, 0
; CHECK-NEXT:    st.b $zero, $t1, 0
; CHECK-NEXT:    st.b $zero, $a1, 0
; CHECK-NEXT:    st.b $zero, $t2, 0
; CHECK-NEXT:    st.b $s8, $a5, 0
; CHECK-NEXT:    ori $s0, $zero, 1
; CHECK-NEXT:    move $s7, $a3
; CHECK-NEXT:  .LBB0_4: # %for.body
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    beqz $a4, .LBB0_9
; CHECK-NEXT:  # %bb.5: # %calc_6g.i
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    move $s7, $zero
; CHECK-NEXT:    bnez $zero, .LBB0_8
; CHECK-NEXT:  # %bb.6: # %calc_6g.i
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    slli.d $s8, $zero, 3
; CHECK-NEXT:    ldx.d $s8, $s1, $s8
; CHECK-NEXT:    jr $s8
; CHECK-NEXT:  .LBB0_7: # %sw.bb12.i.i
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    ori $s7, $zero, 1
; CHECK-NEXT:  .LBB0_8: # %if.else58.i
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    ldx.bu $s7, $a6, $s7
; CHECK-NEXT:    b .LBB0_11
; CHECK-NEXT:    .p2align 4, , 16
; CHECK-NEXT:  .LBB0_9: # %if.end.i
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    andi $s7, $s7, 255
; CHECK-NEXT:    ori $s5, $zero, 50
; CHECK-NEXT:    bltu $s5, $s7, .LBB0_15
; CHECK-NEXT:  # %bb.10: # %if.end.i
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    sll.d $s7, $s3, $s7
; CHECK-NEXT:    and $s8, $s7, $s6
; CHECK-NEXT:    move $s7, $fp
; CHECK-NEXT:    beqz $s8, .LBB0_15
; CHECK-NEXT:  .LBB0_11: # %phy_tssi_get_ofdm_trim_de.exit
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    move $s8, $zero
; CHECK-NEXT:    st.b $zero, $t7, 0
; CHECK-NEXT:    ldx.b $ra, $s2, $t4
; CHECK-NEXT:    st.b $zero, $a2, 0
; CHECK-NEXT:    st.b $zero, $a7, 0
; CHECK-NEXT:    st.b $zero, $t6, 0
; CHECK-NEXT:    st.b $ra, $a0, 0
; CHECK-NEXT:    bnez $s3, .LBB0_13
; CHECK-NEXT:  # %bb.12: # %phy_tssi_get_ofdm_trim_de.exit
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    pcalau12i $ra, %pc_hi20(.LJTI0_1)
; CHECK-NEXT:    addi.d $ra, $ra, %pc_lo12(.LJTI0_1)
; CHECK-NEXT:    ldx.d $s5, $ra, $s4
; CHECK-NEXT:    jr $s5
; CHECK-NEXT:  .LBB0_13: # %phy_tssi_get_ofdm_trim_de.exit
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    bnez $s3, .LBB0_1
; CHECK-NEXT:  # %bb.14: # %phy_tssi_get_ofdm_trim_de.exit
; CHECK-NEXT:    # in Loop: Header=BB0_4 Depth=1
; CHECK-NEXT:    bnez $zero, .LBB0_3
; CHECK-NEXT:    b .LBB0_2
; CHECK-NEXT:  .LBB0_15: # %sw.bb9.i.i
; CHECK-NEXT:    ld.d $s8, $sp, 8 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s7, $sp, 16 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s6, $sp, 24 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s5, $sp, 32 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s4, $sp, 40 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s3, $sp, 48 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s2, $sp, 56 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s1, $sp, 64 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $s0, $sp, 72 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $fp, $sp, 80 # 8-byte Folded Reload
; CHECK-NEXT:    ld.d $ra, $sp, 88 # 8-byte Folded Reload
; CHECK-NEXT:    addi.d $sp, $sp, 96
; CHECK-NEXT:    ret
entry:
  br label %for.body

for.body:
  %1 = phi i8 [ 0, %entry ], [ %.pre, %phy_tssi_get_ofdm_de.exit ]
  %indvars.iv143 = phi i64 [ 0, %entry ], [ 1, %phy_tssi_get_ofdm_de.exit ]
  br i1 %cmp.i, label %calc_6g.i, label %if.end.i

if.end.i:
  switch i8 %1, label %sw.bb9.i.i [
    i8 1, label %phy_tssi_get_ofdm_trim_de.exit
    i8 50, label %phy_tssi_get_ofdm_trim_de.exit
    i8 0, label %phy_tssi_get_ofdm_trim_de.exit
  ]

sw.bb9.i.i:
  ret void

calc_6g.i:
  switch i8 1, label %if.else58.i [
    i8 55, label %sw.bb5.i125.i
    i8 54, label %sw.bb5.i125.i
    i8 53, label %sw.bb5.i125.i
    i8 52, label %sw.bb5.i125.i
    i8 51, label %sw.bb5.i125.i
    i8 50, label %sw.bb5.i125.i
    i8 49, label %sw.bb5.i125.i
    i8 56, label %sw.bb5.i125.i
    i8 57, label %sw.bb5.i125.i
    i8 58, label %sw.bb5.i125.i
    i8 59, label %sw.bb5.i125.i
    i8 60, label %sw.bb5.i125.i
    i8 61, label %sw.bb5.i125.i
    i8 -115, label %sw.bb12.i.i
    i8 -116, label %sw.bb12.i.i
    i8 -117, label %sw.bb12.i.i
    i8 -118, label %sw.bb12.i.i
    i8 -119, label %sw.bb12.i.i
    i8 -120, label %sw.bb12.i.i
    i8 -121, label %sw.bb12.i.i
    i8 -122, label %sw.bb12.i.i
    i8 -123, label %sw.bb12.i.i
    i8 -124, label %sw.bb12.i.i
    i8 -125, label %sw.bb12.i.i
    i8 -126, label %sw.bb12.i.i
    i8 -127, label %sw.bb12.i.i
    i8 77, label %sw.bb6.i124.i
    i8 76, label %sw.bb6.i124.i
    i8 75, label %sw.bb6.i124.i
    i8 74, label %sw.bb6.i124.i
    i8 73, label %sw.bb6.i124.i
    i8 72, label %sw.bb6.i124.i
    i8 71, label %sw.bb6.i124.i
    i8 1, label %sw.bb6.i124.i
    i8 69, label %sw.bb6.i124.i
    i8 68, label %sw.bb6.i124.i
    i8 67, label %sw.bb6.i124.i
    i8 66, label %sw.bb6.i124.i
    i8 65, label %sw.bb6.i124.i
  ]

sw.bb5.i125.i:
  br label %if.else58.i

sw.bb6.i124.i:
  br label %if.else58.i

sw.bb12.i.i:
  br label %if.else58.i

if.else58.i:
  %retval.0.i120.ph.i = phi i64 [ 0, %calc_6g.i ], [ 1, %sw.bb5.i125.i ], [ 1, %sw.bb6.i124.i ], [ 1, %sw.bb12.i.i ]
  %arrayidx63.i = getelementptr [4 x [16 x i8]], ptr %tssi_trim_6g.i, i64 0, i64 %indvars.iv, i64 %retval.0.i120.ph.i
  %2 = load i8, ptr %arrayidx63.i, align 1
  br label %phy_tssi_get_ofdm_trim_de.exit

phy_tssi_get_ofdm_trim_de.exit:
  %retval.0.i = phi i8 [ %2, %if.else58.i ], [ %0, %if.end.i ], [ %0, %if.end.i ], [ %0, %if.end.i ]
  store i8 0, ptr %arrayidx, align 1
  %arrayidx8 = getelementptr [4 x [6 x i8]], ptr %tssi_cck, i64 0, i64 %indvars.iv14, i64 %switch.load
  %3 = load i8, ptr %arrayidx8, align 1
  store i8 0, ptr %h2c, align 1
  store i8 0, ptr %arrayidx14.i, align 1
  store i8 0, ptr %switch.gep, align 1
  store i8 %3, ptr %rtwdev, align 1
  switch i8 0, label %if.else.i106 [
    i8 -87, label %sw.bb27.i.i
    i8 0, label %sw.bb27.i.i
    i8 -89, label %sw.bb27.i.i
    i8 -90, label %sw.bb27.i.i
    i8 -91, label %sw.bb27.i.i
    i8 -92, label %phy_tssi_get_ofdm_de.exit
    i8 -93, label %phy_tssi_get_ofdm_de.exit
    i8 1, label %phy_tssi_get_ofdm_de.exit
    i8 -95, label %sw.bb25.i.i
    i8 -96, label %sw.bb25.i.i
    i8 -97, label %sw.bb25.i.i
    i8 -98, label %sw.bb25.i.i
    i8 -99, label %sw.bb25.i.i
    i8 43, label %phy_tssi_get_ofdm_de.exit
    i8 42, label %phy_tssi_get_ofdm_de.exit
    i8 41, label %phy_tssi_get_ofdm_de.exit
  ]

sw.bb25.i.i:
  br label %if.else.i106

sw.bb27.i.i:
  br label %if.else.i106

if.else.i106:
  %retval.0.i.ph.i107 = phi i64 [ 0, %phy_tssi_get_ofdm_trim_de.exit ], [ 1, %sw.bb25.i.i ], [ 1, %sw.bb27.i.i ]
  %arrayidx26.i109 = getelementptr [4 x [19 x i8]], ptr %tssi_mcs.i, i64 0, i64 %indvars.iv143, i64 %retval.0.i.ph.i107
  %4 = load i8, ptr %arrayidx26.i109, align 1
  br label %phy_tssi_get_ofdm_de.exit

phy_tssi_get_ofdm_de.exit:
  %retval.0.i110 = phi i8 [ %4, %if.else.i106 ], [ 0, %phy_tssi_get_ofdm_trim_de.exit ], [ 0, %phy_tssi_get_ofdm_trim_de.exit ], [ 0, %phy_tssi_get_ofdm_trim_de.exit ], [ 0, %phy_tssi_get_ofdm_trim_de.exit ], [ 0, %phy_tssi_get_ofdm_trim_de.exit ], [ 0, %phy_tssi_get_ofdm_trim_de.exit ]
  store i8 0, ptr %tssi_cck, align 1
  store i8 %retval.0.i, ptr %curr_tssi_cck_de, align 1
  store i8 0, ptr %curr_tssi_trim_de, align 1
  store i8 0, ptr %curr_tssi_cck_de_20m, align 1
  store i8 0, ptr %chan, align 1
  store i8 0, ptr %arrayidx14, align 1
  store i8 %retval.0.i110, ptr %tssi_trim.i, align 1
  br label %for.body
}
