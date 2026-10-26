; Общие подпрограммы HippoGPT (сверены с ref/intref.py).
; Ожидают от включающего файла: адреса буферов, SQLO, EXP63, RSQ8, переменные, LD_* и CUR.

; ---------------- таблица квадратов ----------------
; SQ[n] = n*n для знакового байта n: младшие байты в SQLO, старшие в SQLO+256
init_sq:
        ld ix,0
        ld de,1
        ld c,0
.loop:  ld h,SQLO >> 8
        ld l,c
        ld a,ixl
        ld (hl),a
        inc h
        ld a,ixh
        ld (hl),a
        xor a
        sub c
        ld l,a
        ld a,ixh
        ld (hl),a
        dec h
        ld a,ixl
        ld (hl),a
        add ix,de
        inc de
        inc de
        inc c
        ld a,c
        cp 129
        jr nz,.loop
        ret

; ---------------- умножения ----------------
; mul8u: A * E (без знака) -> HL
mul8u:
        ld hl,0
        ld d,h
        ld b,8
.l:     add hl,hl
        add a,a
        jr nc,.n
        add hl,de
.n:     djnz .l
        ret

; mul16x8s: DE (со знаком) * C (без знака) -> B:HL (24 бита со знаком)
mul16x8s:
        ld a,d
        or a
        push af                 ; S = знак множимого
        jp p,.pos
        xor a
        sub e
        ld e,a
        sbc a,a
        sub d
        ld d,a
.pos:   ld hl,0
        ld b,h
        ld a,c
        dup 8
        add hl,hl
        rl b
        add a,a
        jr nc,1F
        add hl,de
        jr nc,1F
        inc b
1:
        edup
        pop af
        ret p
neg24:  xor a                   ; B:HL = -B:HL
        sub l
        ld l,a
        ld a,0
        sbc a,h
        ld h,a
        ld a,0
        sbc a,b
        ld b,a
        ret

; rshift24: B:HL = (B:HL + 2^(C-1)) >> C, сдвиг арифметический, 1 <= C <= 16
rshift24:
        push de
        ld a,c
        dec a
        add a,a
        add a,low POW2
        ld e,a
        adc a,high POW2
        sub e
        ld d,a
        ld a,(de)
        add a,l
        ld l,a
        inc de
        ld a,(de)
        adc a,h
        ld h,a
        jr nc,.nc
        inc b
.nc:    pop de
        ld a,c
        cp 8
        jr c,.bits
        ld l,h                  ; >> 8 одним махом
        ld h,b
        ld a,b
        rla
        sbc a,a
        ld b,a
        ld a,c
        sub 8
        ret z
.bits:  sra b
        rr h
        rr l
        dec a
        jr nz,.bits
        ret

POW2:   dw 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768

; clamp24: B:HL -> HL в [RQ_LO, RQ_HI] (слова со знаком)
clamp24:
        ld a,h
        rla
        ld a,b
        adc a,0
        jr z,.fits              ; B — расширение знака H
        bit 7,b
        jr nz,.lo
.hi:    ld hl,(RQ_HI)
        ret
.lo:    ld hl,(RQ_LO)
        ret
.fits:  ld de,(RQ_HI)           ; HL > HI ?
        call cmp_signed         ; CF: HL < DE
        jr c,.chklo
        jr z,.ok
        jr .hi
.chklo: ld de,(RQ_LO)
        call cmp_signed
        jr c,.lo
.ok:    ret

; cmp_signed: флаги для HL ? DE со знаком: CF = HL < DE, Z = равны; HL, DE не меняются
cmp_signed:
        ld a,h
        xor 0x80
        ld h,a
        ld a,d
        xor 0x80
        ld d,a
        or a
        sbc hl,de
        push af
        add hl,de
        ld a,h
        xor 0x80
        ld h,a
        ld a,d
        xor 0x80
        ld d,a
        pop af
        ret

; requant: B = число строк (0 = 256), HL -> аккумуляторы (слова), DE -> множители (байты)
;   каждое: clamp((acc*M + 2^(S-1)) >> S); S = RQ_S
;   rq_out: обработчик результата (HL), вызывается с IX = номер (0..)
requant:
        ld ix,0
.loop:  call brk_poll               ; 256 строк — ~0,1 с, BREAK не должен ждать
        push bc
        push hl
        push de
        ld a,(de)
        ld c,a
        ld e,(hl)
        inc hl
        ld d,(hl)
        call mul16x8s
        ld a,(RQ_S)
        ld c,a
        call rshift24
        call clamp24
.out:   call 0                  ; адрес обработчика подставляется
        inc ix
        pop de
        inc de
        pop hl
        inc hl
        inc hl
        pop bc
        djnz .loop
        ret

set_rq: ; HL = lo, DE = hi, A = S, BC = обработчик
        ld (RQ_LO),hl
        ld (RQ_HI),de
        ld (RQ_S),a
        ld (requant.out+1),bc
        ret

; ---------------- RMSNorm ----------------
; norm_prep: HVEC -> HB (h' = h >> s), NS_K, NS_SH (для сдвига вывода out_shift в C)
norm_prep:
        push bc
        ld hl,HVEC
        ld bc,0
        ld a,D_MODEL
.m:     ex af,af'
        ld e,(hl)
        inc hl
        ld d,(hl)
        inc hl
        bit 7,d
        jr z,.p
        xor a
        sub e
        ld e,a
        sbc a,a
        sub d
        ld d,a
.p:     ld a,c
        or e
        ld c,a
        ld a,b
        or d
        ld b,a
        ex af,af'
        dec a
        jr nz,.m
        ld e,0                  ; s: сдвигать, пока |h|max >= 128
.s:     ld a,b
        or a
        jr nz,.sh
        ld a,c
        cp 128
        jr c,.sd
.sh:    srl b
        rr c
        inc e
        jr .s
.sd:    ld a,e
        ld (NS_S),a
        ld ix,HVEC              ; h' в HB
        ld hl,HB
        ld b,D_MODEL
.h1:    ld e,(ix+0)
        ld d,(ix+1)
        inc ix
        inc ix
        ld a,(NS_S)
        or a
        jr z,.h1s
        ld c,a
.h1l:   sra d
        rr e
        dec c
        jr nz,.h1l
.h1s:   ld (hl),e
        inc hl
        djnz .h1
        ld de,HB                ; ssq = sum SQ[h'] -> IXH:B:C
        ld bc,0
        ld ixh,b
        ld ixl,D_MODEL
.h2:    ld a,(de)
        inc de
        ld l,a
        ld h,SQLO >> 8
        ld a,(hl)
        add a,c
        ld c,a
        inc h
        ld a,(hl)
        adc a,b
        ld b,a
        jr nc,.h2c
        inc ixh
.h2c:   dec ixl
        jr nz,.h2
        ld e,0                  ; e: делить на 4, пока ssq >= 256
.e:     ld a,ixh
        or b
        jr z,.ed
        ld a,ixh
        srl a
        rr b
        rr c
        srl a
        rr b
        rr c
        ld ixh,a
        inc e
        jr .e
.ed:    ld a,c
        cp 64
        jr nc,.m64
        ld a,64
.m64:   ld l,a
        ld h,RSQ8 >> 8
        ld a,(hl)
        ld (NS_K),a
        pop bc                  ; sh = 10 + e/2 - out_shift
        ld a,10
        add a,e
        sub c
        ld (NS_SH),a
        ret

; norm_one: A = h' (знаковый байт) -> HL = (h'*K + 2^(sh-1)) >> sh, без ограничения
norm_one:
        or a
        push af
        jp p,.p
        neg
.p:     ld hl,NS_K
        ld e,(hl)
        call mul8u
        pop af
        jp p,.q
        xor a
        sub l
        ld l,a
        sbc a,a
        sub h
        ld h,a
.q:     ld a,(NS_SH)
        ld c,a
        dec a
        add a,a
        add a,low POW2
        ld e,a
        adc a,high POW2
        sub e
        ld d,a
        ld a,(de)
        ld b,a
        inc de
        ld a,(de)
        ld d,a
        ld e,b
        add hl,de
.sh:    sra h
        rr l
        dec c
        jr nz,.sh
        ret

; norm_words: HVEC -> X16 (слова), шаг 1/32, ограничение +-127
norm_words:
        ld c,8
        call norm_prep
        ld ix,HB
        ld iy,X16
        ld b,D_MODEL
.l:     push bc
        ld a,(ix+0)
        inc ix
        call norm_one
        ld de,127
        call clamp_hl
        ld (iy+0),l
        ld (iy+1),h
        inc iy
        inc iy
        pop bc
        djnz .l
        ret

; norm_bytes: HVEC -> XF (байты), шаг 1/16, ограничение +-63
norm_bytes:
        ld c,FINAL_SHIFT
        call norm_prep
        ld ix,HB
        ld iy,XF
        ld b,D_MODEL
.l:     push bc
        ld a,(ix+0)
        inc ix
        call norm_one
        ld de,FINAL_LIM
        call clamp_hl
        ld (iy+0),l
        inc iy
        pop bc
        djnz .l
        ret

; clamp_hl: HL в [-DE, DE] (DE > 0)
clamp_hl:
        bit 7,h
        jr nz,.neg
        or a
        sbc hl,de
        add hl,de
        ret c
        ex de,hl
        ret
.neg:   push hl
        add hl,de               ; HL + DE >= 0 ?
        pop hl
        ret c
        xor a
        sub e
        ld l,a
        sbc a,a
        sub d
        ld h,a
        ret


rq_store8:                      ; QKV8[IX] = L
        push ix
        ld de,QKV8
        add ix,de
        ld (ix+0),l
        pop ix
        ret

rq_addh:                        ; HVEC[IX] += HL с насыщением
        push ix
        add ix,ix
        ld de,HVEC
        add ix,de
        ld e,(ix+0)
        ld d,(ix+1)
        or a
        adc hl,de
        jp po,.ok
        ld hl,32767             ; переполнение: знак слагаемого
        bit 7,d
        jr z,.ok
        inc hl                  ; -32768
.ok:    ld (ix+0),l
        ld (ix+1),h
        pop ix
        ret

; kv_store: k -> KC[slot], AK[slot]; v -> VT[d][slot], V2[d] += v^2 - старое^2
kv_store:
        ld a,(SLOT)             ; KC + slot*16
        ld l,a
        ld h,0
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        ld de,(CUR + LD_KC)
        add hl,de
        ex de,hl
        ld hl,QKV8 + QDIM
        ld bc,D_HEAD
        ldir
        ld ix,QKV8 + QDIM       ; AK[slot] = sum k^2
        ld bc,0
        ld a,D_HEAD
.ak:    ex af,af'
        ld l,(ix+0)
        inc ix
        ld h,SQLO >> 8
        ld a,(hl)
        add a,c
        ld c,a
        inc h
        ld a,(hl)
        adc a,b
        ld b,a
        ex af,af'
        dec a
        jr nz,.ak
        ld a,(SLOT)
        ld l,a
        ld h,0
        add hl,hl
        ld de,(CUR + LD_AK)
        add hl,de
        ld (hl),c
        inc hl
        ld (hl),b
        ld ix,QKV8 + QDIM + D_HEAD      ; v
        ld iy,(CUR + LD_V2)
        ld de,(CUR + LD_VT)
        ld a,(SLOT)
        add a,e
        ld e,a                  ; строки VT по CTX байт, выровнены: не переходит страницу
        ld b,D_HEAD
.v:     push bc
        ld a,(de)               ; старое
        ld l,a
        ld h,SQLO >> 8
        ld c,(hl)
        inc h
        ld b,(hl)               ; BC = старое^2
        ld a,(ix+0)
        ld (de),a
        inc ix
        ld l,a
        dec h
        ld a,(hl)
        inc h
        ld h,(hl)
        ld l,a                  ; HL = новое^2
        or a
        sbc hl,bc               ; разность со знаком, 16 бит
        ld a,h                  ; V2 (24 бита) += расширенная разность
        rla
        sbc a,a
        ld c,a
        ld a,(iy+0)
        add a,l
        ld (iy+0),a
        ld a,(iy+1)
        adc a,h
        ld (iy+1),a
        ld a,(iy+2)
        adc a,c
        ld (iy+2),a
        inc iy
        inc iy
        inc iy
        ld hl,CTX
        add hl,de
        ex de,hl
        pop bc
        djnz .v
    IF PIN
        ld hl,(POS)             ; позиции 0..PIN-1 — ещё и в закреплённые
        ld a,h
        or a
        ret nz
        ld a,l
        cp PIN
        ret nc
        ld l,a                  ; KCP[pos] = k
        ld h,0
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        ld de,(CUR + LD_KCP)
        add hl,de
        ex de,hl
        ld hl,QKV8 + QDIM
        ld bc,D_HEAD
        ldir
        ld a,(SLOT)             ; AKP[pos] = AK[slot]
        ld l,a
        ld h,0
        add hl,hl
        ld de,(CUR + LD_AK)
        add hl,de
        ld c,(hl)
        inc hl
        ld b,(hl)
        ld hl,(POS)
        add hl,hl
        ld de,(CUR + LD_AKP)
        add hl,de
        ld (hl),c
        inc hl
        ld (hl),b
        ld hl,(CUR + LD_VTP)    ; VTP[d][pos] = v_d, V2P[d] += v_d^2
        ld de,(POS)
        add hl,de
        ex de,hl
        ld ix,QKV8 + QDIM + D_HEAD
        ld iy,(CUR + LD_V2P)
        ld b,D_HEAD
.vp:    ld a,(ix+0)
        inc ix
        ld (de),a
        ld l,a
        ld h,SQLO >> 8
        ld a,(hl)
        inc h
        ld h,(hl)
        ld l,a                  ; HL = v^2
        ld a,(iy+0)
        add a,l
        ld (iy+0),a
        ld a,(iy+1)
        adc a,h
        ld (iy+1),a
        jr nc,.vc
        inc (iy+2)
.vc:    inc iy
        inc iy
        inc iy
        ld hl,PIN
        add hl,de
        ex de,hl
        djnz .vp
    ENDIF
        ret

; attn_head: голова HEAD -> X16[HEAD*D_HEAD ..] (слова)
attn_head:
        ld a,(HEAD)             ; q головы -> непосредственные операнды dot16, AQ = sum q^2
        add a,a
        add a,a
        add a,a
        add a,a
        ld e,a
        ld d,0
        ld ix,QKV8
        add ix,de
        ld iy,dot16_imm
        ld bc,0
        ld a,D_HEAD
.q:     ex af,af'
        ld a,(ix+0)
        inc ix
        ld (iy+0),a
        ld de,DOT_STRIDE
        add iy,de
        ld l,a
        ld h,SQLO >> 8
        ld a,(hl)
        add a,c
        ld c,a
        inc h
        ld a,(hl)
        adc a,b
        ld b,a
        ex af,af'
        dec a
        jr nz,.q
        ld (AQ),bc
        ; счёт по слотам
        ld hl,-32768
        ld (SMAX),hl
        ld de,(CUR + LD_KC)
        ld (.kp+1),de
        ld de,(CUR + LD_AK)
        ld (.akp+1),de
        xor a
.t:     ld (.tt+1),a
        call slot_valid
        jr nc,.skip
.kp:    ld de,0                 ; строка K
        call dot16              ; IXL:B:C... -> A:BC? см. gen_dot: результат в IXL:B:C
        ; r = AQ + AK[t] - D  (24 бита)
.akp:   ld hl,0
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld hl,(AQ)
        add hl,de
        ld a,0
        adc a,0                 ; A:HL = AQ + AK
        or a
        sbc hl,bc
        sbc a,ixl               ; A:HL -= IXL:B:C
        ; s = r >> 9 = (A:H) >> 1
        sra a
        rr h
        ld l,h
        ld h,a                  ; HL = r >> 9 (значение помещается в 16 бит)
        ; + bias[HEAD][(POS - t) & (CTX-1)]
        ld a,(HEAD)
        ld e,a
        ld d,0
        dup 6                   ; *CTX (64)
        sla e
        rl d
        edup
        ld a,(POS)
.tt:    sub 0
        and CTX - 1
        add a,e
        ld e,a
        ld a,d
        adc a,0
        ld d,a
        push hl
        ld hl,(CUR + LD_BIAS)
        add hl,de
        ld a,(hl)
        pop hl
        ld e,a
        rla
        sbc a,a
        ld d,a
        add hl,de
        ex de,hl                ; DE = s
        ld a,(.tt+1)
        add a,a
        ld l,a
        ld h,SCORE >> 8
        ld bc,SCORE & 0xFF
        add hl,bc
        ld (hl),e
        inc hl
        ld (hl),d
        ex de,hl
        ld de,(SMAX)            ; максимум
        call cmp_signed
        jr c,.skip
        ld (SMAX),hl
.skip:  ld hl,(.kp+1)
        ld de,D_HEAD
        add hl,de
        ld (.kp+1),hl
        ld hl,(.akp+1)
        inc hl
        inc hl
        ld (.akp+1),hl
        ld a,(.tt+1)
        inc a
        cp CTX
        jp nz,.t
    IF PIN
        ; закреплённые j = 0..min(POS, PIN-1): счёт = (AQ + AKP[j] - D) >> 9 + PINB[голова]
        ld de,(CUR + LD_KCP)
        ld (.kq+1),de
        ld de,(CUR + LD_AKP)
        ld (.akq+1),de
        xor a
.tq:    ld (.ttq+1),a
        call pin_valid
        jr nc,.sq
.kq:    ld de,0
        call dot16
.akq:   ld hl,0
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld hl,(AQ)
        add hl,de
        ld a,0
        adc a,0
        or a
        sbc hl,bc
        sbc a,ixl
        sra a
        rr h
        ld l,h
        ld h,a
        ld a,(HEAD)
        ld e,a
        ld d,0
        push hl
        ld hl,(CUR + LD_PINB)
        add hl,de
        ld a,(hl)
        pop hl
        ld e,a
        rla
        sbc a,a
        ld d,a
        add hl,de
        ex de,hl                ; DE = s
        ld a,(.ttq+1)
        add a,a
        ld l,a
        ld h,0
        ld bc,SCOREP
        add hl,bc
        ld (hl),e
        inc hl
        ld (hl),d
        ex de,hl
        ld de,(SMAX)
        call cmp_signed
        jr c,.sq
        ld (SMAX),hl
.sq:    ld hl,(.kq+1)
        ld de,D_HEAD
        add hl,de
        ld (.kq+1),hl
        ld hl,(.akq+1)
        inc hl
        inc hl
        ld (.akq+1),hl
.ttq:   ld a,0
        inc a
        cp PIN
        jr nz,.tq
    ENDIF
        ; веса e_t, E2, SE; e_t -> операнды dot64
        ld hl,0
        ld (SE),hl
        ld (E2),hl
        xor a
        ld (E2+2),a
        ld iy,dot64_imm
        xor a
.w:     ld (.wt+1),a
        call slot_valid
        ld a,0
        jr nc,.wz
.wt:    ld a,0
        add a,a
        ld l,a
        ld h,SCORE >> 8
        ld bc,SCORE & 0xFF
        add hl,bc
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld hl,(SMAX)
        or a
        sbc hl,de               ; SMAX - s >= 0
        ld a,h
        or a
        ld a,255
        jr nz,.wc
        ld a,l
.wc:    ld l,a
        ld h,EXP63 >> 8
        ld a,(hl)
.wz:    ld (iy+0),a
        ld de,DOT_STRIDE
        add iy,de
        ld e,a                  ; SE += e
        ld hl,(SE)
        ld d,0
        add hl,de
        ld (SE),hl
        ld l,a                  ; E2 += e^2
        ld h,SQLO >> 8
        ld c,(hl)
        inc h
        ld b,(hl)
        ld hl,(E2)
        add hl,bc
        ld (E2),hl
        jr nc,.wn
        ld hl,E2 + 2
        inc (hl)
.wn:    ld a,(.wt+1)
        inc a
        cp CTX
        jr nz,.w
    IF PIN
        ld iy,dot16_imm         ; веса закреплённых — операнды dot16 (q там больше не нужен)
        xor a
.wq:    ld (.wqt+1),a
        call pin_valid
        ld a,0
        jr nc,.wqz
        ld a,(.wqt+1)
        add a,a
        ld l,a
        ld h,0
        ld bc,SCOREP
        add hl,bc
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld hl,(SMAX)
        or a
        sbc hl,de
        ld a,h
        or a
        ld a,255
        jr nz,.wqc
        ld a,l
.wqc:   ld l,a
        ld h,EXP63 >> 8
        ld a,(hl)
.wqz:   ld (iy+0),a
        ld de,DOT_STRIDE
        add iy,de
        ld e,a
        ld hl,(SE)
        ld d,0
        add hl,de
        ld (SE),hl
        ld l,a
        ld h,SQLO >> 8
        ld c,(hl)
        inc h
        ld b,(hl)
        ld hl,(E2)
        add hl,bc
        ld (E2),hl
        jr nc,.wqt
        ld hl,E2 + 2
        inc (hl)
.wqt:   ld a,0
        inc a
        cp PIN
        jr nz,.wq
    ENDIF
        ; выход: o_d = round(8*N/SE), N = (E2 + V2[d] - D)/2
        ld a,(HEAD)             ; IY -> X16[HEAD*16]
        add a,a
        add a,a
        add a,a
        add a,a
        add a,a
        ld e,a
        ld d,0
        ld iy,X16
        add iy,de
        ld hl,(CUR + LD_V2)
        ld (.v2p+1),hl
        ld hl,(CUR + LD_VT)
        ld (.vtp+1),hl
    IF PIN
        ld hl,(CUR + LD_VTP)
        ld (.vtq+1),hl
        ld hl,(CUR + LD_V2P)
        ld (.v2q+1),hl
    ENDIF
        ld b,D_HEAD
.o:     push bc
.vtp:   ld de,0
        call dot64              ; D -> IXL:B:C
.v2p:   ld hl,0
        ld e,(hl)               ; A:HL = V2[d] + E2 - D
        inc hl
        ld d,(hl)
        inc hl
        ld a,(hl)
        ld hl,(E2)
        add hl,de
        ld e,a
        ld a,(E2+2)
        adc a,e
        or a
        sbc hl,bc
        sbc a,ixl
    IF PIN
        push hl                 ; закреплённые: - Dp + V2P[d]
        push af
.vtq:   ld de,0
        call dot16              ; Dp -> IXL:B:C
        pop af
        pop hl
        or a
        sbc hl,bc
        sbc a,ixl
        ex de,hl                ; A:DE
.v2q:   ld hl,0
        ld c,a
        ld a,e
        add a,(hl)
        ld e,a
        inc hl
        ld a,d
        adc a,(hl)
        ld d,a
        inc hl
        ld a,c
        adc a,(hl)
        ex de,hl                ; A:HL
    ENDIF
        sra a                   ; N = /2
        rr h
        rr l
        call attn_div           ; HL = sign(N)*((8|N| + SE/2) / SE)
        ld (iy+0),l
        ld (iy+1),h
        inc iy
        inc iy
        ld hl,(.v2p+1)
        inc hl
        inc hl
        inc hl
        ld (.v2p+1),hl
        ld hl,(.vtp+1)
        ld de,CTX
        add hl,de
        ld (.vtp+1),hl
    IF PIN
        ld hl,(.vtq+1)
        ld de,PIN
        add hl,de
        ld (.vtq+1),hl
        ld hl,(.v2q+1)
        inc hl
        inc hl
        inc hl
        ld (.v2q+1),hl
    ENDIF
        pop bc
        djnz .o
        ret

    IF PIN
; pin_valid: A = номер закреплённого -> CF = 1, если позиция j уже пройдена (j <= POS)
pin_valid:
        ld c,a
        ld hl,(POS)
        ld a,h
        or a
        scf
        ret nz
        ld a,l
        cp c                    ; POS - j: нет заёма — j <= POS
        ccf
        ret
    ENDIF

; slot_valid: A = слот -> CF = 1, если в нём уже есть символ (слот <= POS или POS >= CTX)
slot_valid:
        ld c,a
        ld hl,(POS)
        ld a,h
        or a
        jr nz,.yes
        ld a,l
        cp CTX
        jr nc,.yes
        cp c                    ; POS >= слот
        ccf
        ret c
        ret z
        ret
.yes:   scf
        ret

; attn_div: A:HL = N (24 бита со знаком) -> HL = sign(N) * ((8|N| + SE/2) / SE)
; частное < 1024, потому что |N| <= 63 * SE: десять шагов деления с восстановлением
attn_div:
        ld b,a
        or a
        push af
        jp p,.p
        call neg24
.p:     ld a,b                  ; B:HL *= 8
        add hl,hl
        rla
        add hl,hl
        rla
        add hl,hl
        rla
        ld b,a
        ld de,(SE)              ; + SE/2
        srl d
        rr e
        add hl,de
        jr nc,.nc
        inc b
.nc:    ld de,(SE)              ; C:D:E = SE << 9
        ld c,d
        ld d,e
        ld e,0
        sla d
        rl c
        ld ix,0
        ld a,10
.d:     ex af,af'
        add ix,ix
        or a
        sbc hl,de
        ld a,b
        sbc a,c
        jr c,.r
        ld b,a
        inc ix
        jr .n
.r:     add hl,de
.n:     srl c
        rr d
        rr e
        ex af,af'
        dec a
        jr nz,.d
        push ix
        pop hl
        pop af
        ret p
        xor a
        sub l
        ld l,a
        sbc a,a
        sub h
        ld h,a
        ret


rq_act:                         ; X16[IX] = pre > 0 ? ACT[pre] : 0
        ld a,h
        or a
        jr nz,.z                ; отрицательное (|pre| <= 127)
        ld de,(CUR + LD_ACT)
        add hl,de
        ld l,(hl)
        ld h,0
        jr .s
.z:     ld hl,0
.s:     push ix
        add ix,ix
        ld de,X16
        add ix,de
        ld (ix+0),l
        ld (ix+1),h
        pop ix
        ret

