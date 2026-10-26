; HippoGPT для ZX Spectrum 128K: трансформер на токенах BPE с 4-битными весами.
;
; Арифметика повторяет ref/intref.py бит в бит. Прерывания запрещены всё время: ядра
; таблиц кратных пишут через PUSH, SP занят. Карта памяти (подробнее — README):
;   банк 5  4000-5AFF экран, 5B00-66FF буферы и стек (обнуляются), 6700-7FFF таблицы, множители, холодный код
;   банк 2  8000-9FFF таблица кратных, A000 квадраты, A200 exp, A300-BFFF горячий код
;   C000    банки 0,1 (W1), 3 (W2), 4 (вложения), 6 (внимание слоя 0, Wqkv слоя 1), 7 (Wo слоя 1, кэш, строки)

        include "gen_params.inc"
    IF SCORPION
        device zxspectrum256
    ELSE
        device zxspectrum128
    ENDIF

; ---------------- банк 5: буферы (не грузятся) ----------------
QKV8    equ 0x5B00              ; QKV_ROWS байт
HB      equ 0x5B60              ; 64: h' для нормы
EBUF    equ 0x5BA0              ; CTX: веса позиций
XF      equ 0x5BE0              ; 64: x последней нормы (байты)
SCORE   equ 0x5C20              ; CTX слов
SCOREP  equ 0x5CA0              ; PIN слов: счёт закреплённых
STACK_TOP equ 0x5E00            ; стек 5D00-5DFF
X16     equ 0x5E00              ; 256 слов: вход matvec
ACC     equ 0x6000              ; 256 слов: выход matvec
HVEC    equ 0x6200              ; 64 слова: остаточный поток
LOGITS  equ 0x6280              ; VOCAB слов: суммы выходного слоя
LOW_END equ LOGITS + VOCAB * 2
        assert LOW_END <= 0x6700

; ---------------- банк 2: рабочие таблицы ----------------
PT      equ 0x8000              ; таблица кратных (байт на вес) — до 2 КБ
PTN     equ 0x8000              ; она же в раскладке «два веса в байт» — до 8 КБ
SQLO    equ 0xA000              ; 512: квадраты, строится init_sq

; ================= банк 5, загружаемая часть =================
        org 0x6700
ORG_START:
RSQ8:   include "gen_rsq8.inc"
EXP127: include "gen_exp127.inc"
ACTTAB: include "gen_act.inc"
        assert (RSQ8 & 255) == 0 && EXP127 == RSQ8 + 256 && ACTTAB == RSQ8 + 512
        include "gen_mults.inc"         ; множители строк всех матриц
COLD_CODE:
        include "ui.asm"
; переменные интерфейса
TXT_ATTR:   db 0x47
CUR_ROW:    db 9
CUR_COL:    db 0
WORD_LEN:   db 0
WORDBUF:    ds 32
ZMAX:       ds 3
PSUM:       dw 0
TQ:         dw TQ_DEFAULT
TEMP_KEY:   db TEMP_KEY_DEFAULT          ; цифра температуры ('6' = 0.6)
TICKN:      db 0                        ; кадр песочных часов
STORY_SP:   dw 0                        ; стек в начале story — для BREAK
RND:        dw 0xACE1
GEN_LEFT:   db 0
GEN_N:      dw 0
KEY_CS:     db 0
KEY_SS:     db 0
PROMPT_LEN: db 0
PROMPT:     ds 32
TOKBUF:     ds 2 * 32
T_OUT:      dw 0
T_N:        db 0
P_END:      dw 0
CH_END:     dw 0
SEG:        dw 0                        ; разбор: начало токенов куска в TOKBUF
SEG_N:      db 0                        ; и их число
M_LEFT:     db 0
NEED:       dw 0xFFFF                   ; склейка: последний токен затравки (0xFFFF — нет)
HEAL_SKIP:  db 0                        ; столько символов первого токена уже напечатано
    IF HEAL_WORDS == 0
NOCONT:     db 0                        ; затравка кончилась буквой: первый токен — не с буквы
    ENDIF
H_STR:      dw 0
BANK5_END:
        assert BANK5_END <= 0x8000

; ================= банк 2, загружаемая часть =================
        org 0xA200
EXP63:  include "gen_exp63.inc"
        assert EXP63 == 0xA200

; ---------------- переменные ----------------
POS:    dw 0
SLOT:   db 0
TOK:    dw 0
EM_M:   db 0
    IF INPUT_TABLES
PREV_TOK: dw 0                  ; предыдущий токен — для входных таблиц
EMB_ADD: db 0                   ; emb_one: 0 — записать слово, иначе прибавить
    ENDIF
NS_S:   db 0
NS_K:   db 0
NS_SH:  db 0
CUR:    ds LAYER_SIZE
AQ:     dw 0
AX:     ds 3
E2:     ds 3
SE:     dw 0
SMAX:   dw 0
HEAD:   db 0
RQ_LO:  dw 0
RQ_HI:  dw 0
RQ_S:   db 0

start:
        jp main

; setbank: A = банк 0..7 на C000 (ПЗУ 48K, обычный экран), портит BC.
; Сборка 256: банк 0..15, бит 3 номера у машин свой. Загрузчик (tools/mkdisk.py) ставит переход на
; sb_pent (бит 6 порта #7FFD: Pentagon 512/1024, ZX Evolution), sb_scorp (бит 4 #1FFD: Scorpion, KAY) или
; sb_profi (бит 0 #DFFD: Profi 1024); на стенде (tools/zx256.py) — Scorpion.
setbank:
    IF SCORPION
        jp sb_scorp
sb_pent:
        bit 3,a
        jr z,setbank_lo
        xor 0x48                        ; бит 3 -> бит 6
        jr setbank_lo
sb_profi:
        push af
        and 8
        rrca
        rrca
        rrca
        ld bc,0xDFFD
        out (c),a
        pop af
        and 7
        jr setbank_lo
sb_scorp:
        push af
        and 8
        add a,a
        ld bc,0x1FFD
        out (c),a
        pop af
        and 7
    ENDIF
setbank_lo:
        or 0x10
        ld bc,0x7FFD
        out (c),a
        ret

        include "gen_matvec4.asm"
        include "gen_matvec4n.asm"
        include "gen_dot.asm"
        include "core.asm"

; ---------------- прямой проход ----------------
; forward: HL = токен (0..VOCAB-1) -> LOGITS
forward:
        ld (TOK),hl
        ld a,1                  ; рамка синяя — Хиппо думает
        out (0xFE),a
        ld a,7                  ; множитель строки вложения
        call setbank
        ld de,EMBMUL
        add hl,de
        ld a,(hl)
        ld (EM_M),a
        ld a,EMB_PG             ; строка: 32 байта по два кода
        call setbank
        ld hl,(TOK)
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        ld de,EMB
        add hl,de
        ld ix,HVEC
        ld b,D_MODEL / 2
.e:     ld a,(hl)
        inc hl
        push hl
        push bc
        ld c,a
        and 0x0F
        call emb_one
        ld a,c
        rrca
        rrca
        rrca
        rrca
        and 0x0F
        call emb_one
        pop bc
        pop hl
        djnz .e
    IF INPUT_TABLES
        call input_tables
    ENDIF
        ld a,(POS)
        and CTX - 1
        ld (SLOT),a
        ld hl,LAYERS
        ld b,N_LAYERS
.layer: push bc
        push hl
        ld de,CUR
        ld bc,LAYER_SIZE
        ldir
        call attn_block
        call mlp_block
        pop hl
        ld de,LAYER_SIZE
        add hl,de
        pop bc
        djnz .layer
        call logits
        ld hl,(POS)
        inc hl
        ld (POS),hl
        xor a
        out (0xFE),a
        ret

; emb_one: A = код+7 -> слово (код * EM_M) по IX, IX += 2. Портит A, DE, HL, B.
emb_one:
        push bc
        sub 7
        or a
        push af
        jp p,.p
        neg
.p:     ld hl,EM_M
        ld e,(hl)
        call mul8u
        pop af
        jp p,.s
        xor a
        sub l
        ld l,a
        sbc a,a
        sub h
        ld h,a
.s:
    IF INPUT_TABLES
        ld a,(EMB_ADD)
        or a
        jr z,.st
        ld a,(ix+0)
        add a,l
        ld l,a
        ld a,(ix+1)
        adc a,h
        ld h,a
.st:
    ENDIF
        ld (ix+0),l
        ld (ix+1),h
        inc ix
        inc ix
        pop bc
        ret

    IF INPUT_TABLES
; input_tables: HVEC += строка таблицы предыдущего токена и строка таблицы пары (предыдущий, текущий)
input_tables:
        ld a,1
        ld (EMB_ADD),a
    IF PREV_EMB
        ld a,TAB_PG
        call setbank
        ld hl,(PREV_TOK)
        ld de,PREVMUL
        add hl,de
        ld a,(hl)
        ld (EM_M),a
        ld a,PREV_PG
        call setbank
        ld hl,(PREV_TOK)
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        ld de,0xC000
        add hl,de
        call emb_row_add
    ENDIF
    IF BIG_ROWS
        ld a,TAB_PG             ; строка пары = (H1[предыдущий] ^ H2[текущий]) & (BIG_ROWS - 1)
        call setbank
        ld hl,(PREV_TOK)
        add hl,hl
        ld de,BIGH1
        add hl,de
        ld c,(hl)
        inc hl
        ld b,(hl)
        ld hl,(TOK)
        add hl,hl
        ld de,BIGH2
        add hl,de
        ld a,(hl)
        xor c
        ld c,a
        inc hl
        ld a,(hl)
        xor b
        and (BIG_ROWS - 1) >> 8
        ld b,a                  ; BC = строка
        ld hl,BIGMUL
        add hl,bc
        ld a,(hl)
        ld (EM_M),a
        ld a,b                  ; страница — из BIGPG (512 строк по 32 байта на страницу)
        rra
        and 0x7F
        ld hl,BIGPG
        add a,l
        ld l,a
        adc a,h
        sub l
        ld h,a
        ld a,(hl)
        push bc
        call setbank            ; портит BC
        pop bc
        ld a,b
        and 1
        ld h,a
        ld l,c
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        ld de,0xC000
        add hl,de
        call emb_row_add
    ENDIF
        xor a
        ld (EMB_ADD),a
        ld hl,(TOK)
        ld (PREV_TOK),hl
        ret

; emb_row_add: HL -> строка из D_MODEL кодов по два в байт, множитель EM_M -> HVEC += код * EM_M
emb_row_add:
        ld ix,HVEC
        ld b,D_MODEL / 2
.e:     ld a,(hl)
        inc hl
        push hl
        push bc
        ld c,a
        and 0x0F
        call emb_one
        ld a,c
        rrca
        rrca
        rrca
        rrca
        and 0x0F
        call emb_one
        pop bc
        pop hl
        djnz .e
        ret
    ENDIF

; ---------------- внимание ----------------
attn_block:
        call tick
        call norm_words
        ld b,D_MODEL
        ld ix,X16
        call pt_build
        ld a,(CUR + LD_WQKV_PG)
        call setbank
        ld a,D_MODEL / 64
        ld (mv4_n+2),a
        ld de,(CUR + LD_WQKV)
        ld ix,ACC
        ld iy,QKV_ROWS * 256
        call matvec4
        ld hl,-63
        ld de,63
        ld a,(CUR + LD_SQKV)
        ld bc,rq_store8
        call set_rq
        ld hl,ACC
        ld de,(CUR + LD_MQKV)
        ld b,QKV_ROWS
        call requant
        ld a,7                  ; кэш и добавки по расстоянию — в банке 7
        call setbank
        call kv_store
        xor a
        ld (HEAD),a
.head:  call tick                       ; по голове — часы и BREAK (внимание долгое)
        call attn_head
        ld hl,HEAD
        inc (hl)
        ld a,(hl)
        cp N_HEADS
        jr nz,.head
        call tick
        ld b,QDIM               ; o -> Wo
        ld ix,X16
        call pt_build
        ld a,(CUR + LD_WO_PG)
        call setbank
        ld a,QDIM / 64
        ld (mv4_n+2),a
        ld de,(CUR + LD_WO)
        ld ix,ACC
        ld iy,D_MODEL * 256
        call matvec4
        ld hl,-32768
        ld de,32767
        ld a,(CUR + LD_SO)
        ld bc,rq_addh
        call set_rq
        ld hl,ACC
        ld de,(CUR + LD_MO)
        ld b,D_MODEL
        jp requant

; ---------------- MLP ----------------
mlp_block:
        call tick
        call norm_words
        ld b,D_MODEL
        ld ix,X16
        call pt_build
        ld a,(CUR + LD_W1_PG)
        call setbank
        ld a,D_MODEL / 64
        ld (mv4_n+2),a
        ld de,(CUR + LD_W1)
        ld ix,ACC
        ld iy,(D_FFN & 255) * 256
        call matvec4
        ld hl,-127
        ld de,127
        ld a,(CUR + LD_S1)
        ld bc,rq_act
        call set_rq
        ld hl,ACC
        ld de,(CUR + LD_M1)
        ld b,D_FFN & 255
        call requant
        call tick
        ld b,D_FFN & 255        ; u -> W2, два веса в байт
        ld ix,X16
        call ptn_build
        ld a,(CUR + LD_W2_PG)
        call setbank
        ld a,D_FFN / 32
        ld (mv4n_g+2),a
        ld de,(CUR + LD_W2)
        ld ix,ACC
        ld iy,D_MODEL * 256
        call matvec4n
        ld hl,-32768
        ld de,32767
        ld a,(CUR + LD_S2)
        ld bc,rq_addh
        call set_rq
        ld hl,ACC
        ld de,(CUR + LD_M2)
        ld b,D_MODEL
        jp requant

; ---------------- выходной слой ----------------
; LOGITS[v] = сумма код[v][j] * x[j], x — последняя норма (+-63); масштаб строки — при выборке
logits:
        call norm_bytes
        ld hl,XF                ; байты -> слова для таблицы кратных
        ld ix,X16
        ld b,D_MODEL
.w:     ld a,(hl)
        inc hl
        ld (ix+0),a
        rla
        sbc a,a
        ld (ix+1),a
        inc ix
        inc ix
        djnz .w
        ld b,D_MODEL
        ld ix,X16
        call ptn_build
        ld a,EMB_PG
        call setbank
        ld a,D_MODEL / 32
        ld (mv4n_g+2),a
        ld de,EMB
        ld ix,LOGITS
        ld b,4                          ; четыре куска по VOCAB/4 строк, между ними — tick
.ch:    push bc
        push ix
        ld iy,(VOCAB / 4) * 256
        call matvec4n                   ; DE переходит на следующую строку сам
        call tick
        pop ix
        ld bc,VOCAB / 2                 ; VOCAB/4 слов
        add ix,bc
        pop bc
        djnz .ch
        ret

; ---------------- сброс ----------------
reset_model:
        ld hl,0
        ld (POS),hl
    IF INPUT_TABLES
        ld (PREV_TOK),hl
    ENDIF
        ld a,7
        call setbank
        ld hl,KV_ZERO
        ld de,KV_ZERO + 1
        ld bc,KV_ZERO_LEN - 1
        ld (hl),0
        ldir
    IF PIN
        ld hl,PIN_ZERO
        ld de,PIN_ZERO + 1
        ld bc,PIN_ZERO_LEN - 1
        ld (hl),0
        ldir
    ENDIF
        ret

HOT_END:
        assert HOT_END <= 0xC000
        display "банк 5: ", /h, ORG_START, "..", /h, BANK5_END, "  банк 2: ", /h, EXP63, "..", /h, HOT_END

        include "gen_banks.asm"         ; веса, вложения, кэш, строки — по банкам C000

; образы банков для стенда и упаковки
        page 0
        savebin "bank0.bin", 0xC000, 0x4000
        page 1
        savebin "bank1.bin", 0xC000, 0x4000
        page 3
        savebin "bank3.bin", 0xC000, 0x4000
        page 4
        savebin "bank4.bin", 0xC000, 0x4000
        page 6
        savebin "bank6.bin", 0xC000, 0x4000
        page 7
        savebin "bank7.bin", 0xC000, 0x4000
    IF SCORPION
        page 8
        savebin "bank8.bin", 0xC000, 0x4000
        page 9
        savebin "bank9.bin", 0xC000, 0x4000
        page 10
        savebin "bank10.bin", 0xC000, 0x4000
        page 11
        savebin "bank11.bin", 0xC000, 0x4000
        page 12
        savebin "bank12.bin", 0xC000, 0x4000
        page 13
        savebin "bank13.bin", 0xC000, 0x4000
        page 14
        savebin "bank14.bin", 0xC000, 0x4000
        page 15
        savebin "bank15.bin", 0xC000, 0x4000
    ENDIF
        savebin "bank5.bin", 0x4000, 0x4000
        savebin "bank2.bin", 0x8000, 0x4000
