; HippoGPT: холодная часть — выборка токена, печать, клавиатура, разбор затравки, главный цикл.
; Арифметика выборки и разбор повторяют ref/sampler.py.

CHAT_TOP  equ 9                 ; первая строка чата (над ней шапка)
CHAT_BOT  equ 22                ; последняя строка чата (под ней строка температуры)
ATTR_USER equ 0x47              ; ярко-белый на чёрном — то, что набрал человек
ATTR_HIPPO equ 0x44             ; ярко-зелёный — то, что пишет Хиппо
ATTR_HINT equ 0x46              ; ярко-жёлтый — подсказки
ATTR_TEMP equ 0x43              ; ярко-пурпурный — температура внизу
TEMP_COL  equ 16                ; "Temperature: 0.6" прижата вправо
MAX_TOK   equ 200               ; длина истории в токенах
PROMPT_MAX equ 29               ; затравка помещается в строку после "> "

; ---------------- главный цикл ----------------
main:
        di
        ld sp,STACK_TOP
        call init_sq
        call cls_chat
        ld a,(TEMP_KEY)
        call set_temp
        call status_off                 ; версия слева внизу
        ld a,ATTR_HINT
        ld (TXT_ATTR),a
        ld hl,MSG_HELLO
        call puts
.loop:  call newline_soft
        ld a,ATTR_USER
        ld (TXT_ATTR),a
        ld a,'>'
        call putc
        ld a,' '
        call putc
        call input_line
        call story
        jr .loop

MSG_HELLO:
        db "Start a story and press ENTER.", 10
        db "BREAK or SPACE stops Hippo.", 10
        db "Keys 1-9, 0 set temperature.", 10, 0

; set_temp: A = цифра '0'..'9' -> температура 0.1..0.9 (0 = 1.0): TQ из таблицы, строка внизу
set_temp:
        ld (TEMP_KEY),a
        ld hl,MSG_TEMP_V + 2
        ld (hl),a                       ; "0.N"
        ld b,'0'
        cp '0'
        jr nz,.lo
        ld (hl),'0'                     ; "1.0"
        inc b
.lo:    ld hl,MSG_TEMP_V
        ld (hl),b
        sub '0'
        add a,a
        ld e,a
        ld d,0
        ld hl,TQTAB
        add hl,de
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld (TQ),de
        ; draw_temp: "Temperature: 0.6" справа в нижней строке
draw_temp:
        ld hl,MSG_TEMP
        ld de,ATTR_TEMP * 256 + TEMP_COL
        ; foot_str: HL = строка с нулём -> нижняя строка с колонки E цветом D; курсор и цвет чата не трогает
foot_str:
        push hl
        ld hl,(CUR_ROW)                 ; CUR_ROW и CUR_COL
        ex (sp),hl
        ld a,(TXT_ATTR)
        push af
        ld a,d
        ld (TXT_ATTR),a
        ld a,23
        ld (CUR_ROW),a
        ld a,e
        ld (CUR_COL),a
.s:     ld a,(hl)
        or a
        jr z,.d
        push hl
        call draw_adv
        pop hl
        inc hl
        jr .s
.d:     pop af
        ld (TXT_ATTR),a
        pop hl
        ld (CUR_ROW),hl
        ret

MSG_TEMP:   db "Temperature: "
MSG_TEMP_V: db "0.", TEMP_KEY_DEFAULT, 0
        assert TEMP_KEY_DEFAULT != '0'
TQTAB:  dw TQ_K0, TQ_K1, TQ_K2, TQ_K3, TQ_K4, TQ_K5, TQ_K6, TQ_K7, TQ_K8, TQ_K9

; ---------------- Хиппо пишет: песочные часы и BREAK ----------------
; status_on / status_off: слева внизу часы и "Writing..." / версия
status_on:
        xor a
        ld (TICKN),a
        call draw_icon
        ld a,ATTR_HINT
        ld (0x5800 + 23 * 32),a
        ld hl,MSG_WRITING
        ld de,ATTR_HINT * 256 + 2
        jr foot_str
status_off:
        ld hl,MSG_VER
        ld de,ATTR_TEMP * 256
        jr foot_str

; tick: зовётся из forward между умножениями (~12 раз на токен): следующий кадр часов и проверка SPACE.
; Регистры сохраняет. Ещё чаще SPACE проверяют сами умножения — в начале каждой строки матрицы.
tick:
        push af
        ld a,0x7F                       ; полуряд SPACE
        in a,(0xFE)
        rra
        jr nc,break_now
        push hl
        push de
        push bc
        ld a,(TICKN)
        inc a
        cp ICON_FRAMES
        jr c,.f
        xor a
.f:     ld (TICKN),a
        call draw_icon
        pop bc
        pop de
        pop hl
        pop af
        ret
; brk_poll: SPACE нажат — сразу break_now; иначе ничего не портит (для длинных циклов без tick)
brk_poll:
        push af
        ld a,0x7F
        in a,(0xFE)
        rra
        jp nc,break_now
        pop af
        ret

; break_now: SPACE (и BREAK — это CAPS SHIFT + SPACE) обрывает историю сразу: стек — к началу story,
; дождаться отпускания и выйти через story.end. Сюда прыгают из tick, из умножений и между токенами.
break_now:
        ld sp,(STORY_SP)
        xor a
        out (0xFE),a                    ; рамка снова чёрная
.rel:   ld a,0x7F                       ; ждать отпускания, чтобы пробел не попал в новую затравку
        in a,(0xFE)
        rra
        jr nc,.rel
        jp story.end

; draw_icon: A = кадр -> знакоместо (23, 0)
draw_icon:
        add a,a
        add a,a
        add a,a
        ld e,a
        ld d,0
        ld hl,ICON
        add hl,de
        ld de,0x4000 + 0x1000 + 7 * 32  ; строка 23, колонка 0
        ld b,8
.l:     ld a,(hl)
        ld (de),a
        inc hl
        inc d
        djnz .l
        ret

; песочные часы: песок сверху -> вниз, потом часы переворачиваются (боком)
ICON:   db 0xFF, 0x7E, 0x3C, 0x18, 0x18, 0x24, 0x42, 0xFF
        db 0xFF, 0x5A, 0x3C, 0x18, 0x18, 0x24, 0x5A, 0xFF
        db 0xFF, 0x42, 0x3C, 0x18, 0x18, 0x24, 0x7E, 0xFF
        db 0xFF, 0x42, 0x24, 0x18, 0x18, 0x3C, 0x7E, 0xFF
        db 0x81, 0xC3, 0xA5, 0x99, 0x99, 0xA5, 0xC3, 0x81
ICON_FRAMES equ ($ - ICON) / 8
MSG_WRITING: db "Writing...", 0
MSG_VER:     db "v1.1 "             ; 12 знакомест, как часы и "Writing..."
    IF SCORPION
             db "256K   ", 0
    ELSE
             db "128K   ", 0
    ENDIF

; story: затравка из PROMPT -> история, токен за токеном
story:
        ld (STORY_SP),sp                ; сюда откатывается tick по BREAK
        ld hl,0                         ; журнал токенов — заново (иначе после ~5 историй вылезет за банк 7)
        ld (GEN_N),hl
        call status_on
        call reset_model
        ld hl,0                         ; конец истории = начало новой
        call forward
        call tokenize                   ; PROMPT -> TOKBUF, B = число токенов
        ld hl,0xFFFF
        ld (NEED),hl
        xor a
        ld (HEAL_SKIP),a
    IF HEAL_WORDS == 0
        ld (NOCONT),a
    ENDIF
        ld a,b
        or a
        jr z,.gen0
    IF HEAL_WORDS == 0
        call last_letter                ; затравка кончается буквой — слово законченное: склейки нет,
        jr nc,.heal0                    ; а первый токен не продолжает слово буквами (nocont)
        ld a,1
        ld (NOCONT),a
        jr .feed0
.heal0:
    ENDIF
        dec b                           ; последний токен — не в модель, а на склейку
        ld l,b
        ld h,0
        add hl,hl
        ld de,TOKBUF
        add hl,de
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld (NEED),de
        ld a,b
        or a
        jr z,.gen0
.feed0: ld hl,TOKBUF
.feed:  push bc
        ld e,(hl)
        inc hl
        ld d,(hl)
        inc hl
        push hl
        ex de,hl
        call forward
        pop hl
        pop bc
        djnz .feed
.gen0:  call prompt_word
        ld a,ATTR_HIPPO
        ld (TXT_ATTR),a
        ld a,MAX_TOK
        ld (GEN_LEFT),a
.gen:   call heal                       ; только перед первым токеном (NEED потом сброшен)
    IF HEAL_WORDS == 0
        call nocont
    ENDIF
    IF MIN_EOS
        ld a,(GEN_LEFT)                 ; первые MIN_EOS токенов сказка не кончается: у токена 0 метка p = 0
        cp MAX_TOK - MIN_EOS + 1
        jr c,.eok
        ld hl,0x8000
        ld (LOGITS),hl
.eok:
    ENDIF
        call sample                     ; HL = токен
        ld a,h
        or l
        jr z,.end
        push hl
        call log_tok
        pop hl
        push hl
        call print_tok
        pop hl
        call forward
        call space_pressed
        jp c,break_now
        ld hl,GEN_LEFT
        dec (hl)
        jr nz,.gen
.end:   call status_off
        ld a,10
        call putc
        ret

    IF HEAL_WORDS == 0
; last_letter: CF = 1, если последний символ затравки — латинская буква. B не трогает
last_letter:
        ld hl,PROMPT - 1
        ld a,(PROMPT_LEN)
        ld e,a
        ld d,0
        add hl,de
        ld a,(hl)
        or 0x20
        sub 'a'
        cp 26
        ret
    ENDIF

; prompt_word: последнее слово затравки -> WORDBUF/WORD_LEN (чтобы перенос подхватил его целиком)
prompt_word:
        xor a
        ld (WORD_LEN),a
        ld a,(PROMPT_LEN)
        or a
        ret z
        ld b,a
        ld e,a
        ld d,0
        ld hl,PROMPT
        add hl,de                       ; HL — за последним символом
        ld c,0
.back:  dec hl
        ld a,(hl)
        cp ' '
        jr z,.got
        inc c
        djnz .back
        dec hl
.got:   inc hl
        ld a,c
        or a
        ret z
        ld (WORD_LEN),a
        ld b,0
        ld de,WORDBUF
        ldir
        ret

; ---------------- выборка ----------------
; sample: LOGITS (суммы), LOGMUL (банк 7) -> HL = токен
sample:
        ld a,7
        call setbank
        ; проход 1: zmax = max(acc * LM)
        ld hl,0
        ld (ZMAX),hl
        ld a,0x80
        ld (ZMAX+2),a                   ; -2^23
        ld ix,LOGITS
        ld hl,LOGMUL
        ld bc,VOCAB
.p1:    call brk_poll
        push bc
        push hl
        ld e,(ix+0)
        ld d,(ix+1)
        ld c,(hl)
        call mul16x8s                   ; B:HL = z
        ld a,(ZMAX+2)                   ; z > zmax ? (24 бита со знаком)
        xor 0x80
        ld c,a
        ld a,b
        xor 0x80
        cp c
        jr c,.n1
        jr nz,.y1
        ld de,(ZMAX)
        or a
        sbc hl,de
        add hl,de
        jr c,.n1
        jr z,.n1
.y1:    ld (ZMAX),hl
        ld a,b
        ld (ZMAX+2),a
.n1:    inc ix
        inc ix
        pop hl
        inc hl
        pop bc
        dec bc
        ld a,b
        or c
        jr nz,.p1
        ; проход 2: p = EXP127[min(255, ((zmax - z) * TQ) >> 16)] -> младший байт LOGITS, сумма
        ld hl,0
        ld (PSUM),hl
        ld ix,LOGITS
        ld hl,LOGMUL
        ld bc,VOCAB
.p2:    call brk_poll
        push bc
        push hl
        ld e,(ix+0)
        ld d,(ix+1)
        ld a,d                          ; -32768 — метка склейки: p = EXP127[255] = 0
        xor 0x80
        or e
        jr z,.big
        ld c,(hl)
        call mul16x8s                   ; B:HL = z
        ex de,hl                        ; d = zmax - z
        ld hl,(ZMAX)
        or a
        sbc hl,de
        ld a,(ZMAX+2)
        sbc a,b
        jr nz,.big                      ; d >= 65536
        ex de,hl
        ld bc,(TQ)
        call mul16x16hi                 ; HL = (d * TQ) >> 16
        ld a,h
        or a
        jr nz,.big
        ld a,l
        jr .lut
.big:   ld a,255
.lut:   ld l,a
        ld h,EXP127 >> 8
        ld a,(hl)
        ld (ix+0),a
        ld e,a
        ld d,0
        ld hl,(PSUM)
        add hl,de
        ld (PSUM),hl
        inc ix
        inc ix
        pop hl
        inc hl
        pop bc
        dec bc
        ld a,b
        or c
        jr nz,.p2
        ; проход 3: порог = (rnd * сумма) >> 16, первый токен с накопленной суммой > порога
        call xrnd                       ; HL = случайное
        ex de,hl
        ld bc,(PSUM)
        call mul16x16hi
        ex de,hl                        ; DE = порог
        ld ix,LOGITS
        ld hl,0                         ; накопленная сумма
        ld bc,0                         ; номер токена
.p3:    ld a,(ix+0)
        add a,l
        ld l,a
        jr nc,.nc
        inc h
.nc:    inc ix
        inc ix
        or a                            ; сумма > порога ?
        sbc hl,de
        add hl,de
        jr z,.nx
        jr nc,.got
.nx:    inc bc
        jr .p3
.got:   ld h,b
        ld l,c
        ret

; mul16x16hi: HL = (DE * BC) >> 16, без знака. Младшее слово произведения — в IY
mul16x16hi:
        ld hl,0
        ld iy,0
        ld a,16
.l:     add iy,iy
        adc hl,hl
        sla e
        rl d
        jr nc,.n
        add iy,bc
        jr nc,.n
        inc hl
.n:     dec a
        jr nz,.l
        ret

; xrnd: HL = следующее значение xorshift16 (7, 9, 8) — Джон Меткалф
xrnd:
        ld hl,(RND)
        ld a,h
        rra
        ld a,l
        rra
        xor h
        ld h,a
        ld a,l
        rra
        ld a,h
        rra
        xor l
        ld l,a
        xor h
        ld h,a
        ld (RND),hl
        ret

; ---------------- печать ----------------
; print_tok: HL = токен -> его строка через putc (строки в банке 7)
print_tok:
        ld a,7
        call setbank
        add hl,hl
        ld de,TOKOFF
        add hl,de
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld a,(de)
        or a
        ret z
        ld b,a
        ld a,(HEAL_SKIP)                ; склейка: начало первого токена уже есть в затравке
        or a
        jr z,.c
        ld c,a
        xor a
        ld (HEAL_SKIP),a
        ld a,b
        sub c
        ret z
        ld b,a
.sk:    inc de
        dec c
        jr nz,.sk
.c:     inc de
        ld a,(de)
        push de
        push bc
        call putc
        pop bc
        pop de
        djnz .c
        ret

; puts: HL -> строка с нулём
puts:   ld a,(hl)
        or a
        ret z
        push hl
        call putc
        pop hl
        inc hl
        jr puts

; putc: A = символ; перенос по словам, прокрутка чата
putc:
        cp 10
        jp z,newline
        cp ' '
        jr nz,.ch
        xor a                           ; пробел: слово кончилось
        ld (WORD_LEN),a
        ld a,(CUR_COL)
        or a
        ret z                           ; в начале строки пробел не печатаем
        cp 32
        jp z,newline
        ld a,' '
        jp draw_adv
.ch:    ld c,a
        ld a,(CUR_COL)
        cp 32
        jr c,.fits
        ; строка полна посреди слова: слово целиком на следующую строку
        ld a,(WORD_LEN)
        or a
        jr z,.nl
        cp 32
        jr nc,.nl                       ; слово длиннее строки — просто перенос
        ld b,a                          ; стереть хвост текущей строки
        ld a,32
        sub b
        ld (CUR_COL),a
.er:    push bc
        ld a,' '
        call draw_char
        ld hl,CUR_COL
        inc (hl)
        pop bc
        djnz .er
        push bc
        call newline_keep
        ld a,(WORD_LEN)                 ; перепечатать начало слова
        ld b,a
        ld hl,WORDBUF
.rp:    ld a,(hl)
        inc hl
        push hl
        push bc
        call draw_adv
        pop bc
        pop hl
        djnz .rp
        pop bc
        jr .fits
.nl:    push bc
        call newline_keep
        pop bc
.fits:  ld a,(WORD_LEN)                 ; в буфер слова
        cp 32
        jr nc,.nob
        ld e,a
        ld d,0
        ld hl,WORDBUF
        add hl,de
        ld (hl),c
        inc a
        ld (WORD_LEN),a
.nob:   ld a,c
        ; draw_adv: нарисовать A в курсоре и сдвинуть курсор
draw_adv:
        call draw_char
        ld hl,CUR_COL
        inc (hl)
        ret

newline:
        xor a
        ld (WORD_LEN),a
newline_keep:                           ; WORD_LEN сохраняется
        xor a
        ld (CUR_COL),a
        ld a,(CUR_ROW)
        inc a
        cp CHAT_BOT + 1
        jr c,.ok
        call scroll_chat
        ld a,CHAT_BOT
.ok:    ld (CUR_ROW),a
        ret

; newline_soft: перевод строки, только если курсор не в её начале
newline_soft:
        ld a,(CUR_COL)
        or a
        ret z
        jp newline

; draw_char: A = символ в (CUR_ROW, CUR_COL) шрифтом ПЗУ, цвет TXT_ATTR
draw_char:
        sub ' '
        ld l,a
        ld h,0
        add hl,hl
        add hl,hl
        add hl,hl
        ld de,0x3D00
        add hl,de
        ex de,hl                        ; DE -> глиф
        ld a,(CUR_ROW)
        ld b,a
        and 0x18
        or 0x40
        ld h,a
        ld a,b
        and 7
        rrca
        rrca
        rrca
        ld l,a
        ld a,(CUR_COL)
        or l
        ld l,a                          ; HL -> экран
        ld b,8
.g:     ld a,(de)
        ld (hl),a
        inc de
        inc h
        djnz .g
        ld a,(CUR_ROW)                  ; атрибут
        ld l,a
        ld h,0
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        ld a,(CUR_COL)
        ld e,a
        ld d,0x58
        add hl,de
        ld a,(TXT_ATTR)
        ld (hl),a
        ret

; scroll_chat: строки CHAT_TOP..CHAT_BOT вверх на одну, нижняя — пустая
scroll_chat:
        ld a,CHAT_TOP
.r:     push af
        ld c,a                          ; строка-приёмник
        inc a
        ld b,a                          ; строка-источник
        call row_addr                   ; HL = адрес строки C, DE = строки B
        ld a,8
.px:    push af
        push hl
        push de
        ex de,hl
        ld bc,32
        ldir
        pop de
        pop hl
        inc h
        inc d
        pop af
        dec a
        jr nz,.px
        pop af
        push af
        ld l,a                          ; атрибуты
        ld h,0
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        add hl,hl
        ld de,0x5800
        add hl,de
        ld d,h
        ld e,l
        ld bc,32
        add hl,bc
        ldir
        pop af
        inc a
        cp CHAT_BOT
        jr nz,.r
        ld c,CHAT_BOT                   ; очистить последнюю
        ld b,CHAT_BOT
        call row_addr
        ld a,8
.cl:    push hl
        ld b,32
.z:     ld (hl),0
        inc l
        djnz .z
        pop hl
        inc h
        dec a
        jr nz,.cl
        ret

; row_addr: C, B — строки знакомест -> HL = экран строки C, DE = строки B (верхние линии)
row_addr:
        ld a,c
        call .one
        push hl
        ld a,b
        call .one
        ex de,hl
        pop hl
        ret
.one:   ld l,a
        and 0x18
        or 0x40
        ld h,a
        ld a,l
        and 7
        rrca
        rrca
        rrca
        ld l,a
        ret

; cls_chat: очистить строки CHAT_TOP..CHAT_BOT, курсор в начало чата
cls_chat:
        ld a,CHAT_TOP
.r:     push af
        ld c,a
        ld b,a
        call row_addr
        ld b,8
.px:    push bc
        push hl
        ld b,32
.z:     ld (hl),0
        inc l
        djnz .z
        pop hl
        inc h
        pop bc
        djnz .px
        pop af
        inc a
        cp CHAT_BOT + 1
        jr nz,.r
        ld hl,0x5800 + CHAT_TOP * 32
        ld de,0x5800 + CHAT_TOP * 32 + 1
        ld bc,(CHAT_BOT + 1 - CHAT_TOP) * 32 - 1
        ld (hl),ATTR_USER
        ldir
        ld a,CHAT_TOP
        ld (CUR_ROW),a
        xor a
        ld (CUR_COL),a
        ld (WORD_LEN),a
        ret

; log_tok: HL = токен -> журнал сгенерированных (для сверки на стенде), банк 7
log_tok:
        ex de,hl
        ld hl,(GEN_N)
        push hl
        add hl,hl
        ld bc,GENLOG
        add hl,bc
        ld (hl),e
        inc hl
        ld (hl),d
        pop hl
        inc hl
        ld (GEN_N),hl
        ret

; ---------------- клавиатура ----------------
; scan_key: A = код нажатой клавиши (0..39) или 255; C = 1, если CAPS SHIFT, B = 1, если SYMBOL SHIFT
scan_key:
        ld bc,0xFEFE
        in a,(c)
        cpl
        and 1
        ld (KEY_CS),a
        ld b,0x7F
        in a,(c)
        cpl
        and 2
        ld (KEY_SS),a
        ld d,0                          ; номер полуряда
        ld b,0xFE
.row:   in a,(c)
        cpl
        and 0x1F
        ld e,a
        ld a,d                          ; убрать CAPS (0/бит0) и SYMBOL (7/бит1)
        or a
        jr nz,.n0
        res 0,e
.n0:    cp 7
        jr nz,.n7
        res 1,e
.n7:    ld a,e
        or a
        jr nz,.hit
        rlc b
        inc d
        ld a,d
        cp 8
        jr nz,.row
        ld a,255
        ret
.hit:   ld a,d                          ; код = ряд*5 + бит
        add a,a
        add a,a
        add a,d
        ld d,a
.bit:   rr e
        jr c,.done
        inc d
        jr .bit
.done:  ld a,d
        ret

; getkey: ждать нажатия, потом отпускания. A = ASCII, 13 = ENTER, 8 = удалить, 0 — не наш символ;
; цифры — температура (в словаре их нет)
getkey:
.wait:  call scan_key
        cp 255
        jr z,.wait
        ld b,0                          ; нажатие есть — выждать ~3 мс и опросить заново вместе с шифтами
.dly:   djnz .dly
        call scan_key
        cp 255
        jr z,.wait
        ld e,a
        ld d,0
        ld hl,KEYMAP
        ld a,(KEY_SS)
        or a
        jr z,.ncs
        ld hl,KEYMAP_SS
        jr .map
.ncs:   ld a,(KEY_CS)
        or a
        jr z,.map
        ld hl,KEYMAP_CS
.map:   add hl,de
        ld a,(hl)
        push af
.rel:   call scan_key                   ; ждать отпускания
        cp 255
        jr nz,.rel
        pop af
        ret

; space_pressed: CF = 1, если нажат SPACE (без ожидания)
space_pressed:
        ld bc,0x7FFE
        in a,(c)
        rra
        ccf
        ret

; полуряды: 0 CS Z X C V | 1 A S D F G | 2 Q W E R T | 3 1 2 3 4 5 | 4 0 9 8 7 6 | 5 P O I U Y |
;           6 ENTER L K J H | 7 SPACE SS M N B
KEYMAP:    db 0,"zxcv", "asdfg", "qwert", "12345", "09876", "poiuy", 13,"lkjh", " ",0,"mnb"
KEYMAP_CS: db 0,"ZXCV", "ASDFG", "QWERT", 0,0,0,0,0, 8,0,0,0,0, "POIUY", 13,"LKJH", " ",0,"MNB"
KEYMAP_SS: db 0,":",0,"?",0, 0,0,0,0,0, 0,0,0,0,0, "!",0,0,0,0, 0,0,0,"'",0, 34,";",0,0,0, 13,0,0,"-",0, " ",0,".",",",0

; input_line: строка в PROMPT (длина PROMPT_LEN), эхо через putc
input_line:
        xor a
        ld (PROMPT_LEN),a
.k:     call cursor_on
        call getkey
        push af
        call cursor_off
        pop af
        cp 13
        jr z,.enter
        cp 8
        jr z,.del
        cp '0'
        jr c,.ch
        cp '9' + 1
        jr nc,.ch
        call set_temp
        jr .k
.ch:    or a
        jr z,.k
        ld c,a
        ld a,(PROMPT_LEN)
        cp PROMPT_MAX
        jr nc,.k
        ld e,a
        ld d,0
        ld hl,PROMPT
        add hl,de
        ld (hl),c
        inc a
        ld (PROMPT_LEN),a
        ld a,c
        call draw_adv
        jr .k
.enter: ld a,r                          ; зерно — от того, когда нажали ENTER
        ld hl,(RND)
        xor l
        ld l,a
        ld a,r
        xor h
        ld h,a
        or l
        jr nz,.sd
        inc hl
.sd:    ld (RND),hl
        ret
.del:   ld a,(PROMPT_LEN)
        or a
        jr z,.k
        dec a
        ld (PROMPT_LEN),a
        ld hl,CUR_COL
        dec (hl)
        ld a,' '
        call draw_char
        jr .k

cursor_on:
        ld a,'_'
        jp draw_char
cursor_off:
        ld a,' '
        jp draw_char

; ---------------- склейка последнего токена затравки ----------------
; heal: если NEED задан — токенам, чья строка не начинается со строки NEED, метка LOGITS = -32768 (sample даст
; p = 0); HEAL_SKIP = длина строки NEED (столько символов первого токена уже напечатано). NEED сбрасывается.
heal:   ld hl,(NEED)
        ld a,h
        and l
        inc a
        ret z                           ; 0xFFFF — склейки нет
        ld a,7
        call setbank
        add hl,hl
        ld de,TOKOFF
        add hl,de
        ld e,(hl)
        inc hl
        ld d,(hl)
        ld (H_STR),de                   ; строка NEED: длина, символы
        ld a,(de)
        ld (HEAL_SKIP),a
        ld hl,0xFFFF
        ld (NEED),hl
        ld ix,LOGITS
        ld hl,TOKOFF
        ld bc,VOCAB
.v:     push bc
        push hl
        ld e,(hl)
        inc hl
        ld d,(hl)
        call starts
        jr c,.ok
        ld (ix+0),0
        ld (ix+1),0x80
.ok:    inc ix
        inc ix
        pop hl
        inc hl
        inc hl
        pop bc
        dec bc
        ld a,b
        or c
        jr nz,.v
        ret

    IF HEAL_WORDS == 0
; nocont: если NOCONT — токенам, чья строка начинается с буквы, метка LOGITS = -32768 (p = 0). NOCONT сбрасывается
nocont: ld a,(NOCONT)
        or a
        ret z
        xor a
        ld (NOCONT),a
        ld a,7
        call setbank
        ld ix,LOGITS
        ld hl,TOKOFF
        ld bc,VOCAB
.v:     push bc
        ld e,(hl)
        inc hl
        ld d,(hl)
        inc hl
        ld a,(de)                       ; длина строки (у токена 0 — пусто)
        or a
        jr z,.ok
        inc de
        ld a,(de)
        or 0x20
        sub 'a'
        cp 26
        jr nc,.ok
        ld (ix+0),0
        ld (ix+1),0x80
.ok:    inc ix
        inc ix
        pop bc
        dec bc
        ld a,b
        or c
        jr nz,.v
        ret
    ENDIF

; starts: DE -> строка токена (длина, символы), H_STR -> строка NEED. CF = 1, если токен начинается с NEED
starts: ld hl,(H_STR)
        ld a,(de)
        cp (hl)
        jr c,.no                        ; короче NEED
        ld b,(hl)
.c:     inc de
        inc hl
        ld a,(de)
        cp (hl)
        jr nz,.no
        djnz .c
        scf
        ret
.no:    or a
        ret

; ---------------- разбор затравки ----------------
; tokenize: PROMPT -> TOKBUF (слова), B = число токенов. Куски как в bpe.PRE, внутри куска — слияния BPE,
; как при обучении (bpe_chunk). Таблицы MERGES и CHARID — в банке 7.
tokenize:
        ld a,7
        call setbank
        ld hl,PROMPT
        ld a,(PROMPT_LEN)
        ld e,a
        ld d,0
        add hl,de
        ld (P_END),hl
        ld hl,TOKBUF
        ld (T_OUT),hl
        xor a
        ld (T_N),a
        ld hl,PROMPT
.chunk: ld de,(P_END)               ; HL = начало куска
        or a
        sbc hl,de
        add hl,de
        jr nc,.done
        push hl                         ; конец куска -> CH_END
        ld a,(hl)
        cp ' '
        jr nz,.nosp
        inc hl
        ld de,(P_END)
        or a
        sbc hl,de
        add hl,de
        jr nc,.end1                     ; одинокий пробел в конце
        ld a,(hl)
        cp ' '
        jr z,.end1
.nosp:  call is_alpha
        jr nc,.punct
.alp:   inc hl
        call at_end
        jr nc,.ce
        ld a,(hl)
        call is_alpha
        jr c,.alp
        jr .ce
.punct: inc hl
        call at_end
        jr nc,.ce
        ld a,(hl)
        cp ' '
        jr z,.ce
        call is_alpha
        jr nc,.punct
        jr .ce
.end1:  pop hl                          ; кусок — один пробел
        push hl
        inc hl
.ce:    ld (CH_END),hl
        pop hl
        call bpe_chunk                  ; HL..CH_END -> токены в TOKBUF; HL = CH_END
        jr .chunk
.done:  ld a,(T_N)
        ld b,a
        ret

at_end: push de                         ; CF = 1, если HL < P_END
        ld de,(P_END)
        or a
        sbc hl,de
        add hl,de
        pop de
        ret

is_alpha:                               ; A — буква? CF = 1
        cp 'A'
        ccf
        ret nc
        cp 'Z' + 1
        ret c
        cp 'a'
        ccf
        ret nc
        cp 'z' + 1
        ret

; bpe_chunk: HL -> кусок (до CH_END) -> его токены дописываются в TOKBUF (T_OUT, T_N); HL = CH_END.
; Символы -> базовые токены, затем слияния по номерам: каждое — все пары слева направо (= ref/sampler.tokenize_bpe).
bpe_chunk:
        ld de,(T_OUT)
        ld (SEG),de
        xor a
        ld (SEG_N),a
.ch:    push de
        ld de,(CH_END)
        or a
        sbc hl,de
        add hl,de
        pop de
        jr nc,.sweep
        ld a,(hl)                       ; символ -> базовый токен (старший байт 0)
        inc hl
        push hl
        ld hl,CHARID
        add a,l
        ld l,a
        jr nc,.n1
        inc h
.n1:    ld a,(hl)
        pop hl
        ld (de),a
        inc de
        xor a
        ld (de),a
        inc de
        ld a,(SEG_N)
        inc a
        ld (SEG_N),a
        jr .ch
.sweep: push hl
        ld ix,MERGES
        ld bc,64                        ; токен слияния номер r = 64 + r
.m:     ld a,(SEG_N)
        cp 2
        jr c,.fin                       ; один токен — сливать нечего
        push bc
        call merge_one
        pop bc
        ld de,4
        add ix,de
        inc bc
        ld a,b
        cp VOCAB >> 8
        jr nz,.m
.fin:   ld a,(SEG_N)                    ; T_OUT += 2 * SEG_N, T_N += SEG_N
        ld e,a
        ld hl,T_N
        add a,(hl)
        ld (hl),a
        ld d,0
        ld hl,(SEG)
        add hl,de
        add hl,de
        ld (T_OUT),hl
        pop hl
        ret

; merge_one: в SEG (SEG_N слов) каждую пару ((ix+0..1), (ix+2..3)) слева направо -> BC
merge_one:
        ld e,(ix+0)
        ld d,(ix+1)                     ; DE = левый
        ld hl,(SEG)
        ld a,(SEG_N)
        dec a                           ; пар впереди
.p:     ld (M_LEFT),a
        ld a,(hl)
        cp e
        jr nz,.nx
        inc hl
        ld a,(hl)
        dec hl
        cp d
        jr nz,.nx
        push hl                         ; левый совпал — правый?
        inc hl
        inc hl
        ld a,(hl)
        cp (ix+2)
        jr nz,.no
        inc hl
        ld a,(hl)
        cp (ix+3)
        jr nz,.no
        pop hl                          ; слить: на место левого — BC, правый убрать
        ld (hl),c
        inc hl
        ld (hl),b
        inc hl                          ; HL -> правый
        push hl
        push bc
        ld a,(M_LEFT)                   ; слов после правого = M_LEFT - 1
        dec a
        jr z,.nomv
        add a,a
        ld c,a
        ld b,0
        ld d,h
        ld e,l
        inc hl
        inc hl
        ldir
.nomv:  pop bc
        pop hl                          ; HL -> слово после слитого
        ld a,(SEG_N)
        dec a
        ld (SEG_N),a
        ld e,(ix+0)
        ld d,(ix+1)
        ld a,(M_LEFT)                   ; пар впереди: на две меньше
        sub 2
        ret c
        ret z
        jr .p
.no:    pop hl
.nx:    inc hl
        inc hl
        ld a,(M_LEFT)
        dec a
        ret z
        jr .p
