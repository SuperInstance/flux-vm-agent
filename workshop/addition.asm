; addition.asm — Add two numbers: 3 + 4 = 7
;
; Result: R0 = 7

MOVI R0, 3       ; Load 3 into R0
MOVI R1, 4       ; Load 4 into R1
IADD R0, R0, R1  ; R0 = R0 + R1 = 7
HALT
