; conditional.asm — Demonstrate JZ (jump if zero)
;
; R0 = 0, so JZ should jump over the MOVI R1, 2 instruction.
; Result: R1 = 1 (not 2)

MOVI R0, 0       ; R0 = 0
MOVI R1, 1       ; R1 = 1
JZ R0, skip      ; R0 is 0, so jump to skip
MOVI R1, 2       ; This should be skipped
skip:
HALT
