; stack_ops.asm — Demonstrate PUSH/POP with LIFO ordering
;
; Push 1, then 2. Pop → R2 gets 2, R3 gets 1.
; Result: R2 = 2, R3 = 1

MOVI R0, 1       ; R0 = 1
MOVI R1, 2       ; R1 = 2
PUSH R0          ; stack: [1]
PUSH R1          ; stack: [1, 2]
POP R2           ; R2 = 2 (top of stack)
POP R3           ; R3 = 1
HALT
