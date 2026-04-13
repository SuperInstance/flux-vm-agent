; fibonacci.asm — Compute Fibonacci(7) = 13
;
; Iterative Fibonacci: R2 holds the result.
; Result: R2 = 13

MOVI R0, 7       ; R0 = n = 7
MOVI R1, 0       ; R1 = F(0) = 0
MOVI R2, 1       ; R2 = F(1) = 1
DEC R0           ; decrement loop counter
loop:
  MOV R3, R2      ; R3 = R2 (save current)
  IADD R2, R2, R1 ; R2 = R2 + R1 (next fibonacci)
  MOV R1, R3      ; R1 = old R2
  DEC R0          ; counter--
  JNZ R0, loop    ; continue if counter != 0
HALT
