; factorial.asm — Compute 5! = 120
;
; Uses a loop that multiplies an accumulator by a decrementing counter.
; Result: R1 = 120

MOVI R0, 5       ; R0 = n = 5
MOVI R1, 1       ; R1 = accumulator = 1
loop:
  IMUL R1, R1, R0 ; accumulator *= n
  DEC R0           ; n--
  JNZ R0, loop     ; if n != 0, continue loop
HALT
