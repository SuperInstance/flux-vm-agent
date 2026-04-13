; sum_range.asm — Sum integers from 1 to 100 = 5050
;
; Uses CMP to check if we've reached the upper bound.
; Result: R2 = 5050

MOVI R0, 1       ; R0 = current = 1
MOVI R1, 100     ; R1 = upper bound = 100
MOVI R2, 0       ; R2 = accumulator = 0
loop:
  IADD R2, R2, R0 ; accumulator += current
  CMP R0, R1      ; compare current with upper bound
  JZ R13, done    ; if equal, we're done
  INC R0          ; current++
  JNZ R13, loop   ; continue (JNZ R13 checks if CMP result != 0)
done:
HALT
