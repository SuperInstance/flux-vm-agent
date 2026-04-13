"""
FLUX Bytecode Virtual Machine — standalone bytecode interpreter.

Architecture:
  - 16 general-purpose registers (R0-R15)
  - LIFO operand stack
  - Program counter with relative jumps
  - Cycle-based execution limit for safety

Instruction set (14 opcodes):
  Data movement:  MOVI(0x2B), MOV(0x2C)
  Arithmetic:     IADD(0x08), ISUB(0x09), IMUL(0x0A), IDIV(0x0B)
  Unary:          INC(0x0E), DEC(0x0F)
  Compare:        CMP(0x0D)
  Control flow:   JZ(0x07), JNZ(0x06), JMP(0x05), HALT(0x80)
  Stack:          PUSH(0x01), POP(0x02)
"""

from __future__ import annotations

from typing import List, Optional


# ─── Opcode Table ────────────────────────────────────────────────────────────

OPCODES: dict[str, int] = {
    # Data movement
    "MOVI": 0x2B,
    "MOV": 0x2C,
    # Arithmetic
    "IADD": 0x08,
    "ISUB": 0x09,
    "IMUL": 0x0A,
    "IDIV": 0x0B,
    # Unary
    "INC": 0x0E,
    "DEC": 0x0F,
    # Compare
    "CMP": 0x0D,
    # Control flow
    "JZ": 0x07,
    "JNZ": 0x06,
    "JMP": 0x05,
    "HALT": 0x80,
    # Stack
    "PUSH": 0x01,
    "POP": 0x02,
}

# Reverse lookup: opcode byte → mnemonic
OPCODE_NAMES: dict[int, str] = {v: k for k, v in OPCODES.items()}


# ─── VM State ────────────────────────────────────────────────────────────────

class VMState:
    """Snapshot of VM state at a given point in time.

    Attributes:
        pc: Program counter (byte offset into bytecode).
        gp: Copy of all 16 general-purpose register values.
        stack: Copy of the operand stack.
        halted: Whether the VM has executed HALT.
        cycles: Number of instructions executed so far.
        error: Error message if execution faulted, else None.
    """

    __slots__ = ("pc", "gp", "stack", "halted", "cycles", "error")

    def __init__(
        self,
        pc: int,
        gp: List[int],
        stack: List[int],
        halted: bool,
        cycles: int,
        error: Optional[str],
    ) -> None:
        self.pc = pc
        self.gp = list(gp)
        self.stack = list(stack)
        self.halted = halted
        self.cycles = cycles
        self.error = error

    def __repr__(self) -> str:
        regs = ", ".join(f"R{i}={self.gp[i]}" for i in range(16) if self.gp[i] != 0)
        return (
            f"VMState(pc={self.pc}, halted={self.halted}, "
            f"cycles={self.cycles}, regs=[{regs}], "
            f"stack_depth={len(self.stack)}, error={self.error!r})"
        )


# ─── FLUX Virtual Machine ───────────────────────────────────────────────────

class FluxVM:
    """FLUX bytecode virtual machine with 16 registers and a stack.

    The VM executes FLUX bytecode loaded from a ``bytes`` object. Execution
    continues until a ``HALT`` instruction is encountered, the program counter
    reaches the end of the bytecode, or the cycle limit is exceeded.

    Args:
        bytecode: Raw FLUX bytecode to execute.
        max_cycles: Safety limit on instruction count (default 10 million).

    Example::

        from flux_vm import FluxVM, OPCODES

        # Manually construct: MOVI R0, 3; MOVI R1, 4; IADD R0, R0, R1; HALT
        bc = bytes([0x2B, 0x00, 0x03, 0x00, 0x2B, 0x01, 0x04, 0x00,
                     0x08, 0x00, 0x00, 0x01, 0x80])
        vm = FluxVM(bc)
        vm.execute()
        assert vm.result(0) == 7
    """

    MAX_CYCLES: int = 10_000_000

    def __init__(self, bytecode: bytes, max_cycles: Optional[int] = None) -> None:
        self.bc: bytes = bytecode
        self.gp: List[int] = [0] * 16  # R0-R15
        self.pc: int = 0
        self.halted: bool = False
        self.cycles: int = 0
        self.max_cycles: int = max_cycles if max_cycles is not None else self.MAX_CYCLES
        self.error: Optional[str] = None
        self.stack: List[int] = []
        self._trace: List[VMState] = []  # Optional execution trace

    # ── Register access ───────────────────────────────────────────────────

    def reg(self, idx: int) -> int:
        """Read the value of register *idx* (0–15). Returns 0 for out-of-range."""
        return self.gp[idx] if 0 <= idx < 16 else 0

    def set_reg(self, idx: int, value: int) -> None:
        """Write *value* to register *idx* (0–15). No-op for out-of-range."""
        if 0 <= idx < 16:
            self.gp[idx] = value

    # ── Memory model (bytecode read) ──────────────────────────────────────

    def _u8(self) -> int:
        """Read an unsigned 8-bit value and advance PC."""
        v = self.bc[self.pc]
        self.pc += 1
        return v

    def _i16(self) -> int:
        """Read a signed 16-bit little-endian value and advance PC by 2."""
        lo = self.bc[self.pc]
        hi = self.bc[self.pc + 1]
        self.pc += 2
        val = lo | (hi << 8)
        return val - 65536 if val >= 32768 else val

    # ── Stack operations ──────────────────────────────────────────────────

    def stack_push(self, value: int) -> None:
        """Push *value* onto the operand stack."""
        self.stack.append(value)

    def stack_pop(self) -> int:
        """Pop and return the top value from the operand stack.

        Raises:
            RuntimeError: If the stack is empty (stack underflow).
        """
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()

    def stack_depth(self) -> int:
        """Return the current number of items on the stack."""
        return len(self.stack)

    # ── Execution ─────────────────────────────────────────────────────────

    def execute(self) -> "FluxVM":
        """Run until HALT, end-of-bytecode, cycle limit, or error.

        Returns:
            ``self`` for method chaining.
        """
        self.halted = False
        self.cycles = 0
        self.error = None

        try:
            while (
                not self.halted
                and self.pc < len(self.bc)
                and self.cycles < self.max_cycles
            ):
                # Optional: record trace before each instruction
                if self._trace is not None:
                    self._trace.append(self.snapshot())

                op = self._u8()
                self.cycles += 1
                self._dispatch(op)

        except Exception as exc:
            self.error = str(exc)

        return self

    def _dispatch(self, op: int) -> None:
        """Dispatch a single opcode. Internal — called by :meth:`execute`."""

        if op == 0x80:  # HALT
            self.halted = True

        elif op == 0x2B:  # MOVI Rd, imm16
            d = self._u8()
            self.gp[d] = self._i16()

        elif op == 0x2C:  # MOV Rd, Ra
            d = self._u8()
            a = self._u8()
            self.gp[d] = self.gp[a]

        elif op == 0x08:  # IADD Rd, Ra, Rb
            d, a, b = self._u8(), self._u8(), self._u8()
            self.gp[d] = self.gp[a] + self.gp[b]

        elif op == 0x09:  # ISUB Rd, Ra, Rb
            d, a, b = self._u8(), self._u8(), self._u8()
            self.gp[d] = self.gp[a] - self.gp[b]

        elif op == 0x0A:  # IMUL Rd, Ra, Rb
            d, a, b = self._u8(), self._u8(), self._u8()
            self.gp[d] = self.gp[a] * self.gp[b]

        elif op == 0x0B:  # IDIV Rd, Ra, Rb
            d, a, b = self._u8(), self._u8(), self._u8()
            if self.gp[b] == 0:
                raise RuntimeError("Division by zero")
            self.gp[d] = int(self.gp[a] / self.gp[b])

        elif op == 0x0E:  # INC Rd
            self.gp[self._u8()] += 1

        elif op == 0x0F:  # DEC Rd
            self.gp[self._u8()] -= 1

        elif op == 0x0D:  # CMP Ra, Rb → R13 = sign(Ra - Rb)
            a, b = self._u8(), self._u8()
            if self.gp[a] < self.gp[b]:
                self.gp[13] = -1
            elif self.gp[a] > self.gp[b]:
                self.gp[13] = 1
            else:
                self.gp[13] = 0

        elif op == 0x07:  # JZ Rd, offset
            d = self._u8()
            off = self._i16()
            if self.gp[d] == 0:
                self.pc += off

        elif op == 0x06:  # JNZ Rd, offset
            d = self._u8()
            off = self._i16()
            if self.gp[d] != 0:
                self.pc += off

        elif op == 0x05:  # JMP offset
            off = self._i16()
            self.pc += off

        elif op == 0x01:  # PUSH Rd
            self.stack.append(self.gp[self._u8()])

        elif op == 0x02:  # POP Rd
            if not self.stack:
                raise RuntimeError("Stack underflow")
            self.gp[self._u8()] = self.stack.pop()

        else:
            raise ValueError(f"Unknown opcode: 0x{op:02X} at PC={self.pc - 1}")

    # ── I/O operations ────────────────────────────────────────────────────

    def dump_registers(self) -> str:
        """Return a human-readable dump of all non-zero registers."""
        lines: list[str] = []
        for i in range(16):
            lines.append(f"  R{i:2d} = {self.gp[i]}")
        return "\n".join(lines)

    def dump_stack(self) -> str:
        """Return a human-readable dump of the operand stack (top last)."""
        if not self.stack:
            return "  (empty)"
        return "\n".join(f"  [{i}] = {v}" for i, v in enumerate(self.stack))

    # ── Snapshot / trace ──────────────────────────────────────────────────

    def snapshot(self) -> VMState:
        """Capture a snapshot of the current VM state."""
        return VMState(
            pc=self.pc,
            gp=self.gp,
            stack=self.stack,
            halted=self.halted,
            cycles=self.cycles,
            error=self.error,
        )

    def enable_trace(self, enabled: bool = True) -> None:
        """Enable or disable per-instruction execution tracing."""
        if enabled:
            if self._trace is None:
                self._trace = []
        else:
            self._trace = []

    def get_trace(self) -> List[VMState]:
        """Return recorded execution trace (list of VMState snapshots)."""
        return list(self._trace) if self._trace is not None else []

    # ── Result retrieval ──────────────────────────────────────────────────

    def result(self, reg: int = 0) -> Optional[int]:
        """Get register *reg* value if execution halted cleanly.

        Returns ``None`` if the VM did not halt or an error occurred.
        """
        return self.gp[reg] if (self.halted and self.error is None) else None

    def is_success(self) -> bool:
        """Return ``True`` if the VM halted without error."""
        return self.halted and self.error is None

    # ── Dunder ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "HALTED" if self.halted else "RUNNING"
        if self.error:
            status += f" (error: {self.error})"
        return (
            f"FluxVM(status={status}, pc={self.pc}, "
            f"cycles={self.cycles}, R0={self.gp[0]})"
        )
