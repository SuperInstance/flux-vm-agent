"""
FLUX Assembler & Disassembler — bidirectional assembly ↔ bytecode conversion.

Assembler (text → bytecode):
  - Two-pass assembly: first pass collects labels, second emits bytes.
  - Supports comments (``#`` and ``;``), labels (``name:``), and symbolic
    operands (``R0``–``R15``).
  - Immediate values can be integers or label references resolved as
    relative offsets from the current PC.

Disassembler (bytecode → text):
  - Produces human-readable disassembly with hex addresses.
  - Round-trips cleanly with the assembler for all supported opcodes.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

from flux_vm import OPCODES, OPCODE_NAMES


# ─── Instruction Encoding Helpers ────────────────────────────────────────────

def encode_u8(value: int) -> bytes:
    """Encode an unsigned 8-bit integer."""
    return struct.pack("<B", value & 0xFF)


def encode_i16(value: int) -> bytes:
    """Encode a signed 16-bit little-endian integer."""
    return struct.pack("<h", value)


def decode_i16(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode a signed 16-bit LE integer from *data* at *offset*.

    Returns:
        (value, new_offset) tuple.
    """
    lo = data[offset]
    hi = data[offset + 1]
    val = lo | (hi << 8)
    if val >= 32768:
        val -= 65536
    return val, offset + 2


def instruction_size(mnemonic: str) -> int:
    """Return the byte size of an instruction given its mnemonic.

    Raises:
        ValueError: If the mnemonic is unknown.
    """
    mn = mnemonic.upper()
    if mn == "HALT":
        return 1
    if mn in ("DEC", "INC", "PUSH", "POP"):
        return 2
    if mn in ("IADD", "ISUB", "IMUL", "IDIV"):
        return 4
    if mn in ("MOV", "CMP"):
        return 3
    if mn in ("MOVI", "JNZ", "JZ"):
        return 4
    if mn == "JMP":
        return 3
    raise ValueError(f"Unknown mnemonic: {mn}")


def parse_register(token: str) -> int:
    """Parse a register token like ``R0``–``R15`` into its index.

    Raises:
        ValueError: If the token is not a valid register name.
    """
    token = token.strip().upper()
    if token.startswith("R") and token[1:].isdigit():
        idx = int(token[1:])
        if 0 <= idx < 16:
            return idx
    raise ValueError(f"Invalid register: {token}")


# ─── Assembler ───────────────────────────────────────────────────────────────

class FluxAssembler:
    """Two-pass FLUX assembler: text assembly → bytecode.

    Features:
      - Label resolution with relative offsets for jump targets.
      - Comment stripping (``#`` and ``;`` prefixes).
      - Blank-line tolerance.
      - Symbolic register names (``R0``–``R15``).

    Example::

        asm = FluxAssembler()
        bytecode = asm.assemble(\"\"\"
            MOVI R0, 42
            MOVI R1, 8
            IADD R0, R0, R1
            HALT
        \"\"\")
    """

    def __init__(self) -> None:
        self.opcodes: Dict[str, int] = dict(OPCODES)
        self.labels: Dict[str, int] = {}
        self.lines: List[str] = []

    def assemble(self, text: str) -> bytes:
        """Assemble FLUX assembly text into bytecode.

        Args:
            text: Multi-line assembly source code.

        Returns:
            Compiled bytecode as ``bytes``.

        Raises:
            ValueError: If an unknown mnemonic or unresolvable symbol is found.
        """
        self.labels = {}
        self.lines = []

        # ── First pass: collect labels, strip comments, compute sizes ────
        pc = 0
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue

            # Handle label definitions
            if ":" in line:
                label_part, _, rest = line.partition(":")
                self.labels[label_part.strip()] = pc
                line = rest.strip()
                if not line:
                    continue

            self.lines.append(line)

            # Compute instruction byte size
            mn = line.replace(",", " ").split()[0].upper()
            pc += instruction_size(mn)

        # ── Second pass: emit bytecode ────────────────────────────────────
        bc = bytearray()
        for line in self.lines:
            parts = line.replace(",", " ").split()
            mn = parts[0].upper()

            if mn not in self.opcodes:
                raise ValueError(f"Unknown mnemonic: {mn}")

            op = self.opcodes[mn]

            if mn == "HALT":
                bc.append(op)

            elif mn in ("DEC", "INC", "PUSH", "POP"):
                bc.append(op)
                bc.append(parse_register(parts[1]))

            elif mn in ("IADD", "ISUB", "IMUL", "IDIV"):
                bc.append(op)
                bc.append(parse_register(parts[1]))
                bc.append(parse_register(parts[2]))
                bc.append(parse_register(parts[3]))

            elif mn in ("MOV", "CMP"):
                bc.append(op)
                bc.append(parse_register(parts[1]))
                bc.append(parse_register(parts[2]))

            elif mn == "MOVI":
                bc.append(op)
                bc.append(parse_register(parts[1]))
                val = self._resolve_value(parts[2], len(bc) + 2)
                bc.extend(encode_i16(val))

            elif mn in ("JNZ", "JZ"):
                bc.append(op)
                bc.append(parse_register(parts[1]))
                val = self._resolve_value(parts[2], len(bc) + 2)
                bc.extend(encode_i16(val))

            elif mn == "JMP":
                bc.append(op)
                val = self._resolve_value(parts[1], len(bc) + 2)
                bc.extend(encode_i16(val))

        return bytes(bc)

    def _resolve_value(self, token: str, current_pc: int) -> int:
        """Resolve a token to an integer value.

        Tries integer literal first, then label (converted to relative
        offset from *current_pc*).

        Raises:
            ValueError: If the token cannot be resolved.
        """
        token = token.strip()
        try:
            return int(token)
        except ValueError:
            pass
        if token in self.labels:
            return self.labels[token] - current_pc
        raise ValueError(f"Cannot resolve: {token}")


# ─── Disassembler ────────────────────────────────────────────────────────────

class FluxDisassembler:
    """FLUX disassembler: bytecode → human-readable text.

    Produces output with hex addresses::

        0000: MOVI R0, 42
        0003: HALT

    Example::

        dis = FluxDisassembler()
        text = dis.disassemble(bytecode)
    """

    def __init__(self) -> None:
        self.mnemonics: Dict[int, str] = dict(OPCODE_NAMES)

    def disassemble(self, bytecode: bytes) -> str:
        """Disassemble bytecode into human-readable text.

        Args:
            bytecode: Raw FLUX bytecode.

        Returns:
            Multi-line disassembly string.
        """
        lines: List[str] = []
        pc = 0

        while pc < len(bytecode):
            addr = pc
            op = bytecode[pc]
            pc += 1
            mn = self.mnemonics.get(op, f"??? (0x{op:02X})")

            if op == 0x80:  # HALT
                lines.append(f"{addr:04X}: {mn}")

            elif op in (0x0F, 0x0E, 0x01, 0x02):  # DEC, INC, PUSH, POP
                rd = bytecode[pc]
                pc += 1
                lines.append(f"{addr:04X}: {mn} R{rd}")

            elif op in (0x08, 0x09, 0x0A, 0x0B):  # IADD, ISUB, IMUL, IDIV
                rd = bytecode[pc]
                ra = bytecode[pc + 1]
                rb = bytecode[pc + 2]
                pc += 3
                lines.append(f"{addr:04X}: {mn} R{rd}, R{ra}, R{rb}")

            elif op in (0x2C, 0x0D):  # MOV, CMP
                ra = bytecode[pc]
                rb = bytecode[pc + 1]
                pc += 2
                lines.append(f"{addr:04X}: {mn} R{ra}, R{rb}")

            elif op in (0x2B, 0x06, 0x07):  # MOVI, JNZ, JZ
                rd = bytecode[pc]
                pc += 1
                val, pc = decode_i16(bytecode, pc)
                lines.append(f"{addr:04X}: {mn} R{rd}, {val}")

            elif op == 0x05:  # JMP
                val, pc = decode_i16(bytecode, pc)
                lines.append(f"{addr:04X}: {mn} {val}")

            else:
                lines.append(f"{addr:04X}: {mn}")

        return "\n".join(lines)

    def disassemble_with_bytes(self, bytecode: bytes) -> str:
        """Disassemble bytecode including raw hex bytes for each instruction.

        Returns:
            Multi-line string with address, raw bytes, and mnemonic.
        """
        lines: List[str] = []
        pc = 0

        while pc < len(bytecode):
            addr = pc
            op = bytecode[pc]
            mn = self.mnemonics.get(op, f"??? (0x{op:02X})")

            if op == 0x80:
                raw = f"{op:02X}"
                pc += 1
                lines.append(f"{addr:04X}: {raw:12s}  {mn}")

            elif op in (0x0F, 0x0E, 0x01, 0x02):
                rd = bytecode[pc + 1]
                raw = f"{op:02X} {rd:02X}"
                pc += 2
                lines.append(f"{addr:04X}: {raw:12s}  {mn} R{rd}")

            elif op in (0x08, 0x09, 0x0A, 0x0B):
                rd, ra, rb = bytecode[pc + 1], bytecode[pc + 2], bytecode[pc + 3]
                raw = f"{op:02X} {rd:02X} {ra:02X} {rb:02X}"
                pc += 4
                lines.append(f"{addr:04X}: {raw:12s}  {mn} R{rd}, R{ra}, R{rb}")

            elif op in (0x2C, 0x0D):
                ra, rb = bytecode[pc + 1], bytecode[pc + 2]
                raw = f"{op:02X} {ra:02X} {rb:02X}"
                pc += 3
                lines.append(f"{addr:04X}: {raw:12s}  {mn} R{ra}, R{rb}")

            elif op in (0x2B, 0x06, 0x07):
                rd = bytecode[pc + 1]
                val, _ = decode_i16(bytecode, pc + 2)
                raw = f"{op:02X} {rd:02X} {bytecode[pc+2]:02X} {bytecode[pc+3]:02X}"
                pc += 4
                lines.append(f"{addr:04X}: {raw:12s}  {mn} R{rd}, {val}")

            elif op == 0x05:
                val, _ = decode_i16(bytecode, pc + 1)
                raw = f"{op:02X} {bytecode[pc+1]:02X} {bytecode[pc+2]:02X}"
                pc += 3
                lines.append(f"{addr:04X}: {raw:12s}  {mn} {val}")

            else:
                pc += 1
                lines.append(f"{addr:04X}: {op:02X}           {mn}")

        return "\n".join(lines)


# ─── Convenience ─────────────────────────────────────────────────────────────

def assemble(text: str) -> bytes:
    """Convenience function: assemble text to bytecode."""
    return FluxAssembler().assemble(text)


def disassemble(bytecode: bytes) -> str:
    """Convenience function: disassemble bytecode to text."""
    return FluxDisassembler().disassemble(bytecode)
