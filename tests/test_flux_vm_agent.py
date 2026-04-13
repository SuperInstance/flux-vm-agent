#!/usr/bin/env python3
"""
FLUX VM Agent — comprehensive test suite.

Tests:
  - VM execution (arithmetic, control flow, stack, error handling)
  - Assembler/disassembler round-trip
  - NLP interpreter (vocabulary matching and execution)
  - LCAR bridge (opcode translation, task scheduling)
  - CLI argument parsing

All 32 original tests from flux-py are preserved, plus additional
tests for the new modules (LCAR bridge, CLI, workshop examples).

Run:
  cd fleet/flux-vm-agent && python -m pytest tests/test_flux_vm_agent.py -v
  — or —
  cd fleet/flux-vm-agent && python tests/test_flux_vm_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# Ensure the agent's directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flux_vm import FluxVM, OPCODES, OPCODE_NAMES, VMState
from flux_assembler import (
    FluxAssembler,
    FluxDisassembler,
    assemble,
    disassemble,
    decode_i16,
    encode_i16,
    encode_u8,
    instruction_size,
    parse_register,
)
from flux_interpreter import FluxInterpreter, InterpretResult, Vocabulary, VocabEntry
from lcar_bridge import (
    LCARBridge,
    LCAROp,
    LCARTask,
    OpcodeTranslator,
)


# ─── Test Infrastructure ─────────────────────────────────────────────────────

_passed = 0
_failed = 0


def _report(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  \u2713 PASS: {name}")
    else:
        _failed += 1
        print(f"  \u2717 FAIL: {name}  —  {detail}")


# ─── 1. VM Execution Tests ──────────────────────────────────────────────────

def test_vm_iadd() -> None:
    """Test 1: IADD — 3 + 4 = 7"""
    bc = assemble("MOVI R0, 3\nMOVI R1, 4\nIADD R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("IADD: 3 + 4 = 7", vm.result(0) == 7, f"got {vm.result(0)}")


def test_vm_imul() -> None:
    """Test 2: IMUL — 5 * 6 = 30"""
    bc = assemble("MOVI R0, 5\nMOVI R1, 6\nIMUL R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("IMUL: 5 * 6 = 30", vm.result(0) == 30, f"got {vm.result(0)}")


def test_vm_idiv() -> None:
    """Test 3: IDIV — 20 / 4 = 5"""
    bc = assemble("MOVI R0, 20\nMOVI R1, 4\nIDIV R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("IDIV: 20 / 4 = 5", vm.result(0) == 5, f"got {vm.result(0)}")


def test_vm_loop_label() -> None:
    """Test 4: Loop with label — 3 + 2 + 1 = 6"""
    bc = assemble("MOVI R0, 3\nMOVI R1, 0\nloop: IADD R1, R1, R0\nDEC R0\nJNZ R0, loop\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("Loop: 3+2+1 = 6", vm.result(1) == 6, f"got {vm.result(1)}")


def test_vm_isub() -> None:
    """Test 11: ISUB — 10 - 3 = 7"""
    bc = assemble("MOVI R0, 10\nMOVI R1, 3\nISUB R0, R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("ISUB: 10 - 3 = 7", vm.result(0) == 7, f"got {vm.result(0)}")


def test_vm_inc() -> None:
    """Test 12: INC — R0 = 5, INC → 6"""
    bc = assemble("MOVI R0, 5\nINC R0\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("INC: 5 → 6", vm.result(0) == 6, f"got {vm.result(0)}")


def test_vm_mov() -> None:
    """Test 13: MOV — copy R1=99 into R0"""
    bc = assemble("MOVI R1, 99\nMOV R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("MOV: R1=99 → R0=99", vm.result(0) == 99, f"got {vm.result(0)}")


def test_vm_cmp_lt() -> None:
    """Test 14: CMP — 3 < 5 → R13 = -1"""
    bc = assemble("MOVI R0, 3\nMOVI R1, 5\nCMP R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("CMP: 3 < 5 → R13=-1", vm.result(13) == -1, f"got {vm.result(13)}")


def test_vm_cmp_eq() -> None:
    """Test 15: CMP — 7 == 7 → R13 = 0"""
    bc = assemble("MOVI R0, 7\nMOVI R1, 7\nCMP R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("CMP: 7 == 7 → R13=0", vm.result(13) == 0, f"got {vm.result(13)}")


def test_vm_cmp_gt() -> None:
    """Test 16: CMP — 9 > 4 → R13 = 1"""
    bc = assemble("MOVI R0, 9\nMOVI R1, 4\nCMP R0, R1\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("CMP: 9 > 4 → R13=1", vm.result(13) == 1, f"got {vm.result(13)}")


def test_vm_jz_jump() -> None:
    """Test 17: JZ — R0=0 should jump, R1 stays 1"""
    bc = assemble("MOVI R0, 0\nMOVI R1, 1\nJZ R0, skip\nMOVI R1, 2\nskip: HALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("JZ: R0=0 jumps, R1=1", vm.result(1) == 1, f"got {vm.result(1)}")


def test_vm_jz_no_jump() -> None:
    """Test 18: JZ — R0=5 should not jump, R1 becomes 2"""
    bc = assemble("MOVI R0, 5\nMOVI R1, 1\nJZ R0, 1\nMOVI R1, 2\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("JZ: R0=5 no jump, R1=2", vm.result(1) == 2, f"got {vm.result(1)}")


def test_vm_jmp() -> None:
    """Test 19: JMP — skip over instruction"""
    bc = assemble("MOVI R0, 1\nJMP skip\nMOVI R0, 2\nskip: HALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("JMP: skip, R0=1", vm.result(0) == 1, f"got {vm.result(0)}")


def test_vm_jmp_loop() -> None:
    """Test 20: JMP with label — count 3 iterations"""
    bc = assemble("MOVI R0, 3\nMOVI R1, 0\nloop: INC R1\nDEC R0\nJNZ R0, loop\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("JMP loop: R1=3", vm.result(1) == 3, f"got {vm.result(1)}")


def test_vm_push_pop() -> None:
    """Test 21: PUSH/POP — push 42, pop into R2"""
    bc = assemble("MOVI R0, 42\nPUSH R0\nMOVI R1, 0\nPOP R2\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("PUSH/POP: R2=42", vm.result(2) == 42, f"got {vm.result(2)}")


def test_vm_push_pop_lifo() -> None:
    """Test 22: PUSH/POP LIFO order — push 1,2 pop → 2,1"""
    bc = assemble("MOVI R0, 1\nMOVI R1, 2\nPUSH R0\nPUSH R1\nPOP R2\nPOP R3\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    ok = vm.result(2) == 2 and vm.result(3) == 1
    _report("PUSH/POP LIFO: R2=2, R3=1", ok, f"got R2={vm.result(2)}, R3={vm.result(3)}")


def test_vm_pop_empty() -> None:
    """Test 23: POP on empty stack → error"""
    bc = assemble("POP R0\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    _report("POP empty → Stack underflow",
            vm.error is not None and "Stack underflow" in vm.error,
            f"got error={vm.error}")


# ─── 2. Assembler / Disassembler Tests ───────────────────────────────────────

def test_disasm_basic() -> None:
    """Test 5: Disassembler — MOVI R0, 42; HALT"""
    bc = assemble("MOVI R0, 42\nHALT")
    text = disassemble(bc)
    _report("Disasm: MOVI R0, 42",
            "MOVI R0, 42" in text and "HALT" in text,
            f"got: {text}")


def test_disasm_isub() -> None:
    """Test 30: Disassemble ISUB"""
    bc = assemble("MOVI R0, 10\nMOVI R1, 3\nISUB R0, R0, R1\nHALT")
    text = disassemble(bc)
    _report("Disasm ISUB", "ISUB" in text, f"got: {text}")


def test_disasm_mov() -> None:
    """Test 31: Disassemble MOV"""
    bc = assemble("MOVI R1, 7\nMOV R0, R1\nHALT")
    text = disassemble(bc)
    _report("Disasm MOV R0, R1", "MOV R0, R1" in text, f"got: {text}")


def test_disasm_push_pop() -> None:
    """Test 32: Disassemble PUSH/POP"""
    bc = assemble("MOVI R0, 5\nPUSH R0\nPOP R1\nHALT")
    text = disassemble(bc)
    _report("Disasm PUSH/POP",
            "PUSH R0" in text and "POP R1" in text,
            f"got: {text}")


def test_asm_disasm_roundtrip() -> None:
    """Test: Assemble then disassemble produces consistent output."""
    src = "MOVI R0, 10\nMOVI R1, 20\nIADD R2, R0, R1\nHALT"
    bc = assemble(src)
    text = disassemble(bc)
    _report("Round-trip: asm→disasm",
            "MOVI R0, 10" in text and "IADD R2, R0, R1" in text and "HALT" in text,
            f"got: {text}")


def test_encode_decode_i16() -> None:
    """Test: encode_i16 / decode_i16 round-trip."""
    for val in [0, 1, -1, 127, -128, 32767, -32768, 42, -999]:
        encoded = encode_i16(val)
        decoded, _ = decode_i16(encoded)
        _report(f"i16 round-trip: {val}", decoded == val, f"got {decoded}")


def test_instruction_size() -> None:
    """Test: instruction_size returns correct byte counts."""
    _report("instruction_size HALT", instruction_size("HALT") == 1)
    _report("instruction_size INC", instruction_size("INC") == 2)
    _report("instruction_size IADD", instruction_size("IADD") == 4)
    _report("instruction_size MOV", instruction_size("MOV") == 3)
    _report("instruction_size MOVI", instruction_size("MOVI") == 4)
    _report("instruction_size JMP", instruction_size("JMP") == 3)


def test_parse_register() -> None:
    """Test: parse_register validates R0-R15."""
    _report("parse_register R0", parse_register("R0") == 0)
    _report("parse_register R15", parse_register("R15") == 15)
    ok = False
    try:
        parse_register("R16")
    except ValueError:
        ok = True
    _report("parse_register R16 → ValueError", ok)


# ─── 3. NLP Interpreter Tests ────────────────────────────────────────────────

def test_interp_add() -> None:
    """Test 6: Vocabulary — compute 7 + 5 = 12"""
    interp = FluxInterpreter()
    r = interp.run("compute 7 + 5")
    _report("NL: compute 7 + 5 = 12", r.value == 12, f"got {r.value}")


def test_interp_mul() -> None:
    """Test 7: Vocabulary — compute 8 * 9 = 72"""
    interp = FluxInterpreter()
    r = interp.run("compute 8 * 9")
    _report("NL: compute 8 * 9 = 72", r.value == 72, f"got {r.value}")


def test_interp_factorial() -> None:
    """Test 8: Vocabulary — factorial of 5 = 120"""
    interp = FluxInterpreter()
    r = interp.run("factorial of 5")
    _report("NL: factorial of 5 = 120", r.value == 120, f"got {r.value}")


def test_interp_double() -> None:
    """Test 9: Vocabulary — double 21 = 42"""
    interp = FluxInterpreter()
    r = interp.run("double 21")
    _report("NL: double 21 = 42", r.value == 42, f"got {r.value}")


def test_interp_square() -> None:
    """Test 10: Vocabulary — square 7 = 49"""
    interp = FluxInterpreter()
    r = interp.run("square 7")
    _report("NL: square 7 = 49", r.value == 49, f"got {r.value}")


def test_interp_sub() -> None:
    """Test 24: Vocabulary — compute 20 - 8 = 12"""
    interp = FluxInterpreter()
    r = interp.run("compute 20 - 8")
    _report("NL: compute 20 - 8 = 12", r.value == 12, f"got {r.value}")


def test_interp_div() -> None:
    """Test 25: Vocabulary — compute 100 / 4 = 25"""
    interp = FluxInterpreter()
    r = interp.run("compute 100 / 4")
    _report("NL: compute 100 / 4 = 25", r.value == 25, f"got {r.value}")


def test_interp_fibonacci() -> None:
    """Test 26: Vocabulary — fibonacci of 7 = 13"""
    interp = FluxInterpreter()
    r = interp.run("fibonacci of 7")
    _report("NL: fibonacci of 7 = 13", r.value == 13, f"got {r.value}")


def test_interp_sum_range() -> None:
    """Test 27: Vocabulary — sum 1 to 100 = 5050"""
    interp = FluxInterpreter()
    r = interp.run("sum 1 to 100")
    _report("NL: sum 1 to 100 = 5050", r.value == 5050, f"got {r.value}")


def test_interp_power() -> None:
    """Test 28: Vocabulary — power of 2 to 10 = 1024"""
    interp = FluxInterpreter()
    r = interp.run("power of 2 to 10")
    _report("NL: power of 2 to 10 = 1024", r.value == 1024, f"got {r.value}")


def test_interp_hello() -> None:
    """Test 29: Vocabulary — hello = 42"""
    interp = FluxInterpreter()
    r = interp.run("hello")
    _report("NL: hello = 42", r.value == 42, f"got {r.value}")


def test_interp_no_match() -> None:
    """Test: Interpreter returns None for unmatched input."""
    interp = FluxInterpreter()
    r = interp.run("xyzzy nothing here")
    _report("NL: no match → None", r.value is None)


def test_interp_translate() -> None:
    """Test: Interpreter.translate returns assembly without executing."""
    interp = FluxInterpreter()
    asm = interp.translate("double 7")
    _report("NL: translate double 7",
            asm is not None and "MOVI R0, 7" in asm and "IADD R0, R0, R0" in asm,
            f"got: {asm}")


def test_interp_compile() -> None:
    """Test: Interpreter.compile returns bytecode."""
    interp = FluxInterpreter()
    bc = interp.compile("double 7")
    _report("NL: compile double 7 → bytes", bc is not None and isinstance(bc, bytes))


# ─── 4. LCAR Bridge Tests ───────────────────────────────────────────────────

def test_lcar_opcode_translation() -> None:
    """Test: FLUX opcodes map to correct LCAR operations."""
    _report("LCAR: HALT → SHUTDOWN",
            OpcodeTranslator.flux_to_lcar(0x80) == LCAROp.SHUTDOWN)
    _report("LCAR: IADD → COMPUTE",
            OpcodeTranslator.flux_to_lcar(0x08) == LCAROp.COMPUTE)
    _report("LCAR: CMP → AGGREGATE",
            OpcodeTranslator.flux_to_lcar(0x0D) == LCAROp.AGGREGATE)
    _report("LCAR: PUSH → TRANSFER",
            OpcodeTranslator.flux_to_lcar(0x01) == LCAROp.TRANSFER)
    _report("LCAR: JZ → BARRIER",
            OpcodeTranslator.flux_to_lcar(0x07) == LCAROp.BARRIER)


def test_lcar_reverse_translation() -> None:
    """Test: LCAR ops map back to FLUX opcodes."""
    flux_ops = OpcodeTranslator.lcar_to_flux(LCAROp.COMPUTE)
    _report("LCAR: COMPUTE → [IADD,ISUB,IMUL,IDIV,INC,DEC]",
            0x08 in flux_ops and 0x0A in flux_ops)


def test_lcar_translate_bytecode() -> None:
    """Test: Full bytecode program translates to LCAR operations."""
    bc = assemble("MOVI R0, 42\nMOVI R1, 8\nIADD R0, R0, R1\nHALT")
    result = OpcodeTranslator.translate_bytecode(bc)
    names = [r["lcar_name"] for r in result]
    _report("LCAR: translate_bytecode has 4 ops", len(result) == 4)
    _report("LCAR: first op is LOAD", names[0] == "LOAD")
    _report("LCAR: has COMPUTE op", "COMPUTE" in names)
    _report("LCAR: last op is SHUTDOWN", names[-1] == "SHUTDOWN")


def test_lcar_task_create() -> None:
    """Test: LCARBridge creates task from assembly."""
    bridge = LCARBridge(agent_id="test-agent")
    task = bridge.create_task(
        "add_task", "Compute sum",
        assembly="MOVI R0, 3\nMOVI R1, 4\nIADD R0, R0, R1\nHALT",
    )
    _report("LCAR: task created", task.task_id.startswith("test-agent-task-"))
    _report("LCAR: task has bytecode", task.bytecode is not None and len(task.bytecode) > 0)
    _report("LCAR: task operation", task.operation == LCAROp.COMPUTE)


def test_lcar_task_json_roundtrip() -> None:
    """Test: LCARTask serializes/deserializes to JSON."""
    bc = assemble("MOVI R0, 42\nHALT")
    task = LCARTask(
        task_id="test-001",
        name="test task",
        operation=LCAROp.COMPUTE,
        bytecode=bc,
    )
    json_str = task.to_json()
    restored = LCARTask.from_json(json_str)
    _report("LCAR: JSON round-trip", restored.bytecode == bc and restored.task_id == "test-001")


def test_lcar_execute_task() -> None:
    """Test: LCARBridge executes a task locally."""
    bridge = LCARBridge()
    task = bridge.create_task(
        "compute_42", "Compute 42",
        assembly="MOVI R0, 42\nHALT",
    )
    result = bridge.execute_task(task)
    _report("LCAR: execute task → R0=42", result["result"] == 42, f"got {result}")
    _report("LCAR: execute status SUCCESS", result["status"] == "SUCCESS")


def test_lcar_status_report() -> None:
    """Test: LCARBridge generates status report."""
    bridge = LCARBridge(agent_id="status-test")
    bc = assemble("MOVI R0, 42\nHALT")
    vm = FluxVM(bc)
    vm.execute()
    report = bridge.generate_status_report(vm)
    _report("LCAR: status report agent_id", report["agent_id"] == "status-test")
    _report("LCAR: status report halted", report["halted"] is True)


def test_lcar_schedule_command() -> None:
    """Test: LCARBridge generates scheduling command."""
    bridge = LCARBridge()
    task = bridge.create_task("demo", "compute task", assembly="MOVI R0, 1\nHALT")
    cmd = bridge.schedule_task(task)
    _report("LCAR: schedule command", cmd["command"] == "SCHEDULE")
    _report("LCAR: schedule has analysis", "instruction_count" in cmd["analysis"])


# ─── 5. CLI Parsing Tests ───────────────────────────────────────────────────

def test_cli_run() -> None:
    """Test: CLI parses 'run' subcommand."""
    from cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["run", "program.flux"])
    _report("CLI: run command", args.command == "run" and args.program == "program.flux")


def test_cli_assemble() -> None:
    """Test: CLI parses 'assemble' subcommand."""
    from cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["assemble", "source.asm", "-o", "out.flux"])
    _report("CLI: assemble command",
            args.command == "assemble" and args.output == "out.flux")


def test_cli_interpret() -> None:
    """Test: CLI parses 'interpret' subcommand."""
    from cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["interpret", "compute 7 + 5"])
    _report("CLI: interpret command",
            args.command == "interpret" and args.query == "compute 7 + 5")


def test_cli_debug() -> None:
    """Test: CLI parses 'debug' subcommand."""
    from cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["debug", "program.flux", "--max-steps", "10"])
    _report("CLI: debug command",
            args.command == "debug" and args.max_steps == 10)


def test_cli_status() -> None:
    """Test: CLI parses 'status' subcommand."""
    from cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["status"])
    _report("CLI: status command", args.command == "status")


def test_cli_lcar_schedule() -> None:
    """Test: CLI parses 'lcar-schedule' subcommand."""
    from cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["lcar-schedule", "task.json", "--execute"])
    _report("CLI: lcar-schedule command",
            args.command == "lcar-schedule" and args.execute is True)


# ─── 6. Workshop Example Tests ───────────────────────────────────────────────

def test_workshop_addition() -> None:
    """Test: Workshop addition.asm assembles and runs correctly."""
    workshop_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workshop")
    path = os.path.join(workshop_dir, "addition.asm")
    if not os.path.exists(path):
        _report("Workshop: addition.asm", False, "file not found")
        return
    asm = FluxAssembler()
    bc = asm.assemble(open(path).read())
    vm = FluxVM(bc)
    vm.execute()
    _report("Workshop: addition.asm R0=7", vm.result(0) == 7, f"got {vm.result(0)}")


def test_workshop_factorial() -> None:
    """Test: Workshop factorial.asm computes 5! = 120."""
    workshop_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workshop")
    path = os.path.join(workshop_dir, "factorial.asm")
    if not os.path.exists(path):
        _report("Workshop: factorial.asm", False, "file not found")
        return
    asm = FluxAssembler()
    bc = asm.assemble(open(path).read())
    vm = FluxVM(bc)
    vm.execute()
    _report("Workshop: factorial.asm R1=120", vm.result(1) == 120, f"got {vm.result(1)}")


def test_workshop_fibonacci() -> None:
    """Test: Workshop fibonacci.asm computes F(7) = 13."""
    workshop_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workshop")
    path = os.path.join(workshop_dir, "fibonacci.asm")
    if not os.path.exists(path):
        _report("Workshop: fibonacci.asm", False, "file not found")
        return
    asm = FluxAssembler()
    bc = asm.assemble(open(path).read())
    vm = FluxVM(bc)
    vm.execute()
    _report("Workshop: fibonacci.asm R2=13", vm.result(2) == 13, f"got {vm.result(2)}")


# ─── Test Runner ─────────────────────────────────────────────────────────────

def run_all_tests() -> bool:
    """Execute all tests and report results."""
    global _passed, _failed

    print("=" * 60)
    print("  FLUX VM Agent — Test Suite")
    print("=" * 60)

    # 1. VM execution (original tests 1-4, 11-23)
    print("\n── VM Execution ──")
    test_vm_iadd()          # 1
    test_vm_imul()          # 2
    test_vm_idiv()          # 3
    test_vm_loop_label()    # 4
    test_vm_isub()          # 11
    test_vm_inc()           # 12
    test_vm_mov()           # 13
    test_vm_cmp_lt()        # 14
    test_vm_cmp_eq()        # 15
    test_vm_cmp_gt()        # 16
    test_vm_jz_jump()       # 17
    test_vm_jz_no_jump()    # 18
    test_vm_jmp()           # 19
    test_vm_jmp_loop()      # 20
    test_vm_push_pop()      # 21
    test_vm_push_pop_lifo() # 22
    test_vm_pop_empty()     # 23

    # 2. Assembler / Disassembler (original tests 5, 30-32)
    print("\n── Assembler / Disassembler ──")
    test_disasm_basic()     # 5
    test_disasm_isub()      # 30
    test_disasm_mov()       # 31
    test_disasm_push_pop()  # 32
    test_asm_disasm_roundtrip()
    test_encode_decode_i16()
    test_instruction_size()
    test_parse_register()

    # 3. NLP Interpreter (original tests 6-10, 24-29)
    print("\n── NLP Interpreter ──")
    test_interp_add()       # 6
    test_interp_mul()       # 7
    test_interp_factorial() # 8
    test_interp_double()    # 9
    test_interp_square()    # 10
    test_interp_sub()       # 24
    test_interp_div()       # 25
    test_interp_fibonacci() # 26
    test_interp_sum_range() # 27
    test_interp_power()     # 28
    test_interp_hello()     # 29
    test_interp_no_match()
    test_interp_translate()
    test_interp_compile()

    # 4. LCAR Bridge
    print("\n── LCAR Bridge ──")
    test_lcar_opcode_translation()
    test_lcar_reverse_translation()
    test_lcar_translate_bytecode()
    test_lcar_task_create()
    test_lcar_task_json_roundtrip()
    test_lcar_execute_task()
    test_lcar_status_report()
    test_lcar_schedule_command()

    # 5. CLI Parsing
    print("\n── CLI Parsing ──")
    test_cli_run()
    test_cli_assemble()
    test_cli_interpret()
    test_cli_debug()
    test_cli_status()
    test_cli_lcar_schedule()

    # 6. Workshop examples
    print("\n── Workshop Examples ──")
    test_workshop_addition()
    test_workshop_factorial()
    test_workshop_fibonacci()

    # Summary
    print("\n" + "=" * 60)
    total = _passed + _failed
    print(f"  Results: {_passed} passed, {_failed} failed, {total} total")
    print("=" * 60)

    return _failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
