#!/usr/bin/env python3
"""
FLUX VM Agent — CLI interface.

Subcommands:
  run <program.flux>              Execute a compiled FLUX bytecode program
  assemble <source.asm> -o <out>  Assemble source to bytecode
  disassemble <program.flux>      Show disassembly of a bytecode file
  interpret "<natural language>"   Translate NL to FLUX and execute
  debug <program.flux>            Step-through debugger
  lcar-schedule <task.json>       Schedule a task via LCAR bridge
  onboard                         Set up the agent (create config)
  status                          Show agent status

Usage:
  python cli.py run program.flux
  python cli.py assemble source.asm -o program.flux
  python cli.py disassemble program.flux
  python cli.py interpret "compute 7 + 5"
  python cli.py debug program.flux
  python cli.py lcar-schedule task.json
  python cli.py onboard
  python cli.py status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

# Ensure the agent's directory is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux_vm import FluxVM, OPCODES, OPCODE_NAMES
from flux_assembler import FluxAssembler, FluxDisassembler, disassemble
from flux_interpreter import FluxInterpreter, InterpretResult, Vocabulary
from lcar_bridge import LCARBridge, LCARTask, OpcodeTranslator

# ─── Constants ────────────────────────────────────────────────────────────────

AGENT_NAME = "flux-vm-agent"
VERSION = "1.0.0"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".flux-vm-agent.json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_bytecode(path: str) -> bytes:
    """Load raw bytecode from a file."""
    with open(path, "rb") as f:
        return f.read()


def _load_text(path: str) -> str:
    """Load text content from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_bytecode(path: str, data: bytes) -> None:
    """Write bytecode to a file."""
    with open(path, "wb") as f:
        f.write(data)


def _print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─── Subcommand Handlers ─────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Execute a compiled FLUX bytecode program.

    Args:
        args: Parsed CLI arguments with ``program`` attribute.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        bytecode = _load_bytecode(args.program)
    except FileNotFoundError:
        print(f"Error: File not found: {args.program}", file=sys.stderr)
        return 1

    vm = FluxVM(bytecode)
    vm.execute()

    _print_header("FLUX VM — Execution Result")

    if vm.is_success():
        print(f"  Status:   HALTED (clean)")
        print(f"  Cycles:   {vm.cycles}")
        print(f"  R0:       {vm.reg(0)}")
        if args.verbose:
            print(f"\n{vm.dump_registers()}")
            if vm.stack_depth() > 0:
                print(f"\n  Stack:")
                print(vm.dump_stack())
    else:
        print(f"  Status:   ERROR")
        print(f"  Error:    {vm.error}")
        print(f"  Cycles:   {vm.cycles}")
        print(f"  PC:       {vm.pc}")
        return 1

    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    """Assemble a FLUX source file to bytecode.

    Args:
        args: Parsed CLI arguments with ``source`` and ``output`` attributes.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        text = _load_text(args.source)
    except FileNotFoundError:
        print(f"Error: File not found: {args.source}", file=sys.stderr)
        return 1

    assembler = FluxAssembler()
    try:
        bytecode = assembler.assemble(text)
    except ValueError as e:
        print(f"Assembly error: {e}", file=sys.stderr)
        return 1

    output_path = args.output or args.source.replace(".asm", ".flux")
    _write_bytecode(output_path, bytecode)

    _print_header("FLUX Assembler")
    print(f"  Source:  {args.source}")
    print(f"  Output:  {output_path}")
    print(f"  Size:    {len(bytecode)} bytes")
    print(f"  Labels:  {len(assembler.labels)}")

    if args.verbose:
        dis = FluxDisassembler()
        print(f"\n  Disassembly:")
        for line in dis.disassemble(bytecode).split("\n"):
            print(f"    {line}")

    return 0


def cmd_disassemble(args: argparse.Namespace) -> int:
    """Disassemble a FLUX bytecode file.

    Args:
        args: Parsed CLI arguments with ``program`` attribute.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        bytecode = _load_bytecode(args.program)
    except FileNotFoundError:
        print(f"Error: File not found: {args.program}", file=sys.stderr)
        return 1

    dis = FluxDisassembler()

    _print_header("FLUX Disassembler")
    print(f"  File:  {args.program}")
    print(f"  Size:  {len(bytecode)} bytes\n")

    if args.bytes:
        print(dis.disassemble_with_bytes(bytecode))
    else:
        print(dis.disassemble(bytecode))

    return 0


def cmd_interpret(args: argparse.Namespace) -> int:
    """Translate natural language to FLUX and execute.

    Args:
        args: Parsed CLI arguments with ``query`` attribute.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    interp = FluxInterpreter()
    result: InterpretResult = interp.run(args.query)

    _print_header("FLUX Interpreter")

    if result.value is not None:
        print(f"  Result:   {result.value}")
        print(f"  Message:  {result.message}")
        if result.pattern:
            print(f"  Pattern:  {result.pattern}")
    else:
        print(f"  Result:   (none)")
        print(f"  Message:  {result.message}")

    if args.verbose:
        asm = interp.translate(args.query)
        if asm:
            print(f"\n  Generated assembly:")
            for line in asm.split("\n"):
                print(f"    {line}")

    return 0 if result.value is not None else 1


def cmd_debug(args: argparse.Namespace) -> int:
    """Step-through debugger for FLUX bytecode.

    Executes one instruction at a time, showing VM state after each step.

    Args:
        args: Parsed CLI arguments with ``program`` attribute.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        bytecode = _load_bytecode(args.program)
    except FileNotFoundError:
        print(f"Error: File not found: {args.program}", file=sys.stderr)
        return 1

    dis = FluxDisassembler()
    _print_header("FLUX Debugger")
    print(f"  File:     {args.program}")
    print(f"  Size:     {len(bytecode)} bytes")
    print(f"  Disasm:\n")
    for line in dis.disassemble(bytecode).split("\n"):
        print(f"    {line}")

    print(f"\n  Stepping through execution...\n")

    vm = FluxVM(bytecode)
    vm.enable_trace(True)
    vm.execute()
    trace = vm.get_trace()

    step = 0
    for state in trace:
        step += 1
        non_zero = [(i, state.gp[i]) for i in range(16) if state.gp[i] != 0]
        regs = ", ".join(f"R{i}={v}" for i, v in non_zero) if non_zero else "(all zero)"
        stack_info = f"stack=[{len(state.stack)}]" if state.stack else "stack=[]"
        print(f"  Step {step:3d} | PC={state.pc:04X} | {regs} | {stack_info}")

        if args.max_steps and step >= args.max_steps:
            print(f"\n  (stopped after {args.max_steps} steps)")
            break

    _print_header("Final State")
    print(f"  Halted:   {vm.halted}")
    print(f"  Cycles:   {vm.cycles}")
    print(f"  Error:    {vm.error or '(none)'}")
    print(f"  R0:       {vm.reg(0)}")

    return 0 if vm.is_success() else 1


def cmd_lcar_schedule(args: argparse.Namespace) -> int:
    """Schedule a task via the LCAR bridge.

    Args:
        args: Parsed CLI arguments with ``task_file`` attribute.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        text = _load_text(args.task_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.task_file}", file=sys.stderr)
        return 1

    try:
        task = LCARTask.from_json(text)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Error parsing task JSON: {e}", file=sys.stderr)
        return 1

    bridge = LCARBridge()

    _print_header("LCAR Task Scheduler")
    print(f"  Task ID:    {task.task_id}")
    print(f"  Name:       {task.name}")
    print(f"  Operation:  {task.operation.name}")
    print(f"  Priority:   {task.priority}")
    print(f"  Target:     {task.agent_id}")

    # Show scheduling command
    schedule_cmd = bridge.schedule_task(task)
    print(f"\n  Schedule Command:")
    print(f"    {json.dumps(schedule_cmd, indent=4)}")

    # Execute if requested
    if args.execute:
        print(f"\n  Executing task locally...")
        result = bridge.execute_task(task)
        print(f"    Status:  {result['status']}")
        if result.get("result") is not None:
            print(f"    Result:  {result['result']}")
        print(f"    Cycles:  {result['cycles']}")
        if result.get("error"):
            print(f"    Error:   {result['error']}")

    return 0


def cmd_onboard(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Set up the agent — create configuration file.

    Returns:
        Exit code (0 = success).
    """
    _print_header("FLUX VM Agent — Onboarding")

    config = {
        "agent_name": AGENT_NAME,
        "version": VERSION,
        "registers": 16,
        "opcodes": len(OPCODES),
        "max_cycles": FluxVM.MAX_CYCLES,
        "supported_commands": [
            "run", "assemble", "disassemble",
            "interpret", "debug", "lcar-schedule",
            "onboard", "status",
        ],
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"  Agent:    {AGENT_NAME} v{VERSION}")
    print(f"  Config:   {CONFIG_FILE}")
    print(f"  Registers: {config['registers']}")
    print(f"  Opcodes:   {config['opcodes']}")
    print(f"  Max cycles: {config['max_cycles']:,}")
    print(f"\n  Ready!")

    return 0


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    """Show agent status.

    Returns:
        Exit code (0 = success).
    """
    _print_header("FLUX VM Agent — Status")

    # Check config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"  Config:    Present ({CONFIG_FILE})")
        print(f"  Agent:     {config.get('agent_name', 'unknown')}")
        print(f"  Version:   {config.get('version', 'unknown')}")
    else:
        print(f"  Config:    Not found — run 'onboard' to initialize")
        print(f"  Agent:     {AGENT_NAME} (unconfigured)")

    print(f"\n  Capabilities:")
    print(f"    Registers:   16 (R0-R15)")
    print(f"    Opcodes:     {len(OPCODES)}")
    print(f"    Stack:       LIFO, unbounded")
    print(f"    Max cycles:  {FluxVM.MAX_CYCLES:,}")

    print(f"\n  Opcode table:")
    for name, code in sorted(OPCODES.items(), key=lambda x: x[1]):
        print(f"    {name:8s}  0x{code:02X}")

    vocab = Vocabulary.builtin()
    print(f"\n  NLP patterns: {len(vocab.list_patterns())}")
    for pattern in vocab.list_patterns():
        print(f"    \"{pattern}\"")

    return 0


# ─── Argument Parser ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="flux-vm-agent",
        description="FLUX VM Agent — standalone bytecode VM, assembler, NLP interpreter, LCAR bridge",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p_run = sub.add_parser("run", help="Execute a FLUX bytecode program")
    p_run.add_argument("program", help="Path to .flux bytecode file")
    p_run.add_argument("-v", "--verbose", action="store_true", help="Show all registers and stack")

    # assemble
    p_asm = sub.add_parser("assemble", help="Assemble source to bytecode")
    p_asm.add_argument("source", help="Path to .asm assembly source")
    p_asm.add_argument("-o", "--output", help="Output .flux file (default: source.flux)")
    p_asm.add_argument("-v", "--verbose", action="store_true", help="Show disassembly")

    # disassemble
    p_dis = sub.add_parser("disassemble", help="Show disassembly of a bytecode file")
    p_dis.add_argument("program", help="Path to .flux bytecode file")
    p_dis.add_argument("--bytes", action="store_true", help="Show raw hex bytes")

    # interpret
    p_int = sub.add_parser("interpret", help="Translate natural language to FLUX and execute")
    p_int.add_argument("query", help="Natural language query (e.g. 'compute 7 + 5')")
    p_int.add_argument("-v", "--verbose", action="store_true", help="Show generated assembly")

    # debug
    p_dbg = sub.add_parser("debug", help="Step-through debugger")
    p_dbg.add_argument("program", help="Path to .flux bytecode file")
    p_dbg.add_argument("--max-steps", type=int, default=0, help="Max steps to show (0 = all)")

    # lcar-schedule
    p_lcar = sub.add_parser("lcar-schedule", help="Schedule a task via LCAR bridge")
    p_lcar.add_argument("task_file", help="Path to task JSON file")
    p_lcar.add_argument("--execute", action="store_true", help="Execute the task locally")

    # onboard
    sub.add_parser("onboard", help="Set up the agent")

    # status
    sub.add_parser("status", help="Show agent status")

    return parser


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "run": cmd_run,
        "assemble": cmd_assemble,
        "disassemble": cmd_disassemble,
        "interpret": cmd_interpret,
        "debug": cmd_debug,
        "lcar-schedule": cmd_lcar_schedule,
        "onboard": cmd_onboard,
        "status": cmd_status,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
