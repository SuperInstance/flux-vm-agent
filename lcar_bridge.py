"""
LCAR Bridge — integration layer between the FLUX VM and the LCAR scheduler.

The LCAR (Logical Compute Allocation & Routing) bridge translates FLUX VM
concepts into fleet-scheduling primitives. It provides:

  - Opcode mapping: FLUX opcodes ↔ LCAR task operation codes.
  - Task scheduling: natural-language or assembly commands → fleet task specs.
  - Agent status reporting: VM state → LCAR status payloads.

This module has **no external dependencies** — it operates on pure data
structures and uses the FLUX VM/Assembler for execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, List, Optional

from flux_vm import FluxVM, OPCODES, VMState
from flux_assembler import FluxAssembler, FluxDisassembler


# ─── LCAR Task Operation Codes ───────────────────────────────────────────────

class LCAROp(IntEnum):
    """LCAR fleet-level operation codes, mapped from FLUX opcodes.

    These represent higher-level fleet operations that correspond to
    FLUX VM instruction categories.
    """

    # Scheduling
    SCHEDULE = 0xA0       # Schedule a new task on the fleet
    CANCEL = 0xA1         # Cancel a running task
    RESCHEDULE = 0xA2     # Reschedule a task to a different agent

    # Compute
    COMPUTE = 0xB0        # Execute a compute workload
    AGGREGATE = 0xB1      # Aggregate results from sub-tasks
    REDUCE = 0xB2         # Reduce (map-reduce) across fleet

    # Data
    LOAD = 0xC0           # Load data into VM memory
    STORE = 0xC1          # Store data from VM memory
    TRANSFER = 0xC2       # Transfer data between agents

    # Control
    BARRIER = 0xD0        # Synchronization barrier
    BROADCAST = 0xD1      # Broadcast message to all agents
    GATHER = 0xD2         # Gather results from all agents

    # Status
    STATUS = 0xE0         # Query agent status
    HEARTBEAT = 0xE1      # Heartbeat / liveness check
    SHUTDOWN = 0xE2       # Graceful shutdown


# ─── Opcode Translation ──────────────────────────────────────────────────────

class OpcodeTranslator:
    """Bidirectional translator between FLUX VM opcodes and LCAR operations.

    Maps FLUX instruction categories to fleet-level operations:

    +----------------+------------------+----------------------------------+
    | FLUX Category  | LCAR Operation   | Notes                            |
    +================+==================+==================================+
    | MOVI, MOV      | LOAD             | Data movement → data loading     |
    | IADD, ISUB,..  | COMPUTE          | Arithmetic → fleet compute       |
    | CMP            | AGGREGATE        | Comparison → result aggregation  |
    | JZ, JNZ, JMP   | BARRIER          | Control flow → synchronization   |
    | PUSH, POP      | TRANSFER         | Stack ops → data transfer        |
    | HALT           | SHUTDOWN         | Halt → graceful shutdown         |
    +----------------+------------------+----------------------------------+
    """

    # FLUX opcode → LCAR operation mapping
    FLUX_TO_LCAR: Dict[int, LCAROp] = {
        0x2B: LCAROp.LOAD,      # MOVI
        0x2C: LCAROp.TRANSFER,  # MOV
        0x08: LCAROp.COMPUTE,   # IADD
        0x09: LCAROp.COMPUTE,   # ISUB
        0x0A: LCAROp.COMPUTE,   # IMUL
        0x0B: LCAROp.COMPUTE,   # IDIV
        0x0E: LCAROp.COMPUTE,   # INC
        0x0F: LCAROp.COMPUTE,   # DEC
        0x0D: LCAROp.AGGREGATE, # CMP
        0x07: LCAROp.BARRIER,   # JZ
        0x06: LCAROp.BARRIER,   # JNZ
        0x05: LCAROp.BARRIER,   # JMP
        0x01: LCAROp.TRANSFER,  # PUSH
        0x02: LCAROp.TRANSFER,  # POP
        0x80: LCAROp.SHUTDOWN,  # HALT
    }

    # LCAR operation → list of FLUX opcodes
    LCAR_TO_FLUX: Dict[LCAROp, List[int]] = {}
    for _flux_op, _lcar_op in FLUX_TO_LCAR.items():
        LCAR_TO_FLUX.setdefault(_lcar_op, []).append(_flux_op)

    @classmethod
    def flux_to_lcar(cls, opcode: int) -> Optional[LCAROp]:
        """Translate a FLUX opcode byte to its LCAR operation.

        Args:
            opcode: FLUX opcode byte value.

        Returns:
            The corresponding :class:`LCAROp`, or ``None`` if unmapped.
        """
        return cls.FLUX_TO_LCAR.get(opcode)

    @classmethod
    def lcar_to_flux(cls, op: LCAROp) -> List[int]:
        """Translate an LCAR operation to its FLUX opcodes.

        Args:
            op: LCAR operation code.

        Returns:
            List of FLUX opcode bytes that map to this operation.
        """
        return list(cls.LCAR_TO_FLUX.get(op, []))

    @classmethod
    def translate_bytecode(cls, bytecode: bytes) -> List[Dict[str, Any]]:
        """Translate an entire FLUX bytecode program into LCAR operations.

        Args:
            bytecode: Raw FLUX bytecode.

        Returns:
            List of dicts with ``offset``, ``opcode``, ``lcar_op``, and
            ``lcar_name`` keys.
        """
        dis = FluxDisassembler()
        lines = dis.disassemble(bytecode).split("\n")
        result: List[Dict[str, Any]] = []
        pc = 0

        for line in lines:
            op = bytecode[pc]
            lcar_op = cls.flux_to_lcar(op)
            result.append({
                "offset": pc,
                "opcode": f"0x{op:02X}",
                "instruction": line,
                "lcar_op": lcar_op.value if lcar_op else None,
                "lcar_name": lcar_op.name if lcar_op else "UNMAPPED",
            })
            # Advance PC past this instruction
            if op == 0x80:
                pc += 1
            elif op in (0x0F, 0x0E, 0x01, 0x02):
                pc += 2
            elif op in (0x08, 0x09, 0x0A, 0x0B):
                pc += 4
            elif op in (0x2C, 0x0D):
                pc += 3
            elif op in (0x2B, 0x06, 0x07):
                pc += 4
            elif op == 0x05:
                pc += 3
            else:
                pc += 1

        return result


# ─── Task Specification ──────────────────────────────────────────────────────

@dataclass
class LCARTask:
    """A task specification for the LCAR fleet scheduler.

    Attributes:
        task_id: Unique task identifier.
        name: Human-readable task name.
        operation: LCAR operation code.
        bytecode: FLUX bytecode to execute (optional).
        assembly: FLUX assembly source (optional, used if no bytecode).
        priority: Task priority (0 = highest).
        agent_id: Target agent ID (``"*"`` for any available).
        params: Additional task parameters.
    """

    task_id: str
    name: str
    operation: LCAROp
    bytecode: Optional[bytes] = None
    assembly: Optional[str] = None
    priority: int = 0
    agent_id: str = "*"
    params: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        # Convert bytes to hex string for JSON serialization
        if d.get("bytecode") is not None:
            d["bytecode"] = d["bytecode"].hex()
        d["operation"] = int(d["operation"])
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "LCARTask":
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        if d.get("bytecode") is not None:
            d["bytecode"] = bytes.fromhex(d["bytecode"])
        d["operation"] = LCAROp(d["operation"])
        return cls(**d)


# ─── LCAR Bridge ─────────────────────────────────────────────────────────────

class LCARBridge:
    """Bridge between the FLUX VM and the LCAR scheduler.

    Provides methods to:
      - Create fleet task specifications from FLUX programs.
      - Translate bytecode into LCAR scheduling commands.
      - Generate status reports from VM execution state.

    Example::

        bridge = LCARBridge()
        task = bridge.create_task("my_task", "Compute 42", assembly="MOVI R0, 42\\nHALT")
        print(task.to_json())
    """

    def __init__(self, agent_id: str = "flux-vm-agent") -> None:
        """Initialize the bridge.

        Args:
            agent_id: Identifier for this agent in the fleet.
        """
        self.agent_id: str = agent_id
        self.assembler: FluxAssembler = FluxAssembler()
        self.translator: OpcodeTranslator = OpcodeTranslator
        self._task_counter: int = 0

    def create_task(
        self,
        name: str,
        description: str,
        *,
        bytecode: Optional[bytes] = None,
        assembly: Optional[str] = None,
        priority: int = 0,
        target_agent: str = "*",
    ) -> LCARTask:
        """Create a new LCAR task specification.

        If *assembly* is provided and *bytecode* is not, the assembly
        will be compiled to bytecode automatically.

        Args:
            name: Human-readable task name.
            description: Task description (used to infer operation type).
            bytecode: Pre-compiled FLUX bytecode.
            assembly: FLUX assembly source (compiled if no bytecode given).
            priority: Task priority (0 = highest).
            target_agent: Target agent ID (``"*"`` for any).

        Returns:
            A :class:`LCARTask` ready for scheduling.
        """
        self._task_counter += 1
        task_id = f"{self.agent_id}-task-{self._task_counter:04d}"

        # Infer operation from description keywords
        operation = self._infer_operation(description)

        # Compile assembly if needed
        if bytecode is None and assembly is not None:
            bytecode = self.assembler.assemble(assembly)

        return LCARTask(
            task_id=task_id,
            name=name,
            operation=operation,
            bytecode=bytecode,
            assembly=assembly,
            priority=priority,
            agent_id=target_agent,
        )

    def schedule_task(self, task: LCARTask) -> Dict[str, Any]:
        """Generate a scheduling command for a task.

        Args:
            task: The task to schedule.

        Returns:
            A scheduling command dict with metadata and bytecode analysis.
        """
        # Analyze bytecode if present
        analysis: Dict[str, Any] = {}
        if task.bytecode:
            translated = self.translator.translate_bytecode(task.bytecode)
            op_counts: Dict[str, int] = {}
            for item in translated:
                name = item["lcar_name"]
                op_counts[name] = op_counts.get(name, 0) + 1
            analysis = {
                "instruction_count": len(translated),
                "lcar_operations": op_counts,
                "estimated_cycles": len(translated),
            }

        return {
            "command": "SCHEDULE",
            "task": {
                "task_id": task.task_id,
                "name": task.name,
                "operation": task.operation.name,
                "priority": task.priority,
                "target_agent": task.agent_id,
            },
            "analysis": analysis,
        }

    def execute_task(self, task: LCARTask) -> Dict[str, Any]:
        """Execute a task locally on the FLUX VM.

        Args:
            task: Task with bytecode or assembly to execute.

        Returns:
            Execution result dict with VM state and status.
        """
        if task.bytecode is None and task.assembly is not None:
            task.bytecode = self.assembler.assemble(task.assembly)
        if task.bytecode is None:
            return {
                "task_id": task.task_id,
                "status": "ERROR",
                "error": "No bytecode or assembly provided",
            }

        vm = FluxVM(task.bytecode)
        vm.execute()

        state: Dict[str, Any] = {
            "task_id": task.task_id,
            "status": "SUCCESS" if vm.is_success() else "ERROR",
            "result": vm.result(0),
            "cycles": vm.cycles,
            "halted": vm.halted,
        }
        if vm.error:
            state["error"] = vm.error
        state["registers"] = {f"R{i}": vm.gp[i] for i in range(16)}
        state["stack_depth"] = len(vm.stack)
        return state

    def generate_status_report(self, vm: FluxVM) -> Dict[str, Any]:
        """Generate a status report from VM state.

        Args:
            vm: The VM to report on.

        Returns:
            Status dict suitable for LCAR heartbeat/status messages.
        """
        return {
            "agent_id": self.agent_id,
            "status": "IDLE" if vm.halted else "RUNNING",
            "pc": vm.pc,
            "cycles": vm.cycles,
            "halted": vm.halted,
            "error": vm.error,
            "registers": {f"R{i}": vm.gp[i] for i in range(16)},
            "stack_depth": len(vm.stack),
        }

    def _infer_operation(self, description: str) -> LCAROp:
        """Infer the LCAR operation from a description string.

        Uses simple keyword matching to map descriptions to operations.

        Args:
            description: Human-readable description.

        Returns:
            Best-guess :class:`LCAROp`.
        """
        desc = description.lower()
        if any(kw in desc for kw in ("compute", "calculate", "math")):
            return LCAROp.COMPUTE
        if any(kw in desc for kw in ("aggregate", "sum", "collect")):
            return LCAROp.AGGREGATE
        if any(kw in desc for kw in ("schedule", "dispatch", "assign")):
            return LCAROp.SCHEDULE
        if any(kw in desc for kw in ("transfer", "move", "send")):
            return LCAROp.TRANSFER
        if any(kw in desc for kw in ("sync", "barrier", "wait")):
            return LCAROp.BARRIER
        if any(kw in desc for kw in ("status", "check", "query")):
            return LCAROp.STATUS
        if any(kw in desc for kw in ("broadcast", "notify", "alert")):
            return LCAROp.BROADCAST
        if any(kw in desc for kw in ("shutdown", "stop", "halt")):
            return LCAROp.SHUTDOWN
        return LCAROp.COMPUTE
