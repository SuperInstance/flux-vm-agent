"""
FLUX Natural Language Interpreter — translates English commands to FLUX bytecode.

Architecture:
  - ``Vocabulary``: registry of NL pattern → assembly template mappings.
  - ``FluxInterpreter``: matches input text against the vocabulary,
    fills in parameters, assembles, and executes.

Supported patterns (built-in vocabulary):
  - Arithmetic: ``compute A + B``, ``compute A - B``, ``compute A * B``, ``compute A / B``
  - Functions:  ``factorial of N``, ``fibonacci of N``
  - Transforms: ``double N``, ``square N``
  - Ranges:     ``sum A to B``, ``power of base to exp``
  - Misc:       ``hello`` (returns 42)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from flux_vm import FluxVM
from flux_assembler import FluxAssembler


# ─── Vocabulary Entry ────────────────────────────────────────────────────────

@dataclass
class VocabEntry:
    """A single vocabulary entry mapping a natural language pattern to assembly.

    Attributes:
        pattern: Human-readable pattern string (e.g. ``"compute $a + $b"``).
        assembly: Assembly template with ``${var}`` placeholders.
        result_reg: Register index holding the computation result.
        description: Human-readable description of the computation.
        regex: Compiled regex for matching input text.
    """

    pattern: str
    assembly: str
    result_reg: int = 0
    description: str = ""
    regex: re.Pattern = field(default_factory=lambda: re.compile("^$"), repr=False)


# ─── Vocabulary ──────────────────────────────────────────────────────────────

class Vocabulary:
    """Natural language pattern → assembly template registry.

    Patterns use ``$name`` placeholders that capture integer values from
    the input text. Placeholders are compiled into named regex groups.

    Example::

        vocab = Vocabulary()
        vocab.add("compute $a + $b",
                  "MOVI R0, ${a}\\nMOVI R1, ${b}\\nIADD R0, R0, R1\\nHALT",
                  result_reg=0, description="Add two numbers")
    """

    def __init__(self) -> None:
        self.entries: List[VocabEntry] = []

    def add(
        self,
        pattern: str,
        assembly: str,
        result_reg: int = 0,
        description: str = "",
    ) -> None:
        """Register a new vocabulary entry.

        Args:
            pattern: NL pattern with ``$var`` placeholders for integers.
            assembly: Assembly template using ``${var}`` for substitution.
            result_reg: Register holding the final result.
            description: Human-readable description.
        """
        self.entries.append(VocabEntry(
            pattern=pattern,
            assembly=assembly,
            result_reg=result_reg,
            description=description,
            regex=self._compile_regex(pattern),
        ))

    def match(self, text: str) -> Optional[Dict]:
        """Match *text* against all registered patterns.

        Returns:
            A dict with keys ``assembly``, ``result_reg``, ``description``,
            and ``groups`` (captured placeholder values), or ``None`` if
            no pattern matches.
        """
        for entry in self.entries:
            m = entry.regex.search(text)
            if m:
                return {
                    "assembly": entry.assembly,
                    "result_reg": entry.result_reg,
                    "description": entry.description,
                    "groups": m.groupdict(),
                }
        return None

    def list_patterns(self) -> List[str]:
        """Return a list of all registered pattern strings."""
        return [e.pattern for e in self.entries]

    @staticmethod
    def _compile_regex(pattern: str) -> re.Pattern:
        """Compile a ``$var``-style pattern into a regex.

        Each ``$name`` becomes a named group ``(?P<name>\\d+)`` matching
        one or more digits. All other text is escaped.
        """
        parts = re.split(r"(\$\w+)", pattern)
        regex_parts: List[str] = []
        for p in parts:
            if p.startswith("$"):
                regex_parts.append(f"(?P<{p[1:]}>" r"\d+)")
            else:
                regex_parts.append(re.escape(p))
        return re.compile("".join(regex_parts), re.IGNORECASE)

    @classmethod
    def builtin(cls) -> "Vocabulary":
        """Create a vocabulary pre-loaded with all built-in patterns.

        Returns:
            A new ``Vocabulary`` instance with standard patterns registered.
        """
        vocab = cls()
        vocab.add(
            "compute $a + $b",
            "MOVI R0, ${a}\nMOVI R1, ${b}\nIADD R0, R0, R1\nHALT",
            result_reg=0,
            description="Add two numbers",
        )
        vocab.add(
            "compute $a - $b",
            "MOVI R0, ${a}\nMOVI R1, ${b}\nISUB R0, R0, R1\nHALT",
            result_reg=0,
            description="Subtract two numbers",
        )
        vocab.add(
            "compute $a * $b",
            "MOVI R0, ${a}\nMOVI R1, ${b}\nIMUL R0, R0, R1\nHALT",
            result_reg=0,
            description="Multiply two numbers",
        )
        vocab.add(
            "compute $a / $b",
            "MOVI R0, ${a}\nMOVI R1, ${b}\nIDIV R0, R0, R1\nHALT",
            result_reg=0,
            description="Divide two numbers",
        )
        vocab.add(
            "factorial of $n",
            "MOVI R0, ${n}\nMOVI R1, 1\nloop: IMUL R1, R1, R0\nDEC R0\nJNZ R0, loop\nHALT",
            result_reg=1,
            description="Compute n!",
        )
        vocab.add(
            "fibonacci of $n",
            (
                "MOVI R0, ${n}\nMOVI R1, 0\nMOVI R2, 1\n"
                "DEC R0\nloop: MOV R3, R2\nIADD R2, R2, R1\nMOV R1, R3\n"
                "DEC R0\nJNZ R0, loop\nHALT"
            ),
            result_reg=2,
            description="Compute F(n)",
        )
        vocab.add(
            "double $n",
            "MOVI R0, ${n}\nIADD R0, R0, R0\nHALT",
            result_reg=0,
            description="Double a number",
        )
        vocab.add(
            "square $n",
            "MOVI R0, ${n}\nIMUL R0, R0, R0\nHALT",
            result_reg=0,
            description="Square a number",
        )
        vocab.add(
            "sum $a to $b",
            (
                "MOVI R0, ${a}\nMOVI R1, ${b}\nMOVI R2, 0\n"
                "loop: IADD R2, R2, R0\nCMP R0, R1\nJZ R13, done\n"
                "INC R0\nJNZ R13, loop\ndone: HALT"
            ),
            result_reg=2,
            description="Sum from a to b inclusive",
        )
        vocab.add(
            "power of $base to $exp",
            "MOVI R0, ${base}\nMOVI R1, ${exp}\nMOVI R2, 1\nloop: IMUL R2, R2, R0\nDEC R1\nJNZ R1, loop\nHALT",
            result_reg=2,
            description="Compute base^exp",
        )
        vocab.add(
            "hello",
            "MOVI R0, 42\nHALT",
            result_reg=0,
            description="Returns 42",
        )
        return vocab


# ─── Interpreter ─────────────────────────────────────────────────────────────

@dataclass
class InterpretResult:
    """Result of a natural language interpretation.

    Attributes:
        value: Computed integer result, or ``None`` on failure.
        message: Human-readable status message.
        pattern: The matched vocabulary pattern (if any).
        cycles: Number of VM cycles executed.
    """

    value: Optional[int]
    message: str
    pattern: str = ""
    cycles: int = 0


class FluxInterpreter:
    """Natural language → FLUX bytecode → result.

    Matches input text against the vocabulary, substitutes parameters,
    assembles to bytecode, and executes on a :class:`FluxVM`.

    Example::

        interp = FluxInterpreter()
        result = interp.run("compute 7 + 5")
        assert result.value == 12
    """

    def __init__(self, vocab: Optional[Vocabulary] = None) -> None:
        """Initialize with an optional custom vocabulary.

        Args:
            vocab: Vocabulary instance. Defaults to the built-in vocabulary.
        """
        self.vocab: Vocabulary = vocab or Vocabulary.builtin()
        self.assembler: FluxAssembler = FluxAssembler()

    def run(self, text: str) -> InterpretResult:
        """Interpret natural language text and execute.

        Args:
            text: Natural language input (e.g. ``"compute 3 + 4"``).

        Returns:
            An :class:`InterpretResult` with the computed value and status.
        """
        match = self.vocab.match(text)
        if match is None:
            return InterpretResult(
                value=None,
                message=f"No match for: {text[:80]}",
            )

        asm = match["assembly"]
        for key, val in match["groups"].items():
            asm = asm.replace("${" + key + "}", str(val))

        try:
            bc = self.assembler.assemble(asm)
            vm = FluxVM(bc)
            vm.execute()
            if vm.halted:
                return InterpretResult(
                    value=vm.reg(match["result_reg"]),
                    message=f"OK ({vm.cycles} cycles)",
                    pattern=match["description"],
                    cycles=vm.cycles,
                )
            return InterpretResult(
                value=None,
                message=f"VM error: {vm.error}",
                pattern=match["description"],
                cycles=vm.cycles,
            )
        except Exception as exc:
            return InterpretResult(
                value=None,
                message=f"Assembly error: {exc}",
                pattern=match["description"],
            )

    def translate(self, text: str) -> Optional[str]:
        """Translate natural language to assembly without executing.

        Args:
            text: Natural language input.

        Returns:
            Assembly source code, or ``None`` if no pattern matched.
        """
        match = self.vocab.match(text)
        if match is None:
            return None

        asm = match["assembly"]
        for key, val in match["groups"].items():
            asm = asm.replace("${" + key + "}", str(val))
        return asm

    def compile(self, text: str) -> Optional[bytes]:
        """Translate natural language to compiled bytecode.

        Args:
            text: Natural language input.

        Returns:
            Compiled bytecode, or ``None`` if no pattern matched.
        """
        asm = self.translate(text)
        if asm is None:
            return None
        return self.assembler.assemble(asm)
