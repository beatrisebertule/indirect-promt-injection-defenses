"""
Randomized and Plain defense implementations for benchmarking.
"""

import random
from defenses import Defenses


class DefenseRandomizer:
    """Handles randomization of defense parameters and system prompts."""

    """
    Possible random values for spotlighting
    """
    # 10 role words x 10 format patterns = 100 delimiter pairs
    _data_words = [
        "DATA",
        "INPUT",
        "CONTEXT",
        "CONTENT",
        "EXTERNAL",
        "UNTRUSTED",
        "SOURCE",
        "TEXT",
        "PAYLOAD",
        "DOCUMENT",
    ]
    delimiters = []
    for _w in _data_words:
        _wl = _w.lower()
        delimiters += [
            # <|begin_x|> / <|end_x|>  — llama3 header style
            (f"<|begin_{_wl}|>", f"<|end_{_wl}|>"),
            # <|x_start|> / <|x_end|>  — pipe token style
            (f"<|{_wl}_start|>", f"<|{_wl}_end|>"),
            # [X] / [/X]  — [INST] style
            (f"[{_w}]", f"[/{_w}]"),
            # [BEGIN_X] / [END_X]  — bracket block style
            (f"[BEGIN_{_w}]", f"[END_{_w}]"),
            # <<X>> / <</X>>  — <<SYS>> style
            (f"<<{_w}>>", f"<</{_w}>>"),
            # <x> / </x>  — XML tag style
            (f"<{_wl}>", f"</{_wl}>"),
            # ---BEGIN_X--- / ---END_X---  — markdown separator style
            (f"---BEGIN_{_w}---", f"---END_{_w}---"),
            # BEGIN_X / END_X  — plain block style
            (f"BEGIN_{_w}", f"END_{_w}"),
            # <<<X>>> / <<</X>>>  — arrow bracket style
            (f"<<<{_w}>>>", f"<<</{_w}>>>"),
            # ### X: / ### END_X:  — markdown header style
            (f"### {_w}:", f"### END_{_w}:"),
        ]
    del _w, _wl

    _marking_chars = ["@", "#", "$", "%", "&", "*", "~", "+", "=", "!", "^"]
    # 11 singles + 55 two-char pairs + 110 XYX triples = 176 markers
    datamarkings = _marking_chars[:]
    for _i, _ma in enumerate(_marking_chars):
        for _mb in _marking_chars[_i + 1 :]:
            datamarkings.append(_ma + _mb)  # 2-char unordered pairs
        for _mb in _marking_chars:
            if _ma != _mb:
                datamarkings.append(_ma + _mb + _ma)  # 3-char XYX
    del _i, _ma, _mb

    encodings = [
        "base64",
        "base32",
        "rot13",
        "rot47",
        "hex",
        "octal",
        "binary",
        "unicode_escape",
        "morse",
    ]

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def random_delimiting_params(self) -> dict:
        """Generate random delimiters for each sample."""
        start, end = self._rng.choice(self.delimiters)
        return {"start_marker": start, "end_marker": end}

    def random_datamarking_params(self) -> dict:
        """Generate random marker symbols for each sample."""
        return {"marker": self._rng.choice(self.datamarkings)}

    def random_encoding_params(self) -> dict:
        """Generate random encoding scheme for each sample."""
        return {"encoding": self._rng.choice(self.encodings)}

    def build_random_system_prompt(self, defense_name: str, params: dict) -> str:
        """Generate system prompt for the defense using random params."""
        return getattr(Defenses, f"{defense_name}_system_prompt")(**params)

    def build_defense_prompt(
        self, user_instr: str, ext_content: str, defense_name: str
    ) -> tuple[str, str]:
        """Build defense prompt with random parameters."""
        if defense_name == "delimiting":
            params = self.random_delimiting_params()
        elif defense_name == "datamarking":
            params = self.random_datamarking_params()
        elif defense_name == "encoding":
            params = self.random_encoding_params()
        else:
            raise ValueError(f"Unknown defense: {defense_name}")

        # Apply the defense with random parameters
        defended = getattr(Defenses, defense_name)(ext_content, **params)
        system_prompt = self.build_random_system_prompt(defense_name, params)
        user_msg = f"{user_instr}\n\n{defended}"
        return system_prompt, user_msg


class DefensePlain:
    """Plain (non-randomized) defense implementation using fixed parameters."""

    # Fixed parameters for plain defenses
    _plain_params = {
        "delimiting": {
            "start_marker": "<<<UNTRUSTED_DATA_START>>>",
            "end_marker": "<<<UNTRUSTED_DATA_END>>>",
        },
        "datamarking": {
            "marker": "^",
        },
        "encoding": {
            "encoding": "base64",
        },
    }

    @staticmethod
    def build_defense_prompt(
        user_instr: str, ext_content: str, defense_name: str
    ) -> tuple[str, str]:
        """Build defense prompt with plain/fixed parameters."""
        if defense_name not in DefensePlain._plain_params:
            raise ValueError(f"Unknown defense: {defense_name}")

        params = DefensePlain._plain_params[defense_name]

        # Apply the defense with fixed parameters
        defended = getattr(Defenses, defense_name)(ext_content, **params)
        system_prompt = getattr(Defenses, f"{defense_name}_system_prompt")(**params)
        user_msg = f"{user_instr}\n\n{defended}"
        return system_prompt, user_msg
