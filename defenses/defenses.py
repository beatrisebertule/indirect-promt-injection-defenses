
"""
Code inspired from LLM jailbreaking tutorial
"""

import base64


class Defenses:

    @staticmethod
    def delimiting(
        untrusted_input: str,
        start_marker: str = "<<<UNTRUSTED_DATA_START>>>",
        end_marker: str = "<<<UNTRUSTED_DATA_END>>>",
    ) -> str:
        """
        Spotlighting via Delimiting:
        Encloses the untrusted input with unique boundary markers.
        """
        secure_prompt = f"{start_marker}\n{untrusted_input}\n{end_marker}"
        return secure_prompt

    @staticmethod
    def datamarking(untrusted_input: str, marker: str = "^") -> str:
        """
        Spotlighting via Datamarking:
        Interleaves a special token throughout the untrusted input (between words)
        to provide a continuous signal of provenance.
        """
        marked_input = marker.join(untrusted_input.split())
        secure_prompt = marked_input
        return secure_prompt

    @staticmethod
    def encoding(untrusted_input: str, encoding: str = "base64") -> str:
        """
        Spotlighting via Encoding:
        Encodes the untrusted input so it cannot be parsed as raw instructions
        by the model until it is explicitly evaluated within the context of being data.
        """
        if encoding == "base64":
            secure_prompt = base64.b64encode(untrusted_input.encode()).decode()
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        return secure_prompt
