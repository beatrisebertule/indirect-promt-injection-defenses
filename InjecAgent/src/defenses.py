import base64
import random
import string


class Defenses:

    @staticmethod
    def delimiting(
        text,
        start_marker="<<<UNTRUSTED_DATA_START>>>",
        end_marker="<<<UNTRUSTED_DATA_END>>>"
    ):
        return f"{start_marker}\n{text}\n{end_marker}"

    @staticmethod
    def datamarking(text, marker="^"):
        return marker.join(text.split())

    @staticmethod
    def randomized_datamarking(text):
        marker = random.choice([
            "^",
            "#",
            "@@",
            "§§",
            "|||",
            "%%"
        ])

        return marker.join(text.split())

    @staticmethod
    def randomized_per_word_datamarking(text):
        markers = ["^", "#", "@@", "§§", "|||", "%%"]

        words = text.split()

        if len(words) <= 1:
            return text

        output = []

        for i, word in enumerate(words):
            output.append(word)

            if i != len(words) - 1:
                output.append(random.choice(markers))

        return "".join(output)

    @staticmethod
    def random_token_delimiting(text):

        random_token = ''.join(
            random.choices(
                string.ascii_uppercase + string.digits,
                k=8
            )
        )

        start_marker = f"<<{random_token}_START>>"
        end_marker = f"<<{random_token}_END>>"

        return f"{start_marker}\n{text}\n{end_marker}"

    @staticmethod
    def encoding(text):
        return base64.b64encode(text.encode()).decode()