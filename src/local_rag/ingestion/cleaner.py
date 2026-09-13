import re
import unicodedata


class TextCleaner:
    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        # Join words broken by PDF line wrapping:
        # meta-
        # material
        # ->
        # metamaterial
        text = re.sub(
            r"(?<=\w)-\s*\n\s*(?=\w)",
            "",
            text,
        )

        # Preserve paragraph boundaries before removing
        # unnecessary line breaks.
        text = re.sub(
            r"\n\s*\n+",
            "\n\n",
            text,
        )

        paragraphs = []

        for paragraph in text.split("\n\n"):
            paragraph = re.sub(
                r"\s*\n\s*",
                " ",
                paragraph,
            )

            paragraph = re.sub(
                r"[ \t]+",
                " ",
                paragraph,
            )

            paragraph = paragraph.strip()

            if paragraph:
                paragraphs.append(
                    paragraph
                )

        return "\n\n".join(
            paragraphs
        )