from typing import Dict, Any

BASE_INSTRUCTION = (
    "You are a deterministic converter to UNIMARC in XML.\n"
    "Rules :\n"
    "- ONLY returns a valid and well-formed XML.\n"
    "- UTF-8 encoding, no comment, no text out of tags.\n"
    "- Use the relevant Unimarc fields (200, 210, 700…).\n"
    "- If an info is missing, omit the corresponding field (does not invent anything).\n\n"
    "Metadata :\n{metadata}\n\nXML UNIMARC :"
)

SYSTEM_PROMPT = (
    "You are a deterministic converter to UNIMARC in XML. "
    "Output only valid XML, well-formed and UTF-8 encoded."
)


def make_example(metadata: str, unimarc_record: str, mode: str = "base") -> Dict[str, Any]:
    if mode == "base":
        prompt = BASE_INSTRUCTION.format(metadata=metadata)
        return {"prompt": prompt, "label": unimarc_record}
    elif mode == "chat":
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": metadata},
                {"role": "assistant", "content": unimarc_record},
            ]
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
