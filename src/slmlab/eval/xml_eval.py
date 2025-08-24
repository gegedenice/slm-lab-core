from xml.etree import ElementTree as ET
from typing import List, Tuple

def xml_is_well_formed(s: str) -> bool:
    try:
        ET.fromstring(s)
        return True
    except Exception:
        return False

def extract_field_pairs(s: str) -> List[Tuple[str,str]]:
    pairs = []
    try:
        root = ET.fromstring(s)
    except Exception:
        return pairs
    for df in root.findall("datafield"):
        tag = df.get("tag", "")
        for sf in df.findall("subfield"):
            code = sf.get("code", "")
            pairs.append((tag, code))
    return pairs

def coverage_against_ref(pred_xml: str, ref_xml: str) -> float:
    ref = set(extract_field_pairs(ref_xml))
    if not ref:
        return 0.0
    pred = set(extract_field_pairs(pred_xml))
    return len(ref & pred) / len(ref)