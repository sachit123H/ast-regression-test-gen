"""Cobertura XML parser producing line-level and method-level spectra over `classes.relevant`.

Implements Proposal Section 4.6:
- Coverage is measured with Cobertura on the pre-fix version ONLY over the
  relevant classes (`classes.relevant` exported by Defects4J).
- Extracts:
  - Universe of line elements `(class_fqn, line_number)` -> covered boolean
  - Mapping from each line element to its enclosing method `(class_fqn, method_name, signature)`
    so line-level Ochiai/DStar scores can be aggregated to method level by max score.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


LineElement = str    # Canonical key: "<class_fqn>:<line_number>"
MethodElement = str  # Canonical key: "<class_fqn>#<method_name><signature>"


@dataclass
class CoberturaCoverage:
    """Line coverage and line-to-method mapping extracted from a Cobertura XML report."""

    covered_lines: Set[LineElement]
    all_lines: Set[LineElement]
    line_to_method: Dict[LineElement, MethodElement]


def _matches_relevant(class_name: str, relevant_classes: Optional[Set[str]]) -> bool:
    if not relevant_classes:
        return True
    if class_name in relevant_classes:
        return True
    # Include inner classes (e.g., Foo$Bar) of any relevant class
    outer = class_name.split("$", 1)[0]
    return outer in relevant_classes


def parse_cobertura_xml(
    xml_path: Path,
    relevant_classes: Optional[Iterable[str]] = None,
) -> CoberturaCoverage:
    """Parse a Cobertura `coverage.xml` file filtered to `relevant_classes`."""
    rel_set = set(relevant_classes) if relevant_classes else None
    tree = ET.parse(xml_path)
    root = tree.getroot()

    covered: Set[LineElement] = set()
    all_lines: Set[LineElement] = set()
    line_to_method: Dict[LineElement, MethodElement] = {}

    for cls_el in root.findall(".//class"):
        cls_name = cls_el.attrib.get("name", "")
        if not _matches_relevant(cls_name, rel_set):
            continue

        # 1. Extract method-scoped lines (provides exact line -> method mapping)
        for m_el in cls_el.findall("./methods/method"):
            m_name = m_el.attrib.get("name", "")
            m_sig = m_el.attrib.get("signature", "")
            method_key = f"{cls_name}#{m_name}{m_sig}"
            for line_el in m_el.findall("./lines/line"):
                ln = line_el.attrib["number"]
                elem = f"{cls_name}:{ln}"
                all_lines.add(elem)
                line_to_method[elem] = method_key
                if int(line_el.attrib.get("hits", "0")) > 0:
                    covered.add(elem)

        # 2. Also inspect class-level lines in case static initializers appear only there
        for line_el in cls_el.findall("./lines/line"):
            ln = line_el.attrib["number"]
            elem = f"{cls_name}:{ln}"
            all_lines.add(elem)
            line_to_method.setdefault(elem, f"{cls_name}#<clinit>()V")
            if int(line_el.attrib.get("hits", "0")) > 0:
                covered.add(elem)

    return CoberturaCoverage(
        covered_lines=covered,
        all_lines=all_lines,
        line_to_method=line_to_method,
    )


def build_coverage_matrix(
    test_coverages: List[CoberturaCoverage],
    elements_order: Optional[List[LineElement]] = None,
) -> Tuple[List[LineElement], List[List[int]], Dict[LineElement, MethodElement]]:
    """Build an `m x n` binary coverage matrix `C_T` aligned to a canonical element order."""
    line_to_method: Dict[LineElement, MethodElement] = {}
    if elements_order is None:
        all_elems: Set[LineElement] = set()
        for cov in test_coverages:
            all_elems.update(cov.all_lines)
            line_to_method.update(cov.line_to_method)
        elements = sorted(all_elems)
    else:
        elements = list(elements_order)
        for cov in test_coverages:
            line_to_method.update(cov.line_to_method)

    matrix: List[List[int]] = []
    for cov in test_coverages:
        row = [1 if e in cov.covered_lines else 0 for e in elements]
        matrix.append(row)

    return elements, matrix, line_to_method
