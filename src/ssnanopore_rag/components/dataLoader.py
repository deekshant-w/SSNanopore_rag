import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

PROJECT_DIR = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)


def parse_ris_data(ris_file: Path) -> list[dict[str, str]]:
    """
    Parse a RIS file and return a list of dictionaries for all key value pairs.
    The list is sorted but not separated into entries.

    Args:
        ris_file: Path to the RIS file

    Returns:
        List of dictionaries, where each dictionary contains a 'key' and 'value' field.
        The 'key' is the field type (e.g., 'TI') and the 'value' is the field content.
    """
    entries: list[dict[str, str]] = []

    def process_content(content):
        content = content.strip()
        if not content:
            return
        try:
            key, value = content.split("  - ", 1)
        except ValueError:
            logger.warning(f"Failed to parse content: {content}")
            return
        entries.append({"key": key.strip(), "value": value.strip()})

    with open(ris_file, encoding="utf-8") as file:
        lineContent = ""
        for line in file:
            # if line is of the type <capital char><capital char><space><space><-><space><text>:
            match = re.match(r"^[A-Z][A-Z0-9]  -", line)
            if match:
                # process the previous linecontent
                process_content(lineContent)
                lineContent = line.strip()
            else:
                lineContent += line.strip() + " "
        process_content(lineContent)
    return entries


def check_keys(data: list[dict[str, str]]):
    from collections import Counter

    keys = Counter()
    for item in data:
        # print(item)
        keys.update([item["key"]])
    print(keys)


class Paper(BaseModel):
    model_config = ConfigDict(extra="allow")

    def process(self) -> dict[str, Any]:
        result = {}
        for key, value in self.model_dump().items():
            if not value.strip():
                continue
            match key:
                case "TI":
                    result["title"] = value
                case "AU":
                    result["authors"] = result.get("authors", []) + [value]
                case "AB":
                    result["abstract"] = value
                case "KW":
                    result["keywords"] = result.get("keywords", []) + [value]
                case "TY":
                    result["type of reference"] = value
                case "DO":
                    result["doi"] = value
                case "UR":
                    result["url"] = value
                case "PB":
                    result["publisher"] = value
                case "DA":
                    result["date"] = value
        return result


def convert_ris_data_to_entities(data: list[dict[str, str]]):
    findStage = True  # True -> Find TI
    papers: list[dict[str, Any]] = []
    paper = Paper()
    i = 0
    while i < len(data):
        if findStage:
            if data[i]["key"] == "TI":
                findStage = False
                setattr(paper, data[i]["key"], data[i]["value"])
                i += 1
                continue
            else:
                i += 1
                continue
        else:
            if data[i]["key"] == "TI":
                papers.append(paper.process())
                paper = Paper()
                findStage = True
                continue
            else:
                setattr(paper, data[i]["key"], data[i]["value"])
                i += 1
    papers.append(paper.process())
    return papers


def dataLoadingUtility(ris_file: Path, output_file: Path):
    data = parse_ris_data(ris_file)
    papers = convert_ris_data_to_entities(data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)


def main():
    dataPath = PROJECT_DIR / "data"
    ris_file = dataPath / "kyle_export.ris"
    data = parse_ris_data(ris_file)
    # check_keys(data)
    papers = convert_ris_data_to_entities(data)
    with open(dataPath / "papers.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)


if __name__ == "__main__":
    main()
