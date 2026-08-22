#!/usr/bin/env python3
"""Write a deterministic SPDX inventory for every installed Python distribution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote


LICENSE_CLASSIFIER_MAP = {
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "MIT License": "MIT",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Python Software Foundation License": "PSF-2.0",
}


def normalized_license(metadata: importlib.metadata.PackageMetadata) -> str:
    expression = metadata.get("License-Expression")
    if expression:
        return expression
    for classifier in metadata.get_all("Classifier") or []:
        prefix = "License :: OSI Approved :: "
        if classifier.startswith(prefix):
            return LICENSE_CLASSIFIER_MAP.get(classifier[len(prefix) :], "NOASSERTION")
    return "NOASSERTION"


def distributions() -> list[dict[str, Any]]:
    result = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        classifiers = [
            value
            for value in distribution.metadata.get_all("Classifier") or []
            if value.startswith("License ::")
        ]
        result.append(
            {
                "name": name,
                "version": distribution.version,
                "license_declared": normalized_license(distribution.metadata),
                "license_classifiers": sorted(classifiers),
                "home_page": distribution.metadata.get("Home-page"),
            }
        )
    return sorted(result, key=lambda item: (item["name"].lower(), item["version"]))


def spdx_id(name: str, version: str) -> str:
    stable = hashlib.sha256(f"{name}=={version}".encode()).hexdigest()[:16]
    return f"SPDXRef-Package-{stable}"


def build_spdx(packages: list[dict[str, Any]], runtime_id: str) -> dict[str, Any]:
    identity = hashlib.sha256(
        json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    spdx_packages = []
    relationships = []
    for package in packages:
        package_id = spdx_id(package["name"], package["version"])
        spdx_packages.append(
            {
                "SPDXID": package_id,
                "name": package["name"],
                "versionInfo": package["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": package["license_declared"],
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": (
                            f"pkg:pypi/{quote(package['name'].lower())}@"
                            f"{quote(package['version'])}"
                        ),
                    }
                ],
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": package_id,
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{runtime_id}-python-environment",
        "documentNamespace": f"https://intelligent-liars.invalid/spdx/{runtime_id}/{identity}",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: docker/tinylora-step5/write_sbom.py"],
        },
        "packages": spdx_packages,
        "relationships": relationships,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packages = distributions()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema_version": 1,
        "runtime_id": args.runtime_id,
        "packages": packages,
    }
    (args.output_dir / "python-package-licenses.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "python-packages.spdx.json").write_text(
        json.dumps(build_spdx(packages, args.runtime_id), indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
