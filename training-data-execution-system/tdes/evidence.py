"""Evidence collection and generation."""

from typing import Dict, Any
from pathlib import Path
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)


class EvidenceCollector:
    """Collects and generates evidence for all requirements."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def record(self, requirement: str, status: str, artifact: str, details: Dict[str, Any] = None):
        """Record evidence for a requirement.
        
        Args:
            requirement: Name of requirement
            status: PASS or FAIL
            artifact: Path to supporting artifact
            details: Optional additional details
        """
        self.results[requirement] = {
            "status": status,
            "artifact": artifact,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        log.info(f"Evidence recorded: {requirement} = {status}")
    
    def generate_json(self, output_path: Path):
        """Generate evidence.json."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        log.info(f"Generated {output_path}")
    
    def generate_markdown(self, output_path: Path):
        """Generate evidence.md table."""
        lines = [
            "# Training Data Execution System - Evidence Bundle",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"Total requirements: {len(self.results)}",
            f"Passed: {sum(1 for r in self.results.values() if r['status'] == 'PASS')}",
            f"Failed: {sum(1 for r in self.results.values() if r['status'] == 'FAIL')}",
            "",
            "## Evidence Table",
            "",
            "| Requirement | Result | Evidence | Details |",
            "|-------------|--------|----------|---------|"
        ]
        
        for req, data in self.results.items():
            status = data["status"]
            artifact = data["artifact"]
            details = json.dumps(data.get("details", {}))[:50]
            
            lines.append(f"| {req} | {status} | {artifact} | {details} |")
        
        lines.append("")
        lines.append("## Detailed Results")
        lines.append("")
        
        for req, data in self.results.items():
            lines.append(f"### {req}")
            lines.append(f"- **Status:** {data['status']}")
            lines.append(f"- **Artifact:** {data['artifact']}")
            lines.append(f"- **Timestamp:** {data['timestamp']}")
            if data.get("details"):
                lines.append(f"- **Details:** {json.dumps(data['details'], indent=2)}")
            lines.append("")
        
        output_path.write_text("\n".join(lines))
        
        log.info(f"Generated {output_path}")
    
    def get_status(self, requirement: str) -> str:
        """Get status of a requirement."""
        return self.results.get(requirement, {}).get("status", "UNKNOWN")
    
    def all_passed(self) -> bool:
        """Check if all requirements passed."""
        return all(r["status"] == "PASS" for r in self.results.values())
