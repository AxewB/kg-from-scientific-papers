from logging import Logger
from pathlib import Path
from typing import Any, Optional

import requests

import helpers.logger as lg


class GrobidClient:
    def __init__(self, base_url: str = "http://localhost:8070") -> None:
        self.base_url: str = base_url
        self.logger: Logger = lg.get_logger()

    def is_alive(self) -> bool:
        try:
            r = requests.get(self.base_url, timeout=5)
            return r.status_code == 200
        except requests.RequestException as e:
            self.logger.warning("GROBID not reachable: %s", e)
            return False

    def process_header(self, pdf_path: Path) -> Optional[str]:
        return self._post_pdf("/api/processHeaderDocument", pdf_path)

    def process_fulltext(self, pdf_path: Path) -> str | None:
        params = {
            "consolidateHeader": 1,
            "consolidateCitations": 1,
            "segmentSentences": 1,
        }

        return self._post_pdf(
            "/api/processFulltextDocument",
            pdf_path,
            params=params,
            timeout=180,
        )

    def process_references(self, pdf_path: Path) -> str | None:
        return self._post_pdf("/api/processReferences", pdf_path)

    # private
    def _post_pdf(
        self,
        endpoint: str,
        pdf_path: Path,
        params: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> str | None:
        if not pdf_path.exists():
            self.logger.error("PDF not found: %s", pdf_path)
            return None

        url = f"{self.base_url}{endpoint}"

        try:
            with open(pdf_path, "rb") as f:
                files = {"input": (pdf_path.name, f, "application/pdf")}
                r = requests.post(url, files=files, params=params, timeout=timeout)

            if r.status_code != 200:
                self.logger.error("GROBID error %s on %s", r.status_code, endpoint)
                return None

            return r.text

        except requests.RequestException as e:
            self.logger.error("Request failed: %s", e)
            return None
