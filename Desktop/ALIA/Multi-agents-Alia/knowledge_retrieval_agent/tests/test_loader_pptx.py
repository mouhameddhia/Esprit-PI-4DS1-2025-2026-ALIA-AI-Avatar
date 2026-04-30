"""Tests for PPTX extraction in the loader."""

from __future__ import annotations

import zipfile

from app.ingestion.loader import PDFLoader


def test_loader_extracts_text_from_pptx(tmp_path) -> None:
    """PPTX slides should be extracted from XML text runs, not binary blobs."""

    pptx_path = tmp_path / "sample.pptx"
    slide_xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<p:sld xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'
       xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main'>
  <p:cSld>
    <p:spTree>
      <p:sp>
        <p:txBody>
          <a:p>
            <a:r><a:t>BACTOL composition</a:t></a:r>
            <a:r><a:t>chlorhexidine and excipients</a:t></a:r>
          </a:p>
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
"""

    with zipfile.ZipFile(pptx_path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml)

    loader = PDFLoader(str(tmp_path))
    documents = loader.load()

    assert len(documents) == 1
    assert "BACTOL composition" in documents[0].text
    assert "chlorhexidine" in documents[0].text
