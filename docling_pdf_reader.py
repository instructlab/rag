from enum import Enum
from typing import Iterable

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document as LIDocument
from pydantic import BaseModel

from docling.document_converter import DocumentConverter


class DocumentMetadata(BaseModel):
    dl_doc_hash: str


class DoclingPDFReader(BasePydanticReader):
    class ParseType(str, Enum):
        MARKDOWN = "markdown"
        # JSON = "json"

    parse_type: ParseType = ParseType.MARKDOWN

    def lazy_load_data(self, file_path: str | list[str]) -> Iterable[LIDocument]:
        file_paths = file_path if isinstance(file_path, list) else [file_path]
        converter = DocumentConverter()
        for source in file_paths:
            dl_doc = converter.convert_single(source).output
            match self.parse_type:
                case self.ParseType.MARKDOWN:
                    text = dl_doc.export_to_markdown()
                # case self.ParseType.JSON:
                #     text = dl_doc.model_dump_json()
                case _:
                    raise RuntimeError(
                        f"Unexpected parse type encountered: {self.parse_type}"
                    )
            excl_metadata_keys = ["dl_doc_hash"]
            li_doc = LIDocument(
                doc_id=dl_doc.file_info.document_hash,
                text=text,
                excluded_embed_metadata_keys=excl_metadata_keys,
                excluded_llm_metadata_keys=excl_metadata_keys,
            )
            li_doc.metadata = DocumentMetadata(
                dl_doc_hash=dl_doc.file_info.document_hash,
            ).model_dump()
            yield li_doc
