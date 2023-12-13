import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from haystack import Document, component
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class JsonToDocument:
    """
    Converts JSON content into a Haystack Document. Handles both JSON objects and arrays of objects.

    Usage example:
    ```python
    from your_module import JsonToDocument  # Replace with actual module name

    converter = JsonToDocument()
    results = converter.run(sources=["earthquakes.json"])
    documents = results["documents"]
    ```
    """

    def __init__(self, content_field: str = "content", meta_fields: Optional[List[str]] = None, progress_bar: bool = True):
        """
        :param content_field: The field in JSON to use as the content of the Document. Defaults to 'content'.
        :param meta_fields: List of fields to include in the metadata of the Document. Defaults to None (include all fields).
        :param progress_bar: Show a progress bar for the conversion. Defaults to True.
        """
        self.content_field = content_field
        self.meta_fields = meta_fields
        self.progress_bar = progress_bar

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        """
        Reads JSON content and converts it to Documents.

        :param sources: A list of JSON data sources (file paths or binary objects)
        """
        documents = []

        for source in tqdm(
            sources,
            desc="Converting JSON files to Documents",
            disable=not self.progress_bar,
        ):
            try:
                file_content = self._extract_content(source)
                json_data = json.loads(file_content)

                if isinstance(json_data, list):
                    for item in json_data:
                        document = self._create_document(item)
                        if document:
                            documents.append(document)
                elif isinstance(json_data, dict):
                    document = self._create_document(json_data)
                    if document:
                        documents.append(document)
                else:
                    logger.warning("Unsupported JSON format in %s", source)

            except Exception as e:
                logger.warning("Failed to process %s. Error: %s", source, e)

        return {"documents": documents}

    def _create_document(self, data: dict) -> Optional[Document]:
        """
        Creates a Document from a dictionary.

        :param data: The dictionary to create a Document from.
        :return: The created Document or None if there's an issue.
        """
        try:
            content = data.get(self.content_field, "")
            metadata = {k: v for k, v in data.items() if not self.meta_fields or k in self.meta_fields}
            return Document(content=content, meta=metadata)
        except Exception as e:
            logger.warning("Failed to create document from data. Error: %s", e)
            return None

    def _extract_content(self, source: Union[str, Path, ByteStream]) -> str:
        """
        Extracts content from the given data source.
        :param source: The data source to extract content from.
        :return: The extracted content.
        """
        if isinstance(source, (str, Path)):
            with open(source, 'r', encoding='utf-8') as file:
                return file.read()
        if isinstance(source, ByteStream):
            return source.data.decode('utf-8')

        raise ValueError(f"Unsupported source type: {type(source)}")
