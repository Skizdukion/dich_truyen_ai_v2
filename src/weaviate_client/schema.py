from typing import TypedDict, Optional, List, Any

from weaviate import WeaviateClient, Collection
from weaviate.classes.config import Configure, DataType, Property
from .node_types import ALL_NODE_TYPES


class KnowledgeNode(TypedDict, total=False):
    type: str
    label: str
    name: str
    content: str
    alias: Optional[List[str]]
    metadata: Optional[str]


def validate_knowledge_node(node: dict) -> List[str]:
    errors = []
    if "type" not in node or node["type"] not in ALL_NODE_TYPES:
        errors.append(f"Missing or invalid 'type': {node.get('type')}")
    if "name" not in node or not isinstance(node["name"], str):
        errors.append("Missing or invalid 'name'")
    if "content" not in node or not isinstance(node["content"], str):
        errors.append("Missing or invalid 'content'")
    # Add more validation as needed
    return errors


def create_or_get_collection(client: WeaviateClient, name) -> Collection:
    print(f"Creating or getting collection {name}")
    if client is not None:
        collection = client.collections.get(name)
        
        if collection.exists():
            return collection
        else:
            print(f"Creating collection {name}")
            return client.collections.create(
                name,
                description="A node representing a chunk, translation, memory, etc.",
                vectorizer_config=[
                    Configure.NamedVectors.text2vec_google_aistudio(
                        name="text_vector",
                        source_properties=["text"],
                        model_id="text-embedding-004",  # Google AI Studio model
                    )
                ],
                properties=[
                    Property(
                        name="type",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="label",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="name",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="alias",
                        data_type=DataType.TEXT_ARRAY,
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.TEXT,
                    ),
                ],
            )
