import os
from typing import List, Optional, Dict, Any
from uuid import UUID
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.collections import Collection
from .schema import KnowledgeNode, create_or_get_collection


class WeaviateWrapperClient:
    def __init__(
        self,
        url: Optional[str] = None,
        weaviate_key: Optional[str] = None,
        studio_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.weaviate_key = weaviate_key or os.getenv("WEAVIATE_API_KEY")
        self.studio_key = studio_key or os.getenv("STUDIO_APIKEY")
        self.client = None
        self.collection_name = collection_name

    def connect(self):
        if self.client is not None:
            return self.client

        # Headers for Google AI Studio
        headers = {}
        if self.studio_key:
            headers["X-Goog-Studio-Api-Key"] = self.studio_key

        # Connect to Weaviate
        if self.weaviate_key:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.weaviate_key),
                headers=headers,
            )
        else:
            # For local development without auth
            self.client = weaviate.connect_to_local(
                headers=headers,
                skip_init_checks=True,  # Skip gRPC health checks for local testing
            )

        return self.client

    def get_collection(self) -> Collection:
        """Create the KnowledgeNode collection with Google AI Studio vectorizer."""
        client = self.connect()
        return create_or_get_collection(client, self.collection_name or "KnowledgeNode")

    def insert_knowledge_node(self, node: KnowledgeNode) -> UUID:
        """Insert a single knowledge node and return the generated UUID."""
        collection = self.get_collection()

        # Insert object and get the generated UUID
        result = collection.data.insert(properties=node)  # type: ignore
        return result

    def get_knowledge_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a knowledge node by ID."""
        collection = self.get_collection()

        try:
            obj = collection.query.fetch_object_by_id(node_id)
            return dict(obj.properties) if obj else None  # type: ignore
        except:
            return None

    def query_knowledge_nodes(
        self, filters: Optional[Dict[str, Any]] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query knowledge nodes with filters."""
        collection = self.get_collection()

        # Build filter
        where_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(Filter.by_property(key).equal(value))
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = Filter.and_(*conditions)  # type: ignore

        # Execute query - simplified for v4
        response = collection.query.fetch_objects(limit=limit, filters=where_filter)

        return [dict(obj.properties) for obj in response.objects]  # type: ignore

    def delete_knowledge_node(self, node_id: str) -> bool:
        """Delete a knowledge node by ID."""
        collection = self.get_collection()

        try:
            collection.data.delete_by_id(node_id)
            return True
        except:
            return False

    def search_nodes_by_text(
        self, query_text: str, node_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Semantic search for nodes by text similarity."""
        collection = self.get_collection()

        if node_type is not None:
            response = collection.query.near_text(
                query=query_text, limit=limit, filters=Filter.by_property("type").equal(node_type)
            )
        else:
            response = collection.query.near_text(query=query_text, limit=limit)

        return [dict(obj.properties) for obj in response.objects]  # type: ignore

    def search_nodes_by_vector(
        self, vector: List[float], node_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for nodes by vector similarity."""
        collection = self.get_collection()

        # Build filter
        where_filter = None
        if node_type:
            where_filter = Filter.by_property("type").equal(node_type)

        # Perform vector search - simplified for v4
        response = collection.query.near_vector(
            near_vector=vector, limit=limit, filters=where_filter
        )

        return [dict(obj.properties) for obj in response.objects]  # type: ignore

    def get_node_vector(self, node_id: str) -> Optional[List[float]]:
        """Get the vector embedding for a specific node."""
        collection = self.get_collection()

        try:
            response = collection.query.fetch_object_by_id(node_id, include_vector=True)
            return response.vector if response else None  # type: ignore
        except:
            return None

    def batch_insert_nodes(self, nodes: List[KnowledgeNode]) -> List[UUID]:
        """Batch insert multiple knowledge nodes and return their UUIDs."""
        collection = self.get_collection()
        uuids = []

        with collection.batch.fixed_size(batch_size=100) as batch:
            for node in nodes:
                result = batch.add_object(properties=node)  # type: ignore
                uuids.append(result)

        return uuids

    def close(self):
        """Close the Weaviate connection."""
        if self.client:
            self.client.close()
