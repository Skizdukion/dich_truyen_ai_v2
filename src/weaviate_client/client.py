import os
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.collections import Collection
from .schema import (
    KnowledgeNode,
    create_or_get_collection,
)


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

    def delete_all_nodes(self):
        """Delete all nodes from the collection."""
        print("Deleting all nodes...")
        client = self.connect()
        client.collections.delete(self.collection_name or "KnowledgeNode")

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
            if obj:
                node_data = dict(obj.properties)  # type: ignore
                node_data['uuid'] = str(obj.uuid)  # Add UUID for consistency
                return node_data
            return None
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

        # Include both properties and UUID for consistency
        nodes = []
        for obj in response.objects:
            node_data = dict(obj.properties)  # type: ignore
            node_data['uuid'] = str(obj.uuid)  # Add UUID for consistency
            nodes.append(node_data)
        
        return nodes

    def delete_knowledge_node(self, node_id: str) -> bool:
        """Delete a knowledge node by ID."""
        collection = self.get_collection()

        try:
            collection.data.delete_by_id(node_id)
            return True
        except:
            return False

    def update_node(self, name: str, new_content: Dict[str, Any]) -> bool:
        """Update a knowledge node by name with new content."""
        collection = self.get_collection()

        try:
            # Find node by name
            nodes = self.query_knowledge_nodes({"name": name}, limit=1)
            if not nodes or not nodes[0].get('uuid'):
                print(f"Could not find node with name: {name}")
                return False
            
            node_id = str(nodes[0]['uuid'])
            
            # First, get the existing node to preserve unchanged fields
            existing_node = collection.query.fetch_object_by_id(node_id)
            if not existing_node:
                print(f"Node not found with ID: {node_id}")
                return False
            
            # Merge existing properties with new content
            updated_properties = dict(existing_node.properties)  # type: ignore
            updated_properties.update(new_content)
            
            # Update the node
            collection.data.update(
                uuid=node_id,
                properties=updated_properties
            )
            return True
        except Exception as e:
            print(f"Error updating node {name}: {str(e)}")
            return False

    def count_objects(self) -> int:
        """Count the total number of objects in the collection."""
        collection = self.get_collection()

        try:
            # Use aggregate to count all objects
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count or 0
        except Exception as e:
            print(f"Error counting objects: {str(e)}")
            return 0

    def search_nodes_by_text(
        self, query_text: str, node_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Semantic search for nodes by text similarity."""
        collection = self.get_collection()

        if node_type is not None:
            response = collection.query.near_text(
                query=query_text,
                limit=limit,
                filters=Filter.by_property("type").equal(node_type),
            )
        else:
            response = collection.query.near_text(query=query_text, limit=limit)

        # Include both properties and UUID for deduplication
        nodes = []
        for obj in response.objects:
            node_data = dict(obj.properties)  # type: ignore
            node_data['uuid'] = str(obj.uuid)  # Add UUID for deduplication
            nodes.append(node_data)
        
        return nodes

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

        # Include both properties and UUID for consistency
        nodes = []
        for obj in response.objects:
            node_data = dict(obj.properties)  # type: ignore
            node_data['uuid'] = str(obj.uuid)  # Add UUID for consistency
            nodes.append(node_data)
        
        return nodes

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
