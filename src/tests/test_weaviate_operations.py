#!/usr/bin/env python3
"""
Comprehensive test script for Weaviate client operations (tasks 3.4, 3.5, 3.6, 3.7)
"""

import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from weaviate_client.client import WeaviateWrapperClient
from weaviate_client.schema import KnowledgeNode
from weaviate_client.node_types import CHARACTER, TERM, ITEM, LOCATION, EVENT
import time
from typing import Optional, Dict, List, Any
from uuid import UUID
from weaviate.classes.query import Filter

# Test collection name
TEST_COLLECTION_NAME = "TestKnowledgeNode"


@pytest.fixture
def weaviate_client():
    """Fixture to provide a Weaviate client for tests"""
    client = WeaviateWrapperClient(collection_name=TEST_COLLECTION_NAME)
    yield client
    # Cleanup after tests
    try:
        delete_test_collection(client)
        client.close()
    except:
        pass


def delete_test_collection(wrapper: WeaviateWrapperClient) -> bool:
    """Delete the test collection."""
    try:
        if wrapper.client is not None:
            wrapper.client.collections.delete(TEST_COLLECTION_NAME)
            print(f"    Deleted test collection: {TEST_COLLECTION_NAME}")
            return True
    except Exception as e:
        print(f"    Collection deletion error: {e}")
    return False

def test_insert_operations(weaviate_client):
    """Test single and batch node insertion operations"""
    print("✓ Testing insert operations...")

    # Test single character insertion
    character_node: KnowledgeNode = {
        "type": CHARACTER,
        "label": "hero",
        "name": "Minh Vương",
        "content": "Minh Vương - Nhà vua xứ sở sương mù, kẻ sở hữu kỹ năng Hắc diệt đạo.",
        "alias": ["Minh Vương", "Nhà vua xứ sở sương mù"],
        "metadata": "test_data - Xuất hiện lần đầu trong chương 1",
    }

    character_id = weaviate_client.insert_knowledge_node(character_node)
    assert character_id is not None, "Character insertion failed"
    print(f"  Character inserted with ID: {character_id}")

    # Test batch insertion
    batch_nodes: List[KnowledgeNode] = [
        {
            "type": LOCATION,
            "label": "place",
            "name": "Xứ sở sương mù",
            "content": "Một vùng đất lạnh lẽo, đầy sương mù và linh hồn lang thang.",
            "alias": ["Xứ sở sương mù", "Vùng đất chết"],
            "metadata": "test_data - Địa điểm chính trong phần đầu truyện",
        },
        {
            "type": ITEM,
            "label": "weapon",
            "name": "Thiên Kiếm",
            "content": "Một thanh kiếm thánh chỉ người được chọn mới rút ra được.",
            "alias": ["Thiên Kiếm", "Kiếm của Thần"],
            "metadata": "test_data - Được cất giữ trong Tháp Ánh Sáng",
        },
    ]

    batch_ids = weaviate_client.batch_insert_nodes(batch_nodes)
    assert len(batch_ids) == len(
        batch_nodes
    ), f"Expected {len(batch_nodes)} IDs, got {len(batch_ids)}"
    print(f"  Batch inserted {len(batch_ids)} nodes successfully")


def test_retrieval_operations(weaviate_client):
    """Test retrieval operations"""
    print("✓ Testing retrieval operations...")

    # Insert test data
    character_node: KnowledgeNode = {
        "type": CHARACTER,
        "label": "hero",
        "name": "Test Character",
        "content": "A test character for retrieval testing.",
        "alias": ["Test Char", "TC"],
        "metadata": "test_data - retrieval test",
    }

    character_id = weaviate_client.insert_knowledge_node(character_node)

    # Test get by ID
    retrieved_character = weaviate_client.get_knowledge_node(str(character_id))
    assert retrieved_character is not None, "Character retrieval failed"
    assert (
        retrieved_character.get("name") == "Test Character"
    ), "Character name mismatch"
    print(f"  Get by ID successful: {retrieved_character.get('name')}")

    # Test query without filters
    all_nodes = weaviate_client.query_knowledge_nodes(limit=20)
    assert len(all_nodes) > 0, "Query should return some nodes"
    print(f"  Query without filters: {len(all_nodes)} nodes found")

    # Test query by type
    character_nodes = weaviate_client.query_knowledge_nodes(
        {"type": CHARACTER}, limit=10
    )
    print(f"  Query by type (CHARACTER): {len(character_nodes)} nodes found")


def test_vector_search_operations(weaviate_client):
    """Test vector search operations"""
    print("✓ Testing vector search operations...")

    # Insert test data
    character_node: KnowledgeNode = {
        "type": CHARACTER,
        "label": "hero",
        "name": "Search Test Character",
        "content": "A character for vector search testing with unique content.",
        "alias": ["STC", "Search Char"],
        "metadata": "test_data - vector search test",
    }

    character_id = weaviate_client.insert_knowledge_node(character_node)

    # Test semantic search by text
    search_results = weaviate_client.search_nodes_by_text(
        "Search Test Character", limit=5
    )
    assert len(search_results) > 0, "Semantic search should return results"
    print(
        f"  Semantic search for 'Search Test Character': {len(search_results)} results"
    )

    # Test vector embedding retrieval
    vector = weaviate_client.get_node_vector(str(character_id))
    assert vector is not None, "Vector embedding should be retrieved"
    assert len(vector) > 0, "Vector should have dimensions"
    print(f"  Vector embedding retrieved: {len(vector)} dimensions")


def test_deletion_operations(weaviate_client):
    """Test deletion operations"""
    print("✓ Testing deletion operations...")

    # Insert a test node
    test_node: KnowledgeNode = {
        "type": CHARACTER,
        "label": "test",
        "name": "Delete Test Character",
        "content": "A character for deletion testing.",
        "alias": ["DTC", "Delete Char"],
        "metadata": "test_data - deletion test",
    }

    test_id = weaviate_client.insert_knowledge_node(test_node)

    # Test deletion of existing node
    deleted = weaviate_client.delete_knowledge_node(str(test_id))
    assert deleted is True, "Deletion should succeed"
    print(f"  Deletion successful for character: {test_id}")

    # Verify node is deleted
    retrieved = weaviate_client.get_knowledge_node(str(test_id))
    assert retrieved is None, "Deleted node should not be retrievable"
    print("  Deleted node verification passed")


def test_weaviate_operations():
    """Comprehensive test of all Weaviate operations"""
    print("🧪 Starting comprehensive Weaviate client operations test...")
    print("=" * 60)

    # Initialize test client
    test_client = WeaviateWrapperClient()

    try:
        # Test 3.5: Insert operations
        test_insert_operations(test_client)

        # Test 3.6: Retrieval operations
        test_retrieval_operations(test_client)

        # Test 3.7: Vector search operations
        test_vector_search_operations(test_client)

        # Test deletion operations
        test_deletion_operations(test_client)

        # Final cleanup
        print("✓ Testing final cleanup operations...")
        delete_test_collection(test_client)

        # Close connection
        test_client.close()

        assert True  # Test passed

    except Exception as e:
        print(f"\n❌ Error during comprehensive testing: {e}")
        import traceback

        traceback.print_exc()
        # Ensure cleanup happens even on failure
        try:
            delete_test_collection(test_client)
            test_client.close()
        except:
            pass
        assert False, f"Comprehensive test failed: {e}"


if __name__ == "__main__":
    test_weaviate_operations()
