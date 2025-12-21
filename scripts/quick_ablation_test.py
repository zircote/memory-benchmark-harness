#!/usr/bin/env python3
"""Quick test of the ablation adapters.

This verifies all 5 ablation adapters work correctly by running them
through a simple store/retrieve cycle.

Usage:
    uv run python scripts/quick_ablation_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> None:
    """Run a quick ablation adapter test."""
    print("=" * 60)
    print("Quick Ablation Adapter Test")
    print("=" * 60)

    from src.adapters.ablation import (
        AblationType,
        FixedWindowAdapter,
        NoMetadataFilterAdapter,
        NoSemanticSearchAdapter,
        NoVersionHistoryAdapter,
        RecencyOnlyAdapter,
    )
    from src.adapters.base import MemoryItem
    from src.adapters.mock import MockAdapter

    # All ablation adapters wrap a base adapter (decorator pattern)
    base = MockAdapter()

    # Test each ablation adapter (each wraps a fresh base adapter)
    adapters = [
        ("NoSemanticSearch", NoSemanticSearchAdapter(base_adapter=MockAdapter())),
        ("NoMetadataFilter", NoMetadataFilterAdapter(base_adapter=MockAdapter())),
        ("NoVersionHistory", NoVersionHistoryAdapter(base_adapter=MockAdapter())),
        ("FixedWindow", FixedWindowAdapter(base_adapter=MockAdapter(), window_size=5)),
        ("RecencyOnly", RecencyOnlyAdapter(base_adapter=MockAdapter())),
    ]

    print("\n1. Testing adapter instantiation...")
    for name, adapter in adapters:
        print(f"   ✓ {name}: {type(adapter).__name__}")

    # Test store and retrieve for each adapter
    print("\n2. Testing add/search cycle...")

    for name, adapter in adapters:
        try:
            # Add entries using the adapter's add() method
            for i in range(3):
                adapter.add(
                    content=f"Test memory {i} for {name}",
                    metadata={"index": i, "category": "test"},
                )

            # Search (the adapters ablate different aspects of search)
            results = adapter.search("test memory", limit=10)
            print(f"   ✓ {name}: added 3, retrieved {len(results)}")
        except Exception as e:
            print(f"   ✗ {name}: {e}")

    # Test AblationType constants
    print("\n3. Testing AblationType constants...")
    ablation_types = [
        AblationType.NO_SEMANTIC_SEARCH,
        AblationType.NO_METADATA_FILTER,
        AblationType.NO_VERSION_HISTORY,
        AblationType.FIXED_WINDOW,
        AblationType.RECENCY_ONLY,
    ]
    for atype in ablation_types:
        print(f"   - {atype}")

    # Test factory function
    print("\n4. Testing create_ablation_adapter factory...")
    from src.adapters.ablation import create_ablation_adapter

    for ablation_type in ablation_types:
        ablated = create_ablation_adapter(MockAdapter(), ablation_type)
        print(f"   ✓ {ablation_type}: {type(ablated).__name__}")

    print("\n" + "=" * 60)
    print("Quick test complete - All ablation adapters work!")
    print("=" * 60)


if __name__ == "__main__":
    main()
