from qdrant_client import models

from .store import client, COLLECTION_NAME


stale, offset = [], None

while True:
    batch, offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=500,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    stale += [
        point.id
        for point in batch
        if "document_id" not in point.payload
    ]

    if offset is None:
        break


print(f"Found {len(stale)} stale points (missing document_id).")


if stale:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=stale),
        wait=True,
    )
    print("Deleted.")


print("Remaining:", client.count(collection_name=COLLECTION_NAME).count)
