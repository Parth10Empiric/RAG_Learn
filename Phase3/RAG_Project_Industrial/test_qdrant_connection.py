from .store import client, COLLECTION_NAME


print("Testing Qdrant connection...")


collections = client.get_collections()


print("Connection successful!")

print("\nAvailable collections:")

for collection in collections.collections:
    print("-", collection.name)


print(
    f"\nTarget collection: {COLLECTION_NAME}"
)