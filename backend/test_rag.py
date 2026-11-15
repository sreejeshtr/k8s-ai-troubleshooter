from retriever import retrieve_context

context, sources = retrieve_context(
    "ImagePullBackOff"
)

print("\nSOURCES:")
print(sources)

print("\nCONTEXT:")
print(context)
