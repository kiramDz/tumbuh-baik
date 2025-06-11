from bson import ObjectId

def convert_objectid(doc):
    if isinstance(doc, list):
        return [convert_objectid(d) for d in doc]
    elif isinstance(doc, dict):
        return {k: (str(v) if isinstance(v, ObjectId) else convert_objectid(v)) for k, v in doc.items()}
    else:
        return doc
