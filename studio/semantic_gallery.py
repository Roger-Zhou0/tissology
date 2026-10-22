"""Label-independent development gallery; an unvalidated workflow aid."""
import numpy as np


def representative_ids(ids, features, count=20):
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError('Gallery count must be between 1 and 20')
    values = np.asarray(features, dtype=np.float64)
    if (not ids or len(set(ids)) != len(ids) or values.ndim != 2 or
            values.shape[0] != len(ids) or not np.isfinite(values).all() or
            not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)):
        raise ValueError('Gallery requires unique IDs and normalized finite features')
    # Canonical ID order resolves all ties without catalog order or class labels.
    order = np.argsort(np.asarray(ids), kind='stable')
    values = values[order]
    ordered_ids = [ids[i] for i in order]
    first = int(np.argmax(values @ values.mean(axis=0)))
    selected = [first]
    similarity = values @ values[first]
    while len(selected) < min(count, len(ids)):
        similarity[selected] = np.inf
        following = int(np.argmin(similarity))
        selected.append(following)
        similarity = np.maximum(similarity, values @ values[following])
    return [ordered_ids[i] for i in selected]


def gallery(store):
    if any(store.row(identifier)['role'] != 'train' for identifier in store.ids):
        raise ValueError('Gallery accepts development training images only')
    ids = representative_ids(store.ids, store.features)
    return {'schema':'semantic-gallery-v1', 'catalog_sha256':store.catalog['sha256'],
            'selection':'Nearest feature-space mean, then farthest-point cosine coverage; ID-sorted ties. No class labels used.',
            'scope':'Development tiles only. Varied visual examples are an unvalidated workflow aid, not confirmed tissue concepts.',
            'results':[{'id':identifier, 'image_url':'api/semantic/images/'+identifier} for identifier in ids]}
