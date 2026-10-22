"""Preliminary whole-tile CRC scanning with the frozen deployed CONCH classifier.

Only ``configure`` is suitable for a login node. ``run`` requires the configured
Slurm CPU or legacy GPU allocation; it never downloads weights, fits a model,
or uses collection reference labels.
The two cached feature spaces are deliberately distinct.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import time
import uuid

from .storage import digest, now, read, write
from .execution import execution_backend, execution_provenance

ARTIFACT_ROOT = Path(os.environ.get('TISSUE_LEGACY_ARTIFACT_ROOT', './artifacts')).resolve()
# Immutable original replay identity, also used by collection teaching. New
# prediction releases have their own identities; never repin this constant.
CLASSIFIER_SHA256 = 'c55b4f70800235d2a7882099c877de8a0c76e61bceb523f2bb426196e73b7dce'
CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
SCORE_TYPE = 'Uncalibrated multinomial classifier scores; not verified probabilities of correctness'
SCOPE = ('Preliminary assignment of one dominant colorectal H&E tissue category per image. '
         'Validated task uses 224 × 224 RGB colorectal H&E tiles. No cell counts, '
         'object locations, segmentation, diagnosis, or pixel area measurement. '
         'Every image receives one of nine known classes; unfamiliar content can receive a high score.')
REVIEW_RULE = ('Review suggestion only: top score below 0.8, top-two score margin below 0.2, '
               'or native dimensions/color mode outside the validated tile format. '
               'This heuristic is not calibrated and does not detect out-of-domain images.')


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_frozen(value, name):
    if not isinstance(value, dict) or not value.get('sha256') or value['sha256'] != digest(
            {key: item for key, item in value.items() if key != 'sha256'}):
        raise ValueError(f'{name} differs from its frozen hash')


def freeze(path, value):
    artifact = {**value, 'sha256': digest(value)}
    write(path, artifact)
    return artifact


def configure(root, release_dir=None):
    """Create the administrator-owned configuration, without opening model weights."""
    from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, REVISION
    root = Path(root).resolve()
    config_path = root / 'scan-model.json'
    if config_path.exists() and release_dir is None:
        config = read(config_path)
        validate_config(config)
        return config
    if release_dir is not None:
        from .model_releases import load_release
        release_dir = Path(release_dir).resolve()
        release, _, _, _ = load_release(release_dir)
        if config_path.exists():
            previous = snapshot_config(root)
            if previous.get('release_sha256') == release['sha256']:
                return previous
        config = {
            'schema_version': 2, 'created_at': now(), 'backend': 'cpu', 'name': release['name'],
            'release_directory': str(release_dir), 'release_sha256': release['sha256'],
            'classifier_sha256': release['sha256'], 'coefficients_sha256': release['coefficients_sha256'],
            'checkpoint': str(ARTIFACT_ROOT / 'hf-gated/hub/models--MahmoodLab--CONCH' /
                              'snapshots' / REVISION / 'pytorch_model.bin'),
            'checkpoint_sha256': CHECKPOINT_SHA256,
            'code_directory': str(ARTIFACT_ROOT / 'vendor/CONCH'), 'code_revision': CODE_REVISION,
            'model_revision': REVISION, 'classes': CLASSES, 'batch_size': 4,
            'score_threshold': 0.8, 'margin_threshold': 0.2, 'scope': SCOPE,
            'review_rule': REVIEW_RULE, 'score_type': SCORE_TYPE,
        }
        config['sha256'] = digest(config)
        snapshot_config(root, config)
        write(config_path, config, overwrite=True)
        return config
    classifier_dir = ARTIFACT_ROOT / 'conch-experiment-001'
    classifier = read(classifier_dir / 'classifier.json')
    verify_frozen(classifier, 'Classifier')
    if classifier['sha256'] != CLASSIFIER_SHA256:
        raise ValueError('Configured classifier differs from the deployed classifier')
    config = {
        'schema_version': 1, 'created_at': now(), 'name': 'CONCH + frozen CRC tissue classifier',
        'classifier_directory': str(classifier_dir), 'classifier_sha256': CLASSIFIER_SHA256,
        'coefficients_sha256': classifier['coefficients_sha256'],
        'checkpoint': str(ARTIFACT_ROOT / 'hf-gated/hub/models--MahmoodLab--CONCH' /
                          'snapshots' / REVISION / 'pytorch_model.bin'),
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'code_directory': str(ARTIFACT_ROOT / 'vendor/CONCH'), 'code_revision': CODE_REVISION,
        'model_revision': REVISION, 'classes': CLASSES, 'batch_size': 16,
        'score_threshold': 0.8, 'margin_threshold': 0.2, 'scope': SCOPE,
        'review_rule': REVIEW_RULE, 'score_type': SCORE_TYPE,
    }
    return freeze(config_path, config)


def validate_config(config):
    from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, REVISION
    verify_frozen(config, 'Scan model configuration')
    expected = {'checkpoint_sha256': CHECKPOINT_SHA256,
                'code_revision': CODE_REVISION, 'model_revision': REVISION, 'classes': CLASSES,
                'score_threshold': 0.8, 'margin_threshold': 0.2}
    paths = ['checkpoint', 'code_directory']
    if config.get('schema_version') == 1:
        expected.update(classifier_sha256=CLASSIFIER_SHA256, batch_size=16)
        paths.append('classifier_directory')
        if config.get('backend', 'cuda') != 'cuda':
            raise ValueError('Legacy scan configuration requires its original CUDA backend')
    elif config.get('schema_version') == 2:
        expected.update(backend='cpu', batch_size=4)
        paths.append('release_directory')
        if (not re.fullmatch('[0-9a-f]{64}', config.get('release_sha256', '')) or
                config.get('classifier_sha256') != config['release_sha256']):
            raise ValueError('Scan configuration requires its pinned release identity')
    else:
        raise ValueError('Unsupported scan configuration version')
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Scan configuration does not identify the approved frozen model and review rule')
    for key in paths:
        if not Path(config.get(key, '')).is_absolute():
            raise ValueError('Scan model paths must be absolute administrator configuration')
    if not re.fullmatch('[0-9a-f]{64}', config.get('coefficients_sha256', '')):
        raise ValueError('Scan configuration requires a coefficient checksum')
    if config['schema_version'] == 2:
        from .model_releases import load_release
        release, _, _, _ = load_release(config['release_directory'], config['release_sha256'])
        if config['coefficients_sha256'] != release['coefficients_sha256']:
            raise ValueError('Scan coefficients differ from the pinned release')


def snapshot_config(root, config=None):
    """Freeze the configuration selected at submission; never overwrite a snapshot."""
    root = Path(root).resolve()
    config = read(root / 'scan-model.json') if config is None else config
    validate_config(config)
    directory = root / 'scan-models'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (config['sha256'] + '.json')
    if path.is_symlink():
        raise ValueError('Configuration snapshots must be regular files')
    if not path.exists():
        try:
            publish_cache_file(path, lambda handle: handle.write(
                (json.dumps(config, indent=2, allow_nan=False) + '\n').encode()))
        except FileExistsError:
            pass  # Another submitter may have published the same immutable file.
    if read(path) != config:
        raise ValueError('Existing configuration snapshot differs')
    return config


def load_configuration(root, configuration_sha=None):
    root = Path(root).resolve()
    if configuration_sha is not None:
        if not re.fullmatch('[0-9a-f]{64}', configuration_sha):
            raise ValueError('Invalid configuration snapshot identity')
        path = root / 'scan-models' / (configuration_sha + '.json')
    else:
        path = root / 'scan-model.json'
    if path.is_symlink():
        raise ValueError('Scan configurations must be regular files')
    config = read(path)
    validate_config(config)
    if configuration_sha is not None and config['sha256'] != configuration_sha:
        raise ValueError('Configuration snapshot differs from requested identity')
    return config, path


def checked_image_path(row, collection_dir):
    """Never infer from paths outside the immutable imported collection."""
    path = Path(row['path']).resolve()
    if not path.is_relative_to(Path(collection_dir).resolve()) or not path.is_file():
        raise ValueError('Imported image is missing or outside its collection')
    image_hash = row.get('image_sha256')
    if not isinstance(image_hash, str) or not re.fullmatch('[0-9a-f]{64}', image_hash):
        raise ValueError('Imported image requires its canonical image hash')
    if file_hash(path) != image_hash:
        raise ValueError('Imported image changed after collection freeze')
    if 'source_path' in row:
        source = Path(row['source_path']).resolve()
        if (not source.is_relative_to(Path(collection_dir).resolve()) or not source.is_file() or
                file_hash(source) != row['sha256']):
            raise ValueError('Original imported source changed after collection freeze')
    return path


def image_binding(rows):
    return [{'id': row['id'], 'source_sha256': row['sha256'],
             'image_sha256': row['image_sha256'], 'size_px': row['size_px']} for row in rows]


def score_features(features, coefficients, intercept):
    """Exactly the saved multinomial logistic decision rule; no refit or calibration."""
    import numpy as np
    features = np.asarray(features)
    coefficients, intercept = np.asarray(coefficients), np.asarray(intercept)
    if (features.ndim != 2 or features.shape[1] != 512 or
            coefficients.shape != (9, 512) or intercept.shape != (9,) or
            not all(np.isfinite(item).all() for item in (features, coefficients, intercept)) or
            not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4)):
        raise ValueError('Invalid normalized classifier features or frozen coefficients')
    logits = features @ coefficients.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores


def scan_runtime_metadata(encoder_metadata):
    """Describe the applied classifier, without changing encoder identity fields."""
    if encoder_metadata is None:
        return None  # All features reused; no encoder was loaded in this run.
    return {
        **encoder_metadata,
        'classification': ('Frozen multinomial logistic classifier on L2-normalized '
                           'unprojected CONCH pooled image features; argmax of softmax scores'),
        'score_interpretation': SCORE_TYPE,
    }


def prediction_records(rows, scores):
    import numpy as np
    scores = np.asarray(scores)
    if (scores.shape != (len(rows), len(CLASSES)) or not np.isfinite(scores).all() or
            (scores < 0).any() or (scores > 1).any() or
            not np.allclose(scores.sum(axis=1), 1, atol=1e-7)):
        raise ValueError('Invalid classifier scores')
    results = []
    for row, values in zip(rows, scores):
        order = np.argsort(values)[::-1]
        top_score, margin = float(values[order[0]]), float(values[order[0]] - values[order[1]])
        warnings, reasons = [], []
        if list(row['size_px']) != [224, 224]:
            warnings.append('Native image size differs from the validated 224 × 224 tile task. '
                            'CONCH resizes and center-crops; this is not analysis of every part of a larger image.')
        native_mode = row.get('original_mode', row.get('mode', 'RGB'))
        if native_mode != 'RGB':
            warnings.append('Native color mode differs from the validated RGB tile task; input was converted to RGB.')
        if top_score < 0.8:
            reasons.append('top score below 0.8')
        if margin < 0.2:
            reasons.append('top-two score margin below 0.2')
        if warnings:
            reasons.append('native format outside the validated tile task')
        results.append({
            'id': row['id'], 'predicted_label': CLASSES[int(order[0])],
            'top_k': [{'label': CLASSES[int(index)], 'score': float(values[index])} for index in order[:3]],
            'classification_scores': dict(zip(CLASSES, values.tolist())),
            'confidence': top_score, 'model_score': top_score, 'margin': margin,
            'uncertain': bool(reasons), 'needs_review': bool(reasons), 'review_reasons': reasons,
            'scope_warnings': warnings, 'source_sha256': row['sha256'],
            'image_sha256': row['image_sha256'], 'score_type': SCORE_TYPE,
        })
    return results


def publish_cache_file(path, writer):
    """Publish a fully flushed cache file without replacing an existing artifact."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '-')
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Unlike replace(), link() fails if another artifact already owns the name.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_chunk(directory, index, rows, classifier_features, retrieval_features, binding):
    import numpy as np
    path = directory / f'batch-{index:06d}.npz'
    publish_cache_file(path, lambda handle: np.savez_compressed(
        handle, ids=np.array([row['id'] for row in rows]),
        classifier_embeddings=classifier_features, embeddings=retrieval_features))
    values = {
        'schema_version': 1, 'binding': binding, 'images': image_binding(rows),
        'file_sha256': file_hash(path), 'file': path.name,
    }
    artifact = {**values, 'sha256': digest(values)}
    publish_cache_file(path.with_suffix('.json'), lambda handle: handle.write(
        (json.dumps(artifact, indent=2, allow_nan=False) + '\n').encode()))
    return artifact


def quarantine_incomplete_chunk(path, meta_path):
    """Preserve an interrupted pair for inspection, then permit re-encoding."""
    recovery = path.parent / 'recovered' / (path.stem + '-' + uuid.uuid4().hex)
    recovery.mkdir(parents=True)
    files = []
    for item in (path, meta_path):
        if item.exists():
            files.append({'file': item.name, 'sha256': file_hash(item)})
    # Record intent before moving anything; complete but invalid pairs still fail
    # the strict integrity checks below and are never automatically discarded.
    freeze(recovery / 'recovery.json', {'created_at': now(), 'files': files,
           'reason': 'Incomplete cache pair after interruption; preserved and re-encoded'})
    for item in (path, meta_path):
        if item.exists():
            item.rename(recovery / item.name)
    print(f'Preserved incomplete feature cache in {recovery.name}; re-encoding batch', flush=True)


def load_chunk(directory, index, rows, binding):
    import numpy as np
    path = directory / f'batch-{index:06d}.npz'
    meta_path = path.with_suffix('.json')
    if not path.exists() and not meta_path.exists():
        return None
    # A crash between publishing data and metadata cannot establish a binding.
    # Preserve it with a receipt and recompute this batch; never reuse its values.
    if not path.exists() or not meta_path.exists():
        quarantine_incomplete_chunk(path, meta_path)
        return None
    meta = read(meta_path)
    verify_frozen(meta, 'Cached feature metadata')
    if (meta.get('binding') != binding or meta.get('images') != image_binding(rows) or
            meta.get('file') != path.name or meta.get('file_sha256') != file_hash(path)):
        raise ValueError('Feature cache differs from this frozen collection and model')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'ids', 'classifier_embeddings', 'embeddings'}:
            raise ValueError('Unexpected cached feature arrays')
        if arrays['ids'].tolist() != [row['id'] for row in rows]:
            raise ValueError('Cached feature identities differ')
        features = {key: arrays[key].copy() for key in ('classifier_embeddings', 'embeddings')}
    for value in features.values():
        if (value.shape != (len(rows), 512) or value.dtype != np.float32 or
                not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-4)):
            raise ValueError('Invalid cached normalized feature values')
    return features['classifier_embeddings'], features['embeddings'], meta


def run(root, collection_id, configuration_sha=None):
    if execution_backend() == 'slurm' and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Collection inference must run in a Slurm GPU or CPU allocation')
    if not re.fullmatch('[0-9a-f]{32}', collection_id):
        raise ValueError('Invalid collection identifier')
    root = Path(root).resolve()
    config, configuration_path = load_configuration(root, configuration_sha)
    cpu = config.get('backend') == 'cpu'
    if cpu:
        from .cpu_features import require_cpu_allocation
        threads = require_cpu_allocation()
    elif execution_backend() != 'slurm' or os.environ.get('SLURM_JOB_ACCOUNT') != 'bfqi-delta-gpu':
        raise RuntimeError('Legacy GPU collection inference requires account bfqi-delta-gpu')
    import torch
    if cpu:
        torch.set_num_threads(threads)
    elif not torch.cuda.is_available():
        raise RuntimeError('Collection inference requires one allocated CUDA GPU')
    from .collection_storage import CollectionStore
    from .improve import encoder_identity
    from PIL import Image

    collection_dir = root / collection_id
    store = CollectionStore(root)
    model_record = {'name': config['name'], 'classifier_sha256': config['classifier_sha256'],
                    'configuration_sha256': config['sha256'], 'scope': SCOPE,
                    'review_rule': REVIEW_RULE, 'score_type': SCORE_TYPE,
                    'backend': 'cpu' if cpu else 'cuda',
                    **execution_provenance()}
    store.claim_scan(collection_id, model=model_record)
    started = time.monotonic()
    try:
        manifest = store.manifest(collection_id)
        verify_frozen(manifest, 'Collection manifest')
        rows = manifest['images']
        if not rows or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Collection must contain unique image identities')
        if cpu:
            from .model_releases import load_release
            classifier, identity, coefficients, intercept = load_release(
                config['release_directory'], config['release_sha256'])
            model_record.update(release_id=classifier['release_id'], release_sha256=classifier['sha256'],
                                release_directory=config['release_directory'],
                                validation_result_sha256=classifier['validation']['result_sha256'])
        else:
            from .testsets import _classifier
            classifier, identity, coefficients, intercept = _classifier(config['classifier_directory'])
        if (classifier['sha256'] != config['classifier_sha256'] or
                classifier['coefficients_sha256'] != config['coefficients_sha256'] or
                classifier.get('family', 'multinomial_logistic') != 'multinomial_logistic' or
                classifier.get('feature_key') != 'classifier_embeddings'):
            raise ValueError('Classifier does not match the frozen scan configuration')
        binding = {'manifest_sha256': manifest['sha256'], 'configuration_sha256': config['sha256'],
                   'encoder_identity_sha256': digest(identity), 'worker_source_sha256': file_hash(__file__)}
        directory = collection_dir / 'features'
        directory.mkdir(exist_ok=True)
        predictions, chunks, reused, encoder = [], [], 0, None
        metadata = None
        projection_error = None
        encoding_seconds = 0.0
        for offset in range(0, len(rows), config['batch_size']):
            batch_rows = rows[offset:offset + config['batch_size']]
            paths = [checked_image_path(row, collection_dir) for row in batch_rows]
            cache = load_chunk(directory, offset // config['batch_size'], batch_rows, binding)
            if cache is None:
                if encoder is None:
                    if cpu:
                        from .cpu_features import load_cpu_encoder
                        encoder = load_cpu_encoder(config['checkpoint'], config['code_directory'],
                                                   config['checkpoint_sha256'], threads=threads)
                        model, preprocess, metadata = encoder
                    else:
                        from .conch_infer import load_encoder
                        encoder = load_encoder(config['checkpoint'], config['code_directory'], config['checkpoint_sha256'])
                        model, preprocess, _, _, metadata = encoder
                    if encoder_identity(metadata) != identity:
                        raise ValueError('Current encoder/preprocessing differs from classifier training')
                tensors = []
                for row, path in zip(batch_rows, paths):
                    with Image.open(path) as image:
                        if image.mode != 'RGB' or list(image.size) != list(row['size_px']):
                            raise ValueError('Canonical image dimensions or mode differ from the manifest')
                        tensors.append(preprocess(image))
                if not cpu:
                    torch.cuda.synchronize()
                encode_start = time.monotonic()
                with torch.inference_mode():
                    batch = torch.stack(tensors).to(device='cpu' if cpu else 'cuda', dtype=torch.float32)
                    raw = model.encode_image(batch, proj_contrast=False, normalize=False)
                    if (raw.device.type != ('cpu' if cpu else 'cuda') or raw.shape != (len(batch_rows), 512) or
                            not torch.isfinite(raw).all() or (raw.norm(dim=-1) == 0).any()):
                        raise ValueError('Unexpected CONCH pooled image features')
                    pooled = torch.nn.functional.normalize(raw, dim=-1)
                    projected = torch.nn.functional.normalize(raw @ model.visual.proj_contrast, dim=-1)
                    if projection_error is None:
                        official = model.encode_image(batch[:1], proj_contrast=True, normalize=True)
                        projection_error = float((official - projected[:1]).abs().max().cpu())
                        if not torch.allclose(official, projected[:1], atol=1e-5, rtol=1e-4):
                            raise ValueError('Retrieval projection differs from the official encoder')
                    classifier_features = pooled.cpu().numpy()
                    retrieval_features = projected.cpu().numpy()
                if not cpu:
                    torch.cuda.synchronize()
                encoding_seconds += time.monotonic() - encode_start
                if not torch.isfinite(projected).all():
                    raise ValueError('Nonfinite retrieval features')
                chunk = save_chunk(directory, offset // config['batch_size'], batch_rows,
                                   classifier_features, retrieval_features, binding)
            else:
                classifier_features, retrieval_features, chunk = cache
                reused += len(batch_rows)
            values = score_features(classifier_features, coefficients, intercept)
            predictions.extend(prediction_records(batch_rows, values))
            chunks.append({'file': chunk['file'], 'sha256': chunk['file_sha256'],
                           'metadata_sha256': chunk['sha256']})
            store.scan_progress(collection_id, len(predictions))
            print(f'Collection {collection_id}: {len(predictions)}/{len(rows)} images', flush=True)
        if (store.manifest(collection_id) != manifest or read(configuration_path) != config or
                file_hash(__file__) != binding['worker_source_sha256']):
            raise ValueError('Collection or model configuration changed during inference')
        # Imported source files are immutable through the API. Verify their
        # canonical hashes again to catch filesystem edits during a long scan.
        for row in rows:
            checked_image_path(row, collection_dir)
        if cpu:
            from .cpu_features import _verify_cpu_model
            if encoder is not None:
                _verify_cpu_model(model)
            # Revalidate the actual copied release, never a mutable source head.
            load_release(config['release_directory'], config['release_sha256'])
        model_record.update({
            'manifest_sha256': manifest['sha256'], 'encoder_identity': identity,
            'checkpoint_sha256': config['checkpoint_sha256'], 'code_revision': config['code_revision'],
            'coefficients_sha256': config['coefficients_sha256'], 'worker_source_sha256': binding['worker_source_sha256'],
            'host': socket.gethostname(), 'device': 'cpu' if cpu else torch.cuda.get_device_name(),
            'cpu_threads': threads if cpu else None, 'completed_at': now(),
            'elapsed_seconds': time.monotonic() - started, 'encoding_seconds': encoding_seconds,
            'cached_images_reused': reused, 'newly_encoded_images': len(rows) - reused,
            'batch_size': config['batch_size'], 'feature_chunks': chunks,
            'classifier_representation': classifier['representation'],
            'retrieval_representation': 'L2-normalized projected CONCH image/text features',
            'projection_check_max_error': projection_error,
            'encoder_runtime': scan_runtime_metadata(metadata),
            'reference_labels_used': False, 'model_training': False,
        })
        store.finish_scan(collection_id, predictions=predictions, model=model_record, score_type=SCORE_TYPE)
        print(json.dumps({'collection_id': collection_id, 'images': len(predictions),
                          'cached_images_reused': reused, 'elapsed_seconds': model_record['elapsed_seconds'],
                          'encoding_seconds': encoding_seconds}), flush=True)
        return model_record
    except Exception as error:
        store.fail_scan(collection_id, str(error)[:600])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('configure')
    setup.add_argument('--root', required=True)
    setup.add_argument('--release-dir')
    scan = sub.add_parser('run')
    scan.add_argument('--root', required=True)
    scan.add_argument('--collection-id', required=True)
    scan.add_argument('--configuration-sha')
    args = parser.parse_args()
    if args.command == 'configure':
        config = configure(args.root, release_dir=args.release_dir)
        print(json.dumps({'configuration_sha256': config['sha256'], 'classifier_sha256': config['classifier_sha256']}))
    else:
        run(args.root, args.collection_id, configuration_sha=args.configuration_sha)


if __name__ == '__main__':
    main()
