"""Portable CPU workspace; model artifacts remain administrator-provisioned.

This entrypoint never downloads weights or changes historical study records.
It can start without Delta datasets, and exposes only configured capabilities.
"""
import argparse
import asyncio
import base64
import binascii
import hmac
import json
import os
from pathlib import Path
import shutil
import threading
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse

from .storage import digest, now, read, write


class AccessGate:
    """One private shared workspace, or explicitly public read-only viewing."""
    def __init__(self, app, *, password, readonly, data_dir, public_origin=None):
        self.app, self.password, self.readonly = app, password, readonly
        self.data_dir = Path(data_dir)
        self.origin = public_origin.rstrip('/') if public_origin else None
        self.buffering = 0
        if not readonly and (not password or len(password) < 16):
            raise ValueError('Set TISSUE_ACCESS_PASSWORD to at least 16 characters, or explicitly select TISSUE_PUBLIC_READONLY=1')
        if self.origin:
            parsed = urlsplit(self.origin)
            if (parsed.scheme not in {'https', 'http'} or not parsed.netloc or
                    parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password):
                raise ValueError('TISSUE_PUBLIC_ORIGIN must be an HTTP(S) origin without a path or credentials')

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] == '/healthz':
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get('headers', [])}
        method = scope['method']

        async def reject(status, detail, **extra):
            return await JSONResponse({'detail': detail}, status_code=status,
                headers={'Cache-Control': 'no-store', **extra})(scope, receive, send)

        async def secure_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] = list(message.get('headers', [])) + [
                    (b'x-content-type-options', b'nosniff'),
                    (b'x-frame-options', b'DENY'), (b'referrer-policy', b'same-origin')]
            await send(message)

        if not self.readonly:
            valid = False
            try:
                scheme, encoded = headers.get(b'authorization', b'').split(b' ', 1)
                username, supplied = base64.b64decode(encoded, validate=True).split(b':', 1)
                valid = (scheme.lower() == b'basic' and hmac.compare_digest(username, b'studio') and
                         hmac.compare_digest(supplied, self.password.encode('utf-8')))
            except (ValueError, binascii.Error):
                pass
            if not valid:
                return await reject(401, 'This research workspace requires its instance password',
                                    **{'WWW-Authenticate': 'Basic realm="Research workspace", charset="UTF-8"'})
        if method not in {'GET', 'HEAD', 'OPTIONS'}:
            if self.readonly:
                return await reject(403, 'This public instance is read-only')
            origin = headers.get(b'origin', b'').decode('latin1')
            expected = self.origin or scope.get('scheme', 'http') + '://' + headers.get(b'host', b'').decode('latin1')
            if headers.get(b'sec-fetch-site') == b'cross-site' or (origin and origin != expected):
                return await reject(403, 'Open this workspace directly before saving changes')
            if shutil.disk_usage(self.data_dir).free < 256 * 1024 * 1024:
                return await reject(507, 'Workspace disk is nearly full; the administrator must free space')
            # Bound streaming requests too, before an endpoint can publish data.
            limit = 26 * 1024 * 1024
            try:
                length = int(headers.get(b'content-length', b'0'))
            except ValueError:
                return await reject(400, 'Invalid request length')
            if length < 0 or length > limit:
                return await reject(413, 'The request exceeds the hosted upload limit')
            # A single ASGI event loop owns this counter. Hold admission through
            # response completion so queued bodies cannot accumulate in memory.
            if self.buffering >= 4:
                return await reject(429, 'Four writes are in progress; retry shortly', **{'Retry-After': '2'})
            self.buffering += 1
            try:
                body = bytearray()
                async with asyncio.timeout(120):
                    while True:
                        message = await receive()
                        if message['type'] == 'http.disconnect':
                            return
                        body.extend(message.get('body', b''))
                        if len(body) > limit:
                            return await reject(413, 'The request exceeds the hosted upload limit')
                        if not message.get('more_body', False):
                            break
                original_receive, delivered = receive, False

                async def replay():
                    nonlocal delivered
                    if delivered:
                        return await original_receive()
                    delivered = True
                    return {'type': 'http.request', 'body': bytes(body), 'more_body': False}

                return await self.app(scope, replay, secure_send)
            except TimeoutError:
                return await reject(408, 'Request upload timed out')
            finally:
                self.buffering -= 1
        return await self.app(scope, receive, secure_send)


class CollectionTextEncoder:
    """Use the pinned text tower without loading a prepared image library."""
    def __init__(self, directory):
        from .study import verify
        self.directory = Path(directory).resolve()
        self.encoder_config = read(self.directory / 'encoder.json')
        verify(self.encoder_config)
        self.encoder, self.lock = None, threading.Lock()

    def encode(self, texts):
        with self.lock:
            if self.encoder is None:
                from .semantic_encoder import TextEncoder
                self.encoder = TextEncoder(self.encoder_config)
            return self.encoder.encode(texts)


def initialize(directory):
    root = Path(directory).expanduser().resolve()
    if root.is_relative_to(Path(__file__).parent.resolve() / 'static'):
        raise ValueError('Workspace data must be outside publicly served assets')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'collections').mkdir(exist_ok=True)
    return root


def _bundle_path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('Bundle paths must be relative to the private bundle directory')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.exists():
        raise ValueError('A required bundle artifact is absent or outside the bundle')
    return path


def configure_bundle(directory, bundle_directory):
    """Bind unchanged approved assets to this host; never rebase old predictions."""
    from .collection_scan import CLASSES, REVIEW_RULE, SCOPE, SCORE_TYPE, snapshot_config
    from .conch_infer import CHECKPOINT_SHA256, CODE_REVISION, REVISION, file_hash, verify_code
    from .model_releases import load_release
    root = initialize(directory)
    bundle_root = Path(bundle_directory).expanduser().resolve()
    bundle = read(bundle_root / 'bundle.json')
    if bundle.get('schema') != 'tissue-private-model-bundle-v1':
        raise ValueError('Unsupported private model bundle schema')
    paths = {key: _bundle_path(bundle_root, bundle[key]) for key in
             ('release_directory', 'checkpoint', 'code_directory')}
    release, _, _, _ = load_release(paths['release_directory'])
    verify_code(paths['code_directory'])
    if file_hash(paths['checkpoint']) != CHECKPOINT_SHA256:
        raise ValueError('The checkpoint does not match the pinned CONCH weights')
    config = {
        'schema_version': 2, 'created_at': now(), 'backend': 'cpu', 'name': release['name'],
        **{k: str(v) for k, v in paths.items()},
        'release_sha256': release['sha256'], 'classifier_sha256': release['sha256'],
        'coefficients_sha256': release['coefficients_sha256'], 'checkpoint_sha256': CHECKPOINT_SHA256,
        'code_revision': CODE_REVISION, 'model_revision': REVISION, 'classes': CLASSES,
        'batch_size': 4, 'score_threshold': 0.8, 'margin_threshold': 0.2,
        'scope': SCOPE, 'review_rule': REVIEW_RULE, 'score_type': SCORE_TYPE,
    }
    target = root / 'collections' / 'scan-model.json'
    if target.exists():
        previous = read(target)
        comparable = lambda value: {k: v for k, v in value.items() if k not in {'created_at', 'sha256'}}
        if comparable(previous) != comparable(config):
            raise ValueError('An existing scan configuration differs; use a new workspace for this bundle')
        from .collection_scan import validate_config
        validate_config(previous)
        config = previous
    else:
        config['sha256'] = digest(config)
    settings = {'schema': 'tissue-portable-assets-v1', 'scan_configuration_sha256': config['sha256'],
                'model_bundle': str(bundle_root)}
    if 'text_weights' in bundle:
        weights = _bundle_path(bundle_root, bundle['text_weights'])
        if file_hash(weights) != bundle.get('text_weights_sha256'):
            raise ValueError('Extracted text weights do not match their private bundle receipt')
        text_dir = root / 'text-encoder'
        text_dir.mkdir(exist_ok=True)
        text_config = {'code_directory': str(paths['code_directory']), 'code_revision': CODE_REVISION,
                       'source_checkpoint_sha256': CHECKPOINT_SHA256,
                       'weights_path': str(weights), 'weights_sha256': bundle['text_weights_sha256']}
        text_config['sha256'] = digest(text_config)
        existing = text_dir / 'encoder.json'
        if existing.exists() and read(existing) != text_config:
            raise ValueError('The existing text encoder differs; use a new workspace')
        if not existing.exists():
            write(existing, text_config)
        settings['text_encoder_directory'] = str(text_dir)
    if 'replay_directory' in bundle:
        settings['replay_directory'] = str(_bundle_path(bundle_root, bundle['replay_directory']))
    if bundle.get('external_evidence'):
        entries = {}
        for row in bundle['external_evidence']:
            original = row['source_path']
            if not isinstance(original, str) or not Path(original).is_absolute() or original in entries:
                raise ValueError('External evidence must have unique original absolute paths')
            copied = _bundle_path(bundle_root, row['path'])
            if file_hash(copied) != row['sha256']:
                raise ValueError('An external evidence copy differs from its original checksum')
            entries[original] = {'path': row['path'], 'sha256': row['sha256']}
        mapping = {'schema': 'tissue-artifact-map-v1', 'entries': entries}
        map_path = root / 'artifact-map.json'
        if map_path.exists() and read(map_path) != mapping:
            raise ValueError('Existing evidence relocation differs; use a new workspace')
        if not map_path.exists():
            write(map_path, mapping)
        settings['artifact_map'] = str(map_path)
        settings['artifact_map_sha256'] = file_hash(map_path)
    snapshot_config(root / 'collections', config)
    if not target.exists():
        write(target, config)
    settings['sha256'] = digest(settings)
    settings_path = root / 'portable-assets.json'
    if settings_path.exists() and read(settings_path) != settings:
        raise ValueError('Installed portable assets differ; use a new workspace')
    if not settings_path.exists():
        write(settings_path, settings)
    return settings


def from_environment():
    from .app import create_app
    from .local_scheduler import LocalScanLauncher
    from .study import verify
    root = initialize(os.environ.get('TISSUE_DATA_DIR', './local-data'))
    readonly = os.environ.get('TISSUE_PUBLIC_READONLY', '0') == '1'
    password = os.environ.get('TISSUE_ACCESS_PASSWORD', '')
    # Validate access configuration before any expensive model work.
    public_origin = os.environ.get('TISSUE_PUBLIC_ORIGIN') or os.environ.get('RENDER_EXTERNAL_URL')
    AccessGate(None, password=password, readonly=readonly, data_dir=root, public_origin=public_origin)
    backend = os.environ.setdefault('TISSUE_EXECUTION_BACKEND', 'local')
    if backend != 'local':
        raise ValueError('The portable entrypoint requires TISSUE_EXECUTION_BACKEND=local')
    settings_path = root / 'portable-assets.json'
    settings = read(settings_path) if settings_path.exists() else {}
    if settings:
        verify(settings)
        config = read(root / 'collections' / 'scan-model.json')
        verify(config)
        if config['sha256'] != settings['scan_configuration_sha256']:
            raise ValueError('Installed model configuration changed; restore its frozen record')
        if settings.get('artifact_map'):
            from .conch_infer import file_hash
            if file_hash(settings['artifact_map']) != settings['artifact_map_sha256']:
                raise ValueError('Installed evidence relocation changed')
            os.environ['TISSUE_ARTIFACT_MAP'] = settings['artifact_map']
            os.environ['TISSUE_MODEL_BUNDLE'] = settings['model_bundle']
        if settings.get('replay_directory'):
            os.environ.setdefault('TISSUE_REPLAY_DIR', settings['replay_directory'])
    collections = root / 'collections'
    launcher = LocalScanLauncher(collections)
    scans = launcher.available() and not readonly
    text = CollectionTextEncoder(settings['text_encoder_directory']) if settings.get('text_encoder_directory') else None
    region = None
    if scans:
        from .region_encoder import RegionEncoder
        region = RegionEncoder(collections / 'scan-model.json')
    app = create_app(
        directory=os.environ.get('TISSUE_STUDY_DIR'),
        spatial_directory=os.environ.get('TISSUE_SPATIAL_DIR'),
        research_directory=os.environ.get('TISSUE_RESEARCH_DIR'),
        collections_directory=collections, collection_import_roots=[],
        collection_submit_scan=launcher if scans else None,
        collection_scans_enabled=scans, collection_text_encoder=text,
        collection_region_encoder=region,
        collection_limits={'max_images': 1000, 'max_collection_bytes': 512 * 1024 * 1024})
    app.add_middleware(AccessGate, password=password, readonly=readonly, data_dir=root,
                       public_origin=public_origin)

    @app.get('/healthz')
    def healthz():
        return {'status': 'ok'}

    @app.get('/api/deployment')
    def deployment():
        return {'execution': 'local_cpu', 'readonly': readonly, 'scans_available': scans,
                'shared_workspace': True, 'model_bundle_installed': bool(settings),
                'text_search_configured': text is not None}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Initialize an empty workspace without models or datasets')
    init.add_argument('--data-dir', default=os.environ.get('TISSUE_DATA_DIR', './local-data'))
    configure = sub.add_parser('configure', help='Verify and attach a private approved model bundle')
    configure.add_argument('--data-dir', default=os.environ.get('TISSUE_DATA_DIR', './local-data'))
    configure.add_argument('--bundle', required=True)
    args = parser.parse_args()
    if args.command == 'configure':
        configure_bundle(args.data_dir, args.bundle)
        print('Private model bundle configured; no historical predictions were changed.')
    else:
        initialize(args.data_dir)
        print('Workspace initialized; no model or dataset was downloaded.')


if __name__ == '__main__':
    main()
