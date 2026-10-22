"""Read immutable, independently checked classifier releases without activation.

A release records its actual fitting lineage. It is deliberately distinct from
the historical single-experiment ``classifier.json`` format.
"""
from pathlib import Path
import re

from .conch_infer import (CHECKPOINT_SHA256, CLASSIFIER_REPRESENTATION, CODE_REVISION,
                         MODEL, REPRESENTATION, REVISION, file_hash)
from .improve import encoder_identity
from .storage import digest, read

CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
RELEASE_CHECKS = {'nct_equal_class_macro_nondecrease', 'nct_stroma_correct_nondecrease',
                  'other_nct_classes_at_most_one_lost_correct', 'starc_small_strict_correct_improvement'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sealed(path):
    value = read(path)
    _require(isinstance(value, dict) and value.get('sha256') == digest(
        {key: item for key, item in value.items() if key != 'sha256'}), 'Release artifact hash differs')
    return value


def _inside(directory, relative):
    path = Path(relative)
    _require(not path.is_absolute() and '..' not in path.parts, 'Release file path must be relative')
    target = directory / path
    _require(not target.is_symlink() and target.resolve().is_relative_to(directory) and target.is_file(),
             'Release file is missing or outside its directory')
    return target


def load_release(directory, expected_sha256=None):
    """Return (release, normalized encoder identity, coefficients, intercept).

    ``expected_sha256`` is the canonical release hash pinned by the scan config.
    Verify copied evidence, not mutable original paths recorded for provenance.
    """
    import numpy as np
    directory = Path(directory).resolve()
    release_path = directory / 'release.json'
    _require(release_path.stat().st_size <= 2 * 1024 * 1024, 'Release metadata exceeds size bound')
    release = _sealed(release_path)
    _require(expected_sha256 is None or release['sha256'] == expected_sha256, 'Release differs from pinned identity')
    _require(release.get('schema') == 'tissue-model-release-v1' and
             isinstance(release.get('release_id'), str) and
             re.fullmatch('[a-z0-9][a-z0-9-]{0,79}', release['release_id']) and
             release.get('status') == 'validated_inactive' and release.get('activation_performed') is False,
             'Unsupported immutable release metadata')
    _require(release.get('model_family') == 'multinomial-logistic-regression' and
             release.get('classes') == CLASSES and release.get('feature_key') == 'classifier_embeddings' and
             release.get('feature_dimension') == 512 and release.get('representation') == CLASSIFIER_REPRESENTATION,
             'Release classifier family, classes or representation differs')
    identity = encoder_identity(release['encoder_identity'])
    expected = {'model': MODEL, 'model_revision': REVISION, 'code_revision': CODE_REVISION,
                'checkpoint_sha256': CHECKPOINT_SHA256, 'representation': REPRESENTATION,
                'classifier_representation': CLASSIFIER_REPRESENTATION,
                'classifier_feature_key': 'classifier_embeddings', 'feature_transform': CLASSIFIER_REPRESENTATION}
    _require(all(identity.get(key) == value for key, value in expected.items()), 'Release encoder identity differs')
    preprocessing = identity.get('preprocessing', {})
    expected_preprocessing = {'image_size': 448, 'resize': 'bicubic shortest edge',
        'crop': 'center 448x448', 'color_mode': 'RGB',
        'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
    _require(all(preprocessing.get(key) == value for key, value in expected_preprocessing.items()),
             'Release preprocessing differs from pinned CONCH')
    files, total = {}, 0
    for record in release['files']:
        name = record['path']
        _require(name not in files, 'Release contains duplicate file paths')
        path = _inside(directory, name)
        size = path.stat().st_size
        total += size
        _require(size == record['bytes'] and size <= 32 * 1024 * 1024 and total <= 96 * 1024 * 1024,
                 'Release file size differs or exceeds bound')
        _require(file_hash(path) == record['sha256'], 'Release file checksum differs: ' + name)
        files[name] = record

    def evidence(name):
        _require(name in files, 'Release evidence is missing from the checksum inventory')
        return _sealed(_inside(directory, name))

    validation = release['validation']
    result = evidence(validation['result_file'])
    verification = evidence(validation['verification_file'])
    protocol = evidence('evidence/validation/protocol.json')
    _require(validation.get('release_checks_passed') is True and validation.get('independently_verified') is True and
             validation['result_sha256'] == result['sha256'] and
             validation['verification_sha256'] == verification['sha256'] and
             validation['protocol_sha256'] == protocol['sha256'], 'Release validation lineage differs')
    _require(set(result['release_checks']) == RELEASE_CHECKS and
             all(value is True for value in result['release_checks'].values()) and
             result['release_checks'] == verification['release_checks'] == validation['release_checks'] and
             result['release_checks_passed'] is True and verification['release_checks_passed'] is True and
             verification['passed'] is True, 'Release acceptance or independent verification did not pass')
    _require(verification['result_file_sha256'] == files[validation['result_file']]['sha256'] and
             result['protocol_file_sha256'] == verification['protocol_file_sha256'] ==
             files['evidence/validation/protocol.json']['sha256'], 'Release verification file binding differs')
    _require(result['candidate_sha256'] == verification['candidate_sha256'] == protocol['candidate_sha256'] ==
             release['coefficients_sha256'] and result['candidate_id'] == verification['candidate_id'] ==
             protocol['candidate_id'] == release['training']['candidate_id'], 'Release candidate differs from evaluated model')
    _require(result['new_images'] == validation['new_images'] == 1650 and
             result['new_cohorts'] == validation['new_cohorts'] == 0 and
             result['patient_disjointness_verified'] is False and
             validation['patient_disjointness_verified'] is False and
             all(result[key] is False for key in ('training_performed', 'encoder_training', 'production_modified',
                 'post_test_refitting', 'alternate_candidate_selection')), 'Release overstates validation scope')
    training = evidence('evidence/training/training-inputs.json')
    selection = evidence('evidence/training/selection.json')
    _require(release['training']['images'] == training['fit_images'] == 6100 and
             release['training']['original_images'] == len(training['original_fit_ids']) == 5400 and
             release['training']['new_source_images'] == len(training['new_fit_ids']) == 700 and
             len(set(training['original_fit_ids'] + training['new_fit_ids'])) == 6100 and
             release['training']['training_inputs_sha256'] == training['sha256'] and
             release['training']['selection_sha256'] == selection['sha256'] and
             selection['selected_id'] == release['training']['candidate_id'] and
             selection['selected_model']['file_sha256'] == release['coefficients_sha256'] and
             selection['selected_model']['recipe'] == release['training']['selected_recipe'],
             'Release fitting lineage differs')
    prior = evidence(release['historical_reused_evaluations']['result_file'])
    _require(prior['sha256'] == release['historical_reused_evaluations']['result_sha256'] and
             prior['external_checks_passed'] is False and
             prior['evaluations']['crc_val_reused']['models']['original']['per_class_counts']['STR'] ==
             {'support': 421, 'correct': 383} and
             prior['evaluations']['crc_val_reused']['models']['selected']['per_class_counts']['STR'] ==
             {'support': 421, 'correct': 381}, 'Prior CRC stroma regression must remain recorded')
    coefficient_name = release['coefficients_file']
    _require(coefficient_name in files and files[coefficient_name]['sha256'] == release['coefficients_sha256'],
             'Release coefficient checksum binding differs')
    with np.load(_inside(directory, coefficient_name), allow_pickle=False) as arrays:
        _require(set(arrays.files) == {'classes', 'coefficients', 'intercept'} and
                 arrays['classes'].tolist() == CLASSES, 'Release coefficient class order differs')
        coefficients, intercept = arrays['coefficients'].copy(), arrays['intercept'].copy()
    _require(coefficients.shape == (9, 512) and intercept.shape == (9,) and
             np.isfinite(coefficients).all() and np.isfinite(intercept).all(), 'Invalid release coefficients')
    return release, identity, coefficients, intercept
