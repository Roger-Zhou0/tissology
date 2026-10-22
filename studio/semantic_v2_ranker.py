"""Frozen candidate: binary personal relevance in projected CONCH image space.

This adapter is separate from v1 and contains no tissue taxonomy or query lookup.
Text-only performance is limited; the discriminative model requires explicit
matching and nonmatching examples. Outputs are rankings, not probabilities.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms < 1e-12):
        raise ValueError('Invalid CONCH feature values')
    return values / norms


class PersonalRelevanceRanker:
    def __init__(self, store, config_path):
        self.config_path = Path(config_path).resolve()
        self.config_file_sha256 = file_sha(self.config_path)
        self.config = json.loads(self.config_path.read_text())
        payload = {k:v for k,v in self.config.items() if k != 'sha256'}
        actual = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if self.config['sha256'] != actual:
            raise ValueError('Ranking configuration hash changed')
        c = self.config
        if (c['schema'] != 'semantic-personal-relevance-rbf-v2' or
                c['algorithm'] != {'kernel':'rbf','C':10.0,'gamma':5.0,'class_weight':'balanced'} or
                c['source_sha256'] != file_sha(__file__) or
                c['catalog_sha256'] != store.catalog['sha256'] or
                c['index_file_sha256'] != store.catalog['index_sha256'] or
                c['encoder_sha256'] != store.encoder_config['sha256']):
            raise ValueError('Ranking source, feature space or encoder differs from the frozen candidate')
        if c['prefixes'] != ['a histopathology image showing ', 'H&E stained tissue showing ',
                             'a microscopic image of tissue showing ']:
            raise ValueError('Frozen generic text context changed')
        self.store = store
        self.features = unit(store.features)
        if self.features.shape != (len(store.ids),512):
            raise ValueError('Candidate requires projected512-dimensional CONCH features')
        self.version = c['version']
        self.evaluation_asset = c.get('evaluation_asset')
        self.gallery_ids = c['ordered_gallery_ids']
        if len(self.gallery_ids) != 40 or len(set(self.gallery_ids)) != 40 or not set(self.gallery_ids) <= set(store.ids):
            raise ValueError('Frozen nested gallery differs from this library')

    def rank(self, text, positive_ids, negative_ids, exclusions=''):
        if file_sha(self.config_path) != self.config_file_sha256:
            raise ValueError('Ranking configuration changed after initialization')
        if not isinstance(text,str) or not text.strip() or len(text)>600:
            raise ValueError('Search text must contain1–600 characters')
        if not isinstance(exclusions,str) or len(exclusions)>600:
            raise ValueError('Exclusions must contain0–600 characters')
        for examples in (positive_ids,negative_ids):
            if not isinstance(examples,list) or len(examples)!=len(set(examples)) or not set(examples)<=set(self.store.ids):
                raise ValueError('Feedback examples must be unique IDs from this library')
        if set(positive_ids)&set(negative_ids) or len(positive_ids)+len(negative_ids)>40:
            raise ValueError('Use at most40 distinct matching/nonmatching examples')
        prefixes = self.config['prefixes']
        context = unit(unit(self.store.encode([p+text.strip() for p in prefixes])).mean(axis=0))
        excluded_vector = None
        if exclusions.strip():
            excluded_vector = unit(self.store.encode([exclusions.strip()]))[0]
            context = unit(context - 0.3*excluded_vector)
        text_scores = self.features @ context
        scores = text_scores.copy()
        trained = bool(positive_ids and negative_ids)
        absent_class = not trained
        notices = ['Ranking scores are not probabilities or automatic verification of a description.',
                   'Text-only informal grounding remains weak; explicit relevance examples can define your goal.']
        provenance = {'version':self.version,'config_sha256':self.config['sha256'],
             'config_file_sha256':self.config_file_sha256, 'source_sha256':self.config['source_sha256'],
             'catalog_sha256':self.config['catalog_sha256'],'index_file_sha256':self.config['index_file_sha256'],
             'encoder_sha256':self.config['encoder_sha256'], 'representation':'L2-normalized projected CONCH image features512D',
             'encoded_positive_texts':[p+text.strip() for p in prefixes],
             'encoded_exclusion_text':exclusions.strip(),'exclusion_weight':0.3,
             'query_embedding':context.tolist(),
             'excluded_embedding':None if excluded_vector is None else excluded_vector.tolist(),
             'positive_ids':list(positive_ids),'negative_ids':list(negative_ids),
             'reviewed_image_sha256':{i:self.store.rows[i]['image_sha256'] for i in positive_ids+negative_ids},
             'review_count':len(positive_ids)+len(negative_ids), 'abstain':absent_class,
             'feedback_overrides_text_ranking':trained,
             'guidance_rule':'Separate exclusion embedding affects text fallback. With both feedback classes, explicit examples define the binary relevance score.',
             'development_result_sha256':self.config['development_result_sha256']}
        if trained:
            from sklearn.svm import SVC
            import sklearn
            training_ids = positive_ids+negative_ids
            positions = [self.store.positions[i] for i in training_ids]
            labels = np.r_[np.ones(len(positive_ids)),np.zeros(len(negative_ids))]
            model = SVC(C=10.0,gamma=5.0,class_weight='balanced',kernel='rbf')
            model.fit(self.features[positions], labels)
            scores = model.decision_function(self.features)
            provenance['binary_model'] = {'algorithm':self.config['algorithm'],
                'sklearn_version':sklearn.__version__,'support_ids':[training_ids[i] for i in model.support_],
                'dual_coefficients':model.dual_coef_.tolist(),'intercept':model.intercept_.tolist(),
                'classes':model.classes_.tolist(),'n_support':model.n_support_.tolist()}
            provenance['binary_model']['sha256'] = hashlib.sha256(json.dumps(
                provenance['binary_model'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            notices += ['Matching and nonmatching examples define this ranking; they do not change tissue labels or model weights.',
                        'Text and exclusions remain visible guidance; binary example feedback controls the final ranking.']
            method = 'personal_binary_rbf'
        else:
            method = 'generic_context_text_fallback'
            notices.append('No learned relevance ranking: provide at least one matching and one nonmatching example. Full exploratory text fallback scores are retained; this case abstains from the guided model.')
        return {'scores':np.asarray(scores),'text_scores':np.asarray(text_scores),'method':method,
                'provenance':provenance,'notices':notices,
                'score_interpretation':'Binary relevance decision value' if trained else 'CONCH contextual text-image cosine similarity'}


def load_ranker(store, config_path):
    return PersonalRelevanceRanker(store,config_path)
