"""Optional, versioned search guidance; never tissue labels or predictor updates."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Literal
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .feedback_storage import freeze
from .semantic import bounded_text
from .semantic_examples import validate_examples
from .storage import now, read
from .study import verify

FIELDS = ('goal','inclusion_criteria','exclusions','positive_ids','negative_ids','reviewer','note','scope','effective_query')


class GuidanceStore:
    def __init__(self, semantic):
        self.semantic = semantic
        self.directory = semantic.directory / 'guidance'

    @contextmanager
    def lock(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / '.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def load(self, kind, identifier):
        if kind not in ('definitions','applications','retractions'):
            raise ValueError('Unknown guidance record')
        self.semantic._id(identifier)
        value = read(self.directory / kind / (identifier+'.json'))
        verify(value)
        if value['id'] != identifier or value['catalog_sha256'] != self.semantic.catalog['sha256']:
            raise ValueError('Guidance record differs from this library')
        if kind == 'retractions':
            definition = self.definition(value['definition_id'])
            if definition['sha256'] != value['definition_sha256'] or definition['root_id'] != value['root_id']:
                raise ValueError('Guidance retraction binding changed')
        return value

    def records(self, kind):
        return [self.load(kind,p.stem) for p in sorted((self.directory/kind).glob('*.json'))]

    def status(self, definition):
        if any(r['root_id']==definition['root_id'] for r in self.records('retractions')):
            return 'retracted'
        if any(r.get('parent_id')==definition['id'] for r in self.records('definitions')):
            return 'superseded'
        return 'active'

    def definition(self, identifier):
        value = self.load('definitions',identifier)
        if value['parent_id']:
            parent = self.load('definitions',value['parent_id'])
            if parent['sha256'] != value['parent_sha256'] or parent['root_id'] != value['root_id']:
                raise ValueError('Guidance revision binding changed')
        elif value['root_id'] != identifier:
            raise ValueError('Guidance root binding changed')
        return value

    def active(self, identifier):
        definition = self.definition(identifier)
        if self.status(definition) != 'active':
            raise ValueError('Guidance has been revised or retracted; explicitly apply a current definition')
        return definition

    def save(self, *, goal, inclusion_criteria='', exclusions='', positive_ids=None, negative_ids=None,
             reviewer, note='', scope='current_task', parent_id=None):
        goal = bounded_text(goal,300,'Research goal')
        inclusion = bounded_text(inclusion_criteria,280,'Useful visual features',True)
        exclusions = bounded_text(exclusions,600,'Exclusions',True)
        reviewer = bounded_text(reviewer,100,'Researcher')
        note = bounded_text(note,2000,'Notes',True)
        if scope not in ('current_task','saved_definition'):
            raise ValueError('Choose this task or a saved definition')
        positive,negative = validate_examples(self.semantic,positive_ids or [],negative_ids or [])
        for identifier in positive+negative:
            if self.semantic.row(identifier)['role'] != 'train':
                raise ValueError('Guidance examples must be development images')
        effective = goal + ('. '+inclusion if inclusion else '')
        with self.lock():
            parent = self.active(parent_id) if parent_id else None
            identifier = uuid.uuid4().hex
            return freeze(self.directory/'definitions'/(identifier+'.json'), {
                'schema':'task-guidance-definition-v1','id':identifier,'created_at':now(),
                'catalog_sha256':self.semantic.catalog['sha256'],'task_kind':'appearance_search',
                'root_id':parent['root_id'] if parent else identifier,'parent_id':parent_id,
                'parent_sha256':parent['sha256'] if parent else None,
                'goal':goal,'inclusion_criteria':inclusion,'exclusions':exclusions,
                'positive_ids':positive,'negative_ids':negative,'reviewer':reviewer,'note':note,
                'scope':scope,'effective_query':effective,
                'purpose':'Optional search relevance guidance only; not tissue labels, global predictor updates or verified exclusion rules'})

    def apply(self, identifier):
        with self.lock():
            definition = self.active(identifier)
            application_id = uuid.uuid4().hex
            return freeze(self.directory/'applications'/(application_id+'.json'), {
                'schema':'task-guidance-application-v1','id':application_id,'created_at':now(),
                'catalog_sha256':self.semantic.catalog['sha256'],'definition_id':identifier,
                'definition_sha256':definition['sha256'],
                **{key:definition[key] for key in FIELDS}})

    def application(self, identifier):
        value = self.load('applications',identifier)
        definition = self.active(value['definition_id'])
        if (value['definition_sha256'] != definition['sha256'] or
                any(value[key] != definition[key] for key in FIELDS)):
            raise ValueError('Applied guidance differs from its definition')
        return value

    def retract(self, identifier, reviewer, note=''):
        with self.lock():
            definition = self.active(identifier)
            record_id = uuid.uuid4().hex
            return freeze(self.directory/'retractions'/(record_id+'.json'), {
                'schema':'task-guidance-retraction-v1','id':record_id,'created_at':now(),
                'catalog_sha256':self.semantic.catalog['sha256'],'definition_id':identifier,
                'definition_sha256':definition['sha256'],'root_id':definition['root_id'],
                'reviewer':bounded_text(reviewer,100,'Researcher'),'note':bounded_text(note,2000,'Retraction note',True)})

    def public(self):
        definitions = self.records('definitions')
        return {'available':True,'definitions':[{**row,'status':self.status(row)} for row in
                sorted(definitions,key=lambda r:r['created_at'],reverse=True)[:100]]}


class StrictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')


class DefinitionRequest(StrictRequest):
    goal:str=Field(min_length=1,max_length=300)
    inclusion_criteria:str=Field(default='',max_length=280)
    exclusions:str=Field(default='',max_length=600)
    positive_ids:list[str]=Field(default_factory=list,max_length=40)
    negative_ids:list[str]=Field(default_factory=list,max_length=40)
    reviewer:str=Field(min_length=1,max_length=100)
    note:str=Field(default='',max_length=2000)
    scope:Literal['current_task','saved_definition']='current_task'
    parent_id:str|None=None


class RetractionRequest(StrictRequest):
    reviewer:str=Field(min_length=1,max_length=100)
    note:str=Field(default='',max_length=2000)


def register_guidance(app, semantic):
    store=GuidanceStore(semantic) if semantic else None
    def configured():
        if store is None: raise HTTPException(409,'Search guidance is not configured')
        return store
    @app.get('/api/guidance')
    def status():
        return store.public() if store else {'available':False,'definitions':[]}
    @app.post('/api/guidance/definitions')
    def save(request:DefinitionRequest):
        return configured().save(**request.model_dump())
    @app.get('/api/guidance/definitions/{identifier}')
    def definition(identifier:str):
        record=configured().definition(identifier)
        return {**record,'status':configured().status(record)}
    @app.post('/api/guidance/definitions/{identifier}/apply')
    def apply(identifier:str,request:StrictRequest):
        return configured().apply(identifier)
    @app.post('/api/guidance/definitions/{identifier}/retract')
    def retract(identifier:str,request:RetractionRequest):
        return configured().retract(identifier,**request.model_dump())
    return store
