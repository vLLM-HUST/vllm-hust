"""Request-bound real preparation cache; no model dispatch from this route."""
import asyncio
import hashlib
import json
import time
from jinja2 import Template


class PreparedInputCache:
    def __init__(self):
        self.templates = {}
        self.tokens = {}
        self.waiting = 0
        self.active = 0
        self.lock = asyncio.Lock()

    def clear(self):
        if self.active or self.waiting:
            raise ValueError('cannot clear an active preparation cache')
        before = {'templates': len(self.templates), 'tokenizations': len(self.tokens)}
        self.templates.clear()
        self.tokens.clear()
        return before

    async def prepare(self, body, tokenizer):
        text = body['text']
        template = body.get('template', '{{ text }}')
        request_id = body['request_id']
        if not isinstance(text, str) or not isinstance(template, str) or len(text) > 200000:
            raise ValueError('bounded string input required')
        # Only serialization of actual input is counted, not caller estimates.
        serialized = json.dumps({'text': text, 'template': template}, ensure_ascii=False).encode()
        key = hashlib.sha256(serialized).hexdigest()
        enqueued = time.monotonic_ns()
        # The snapshot is taken *after* this request joins the preparation
        # queue, so the depth it reports includes this request.  Capturing it
        # before the enqueue made the field structurally always zero, which the
        # live observability gate correctly treats as "never wired".
        self.waiting += 1
        observed = {
            'request_id': request_id, 'clock_id': 'monotonic_ns', 'observed_at_ns': enqueued,
            'tokenizer_identity': type(tokenizer).__module__ + '.' + type(tokenizer).__qualname__,
            'template_identity': hashlib.sha256(template.encode()).hexdigest(),
            'tokenizer_cache_state': 'warm' if key in self.tokens else 'cold',
            'template_cache_state': 'warm' if template in self.templates else 'cold',
            'serialization_work_units': len(serialized),
            'preprocessor_queue_depth': self.waiting,
            'preprocessor_queue_ahead': max(0, self.waiting - 1),
            'preprocessor_active_count': self.active,
            'preprocessor_pressure_ratio': self.waiting / (self.waiting + self.active) if self.waiting + self.active else 0.0,
            'prepared_input_available': key in self.tokens,
            'source': 'PreparedInputCache actual template/token caches and async lock queue',
            'queue_depth_semantics': ('requests enqueued for preparation at this pre-action '
                                      'boundary, including this request'),
            'queue_ahead_semantics': ('requests enqueued ahead of this request; '
                                      'preprocessor_queue_depth - 1'),
            'prepared_input_available_semantics': 'the token ids existed in the provider cache before this preparation ran',
        }
        acquired = False
        try:
            async with self.lock:
                self.waiting -= 1
                acquired = True
                self.active += 1
                decision_at_ns = time.monotonic_ns()
                try:
                    compiled = self.templates.get(template)
                    if compiled is None:
                        compiled = Template(template)
                        self.templates[template] = compiled
                    rendered = compiled.render(text=text)
                    ids = self.tokens.get(key)
                    if ids is None:
                        ids = await asyncio.to_thread(tokenizer.encode, rendered, add_special_tokens=False)
                        self.tokens[key] = list(ids)
                    elapsed = time.monotonic_ns() - enqueued
                    return {'schema':'anise.tokens-prompt.real-preparation.v1', 'pre_action': observed,
                            'decision_at_ns': decision_at_ns, 'finished_at_ns': time.monotonic_ns(),
                            'preparation_elapsed_ns': elapsed, 'token_ids': list(ids), 'rendered_text': rendered,
                            'input_sha256': key, 'token_count': len(ids),
                            'cache_semantics':'provider-owned rendered-text tokenization and compiled Jinja cache; not tokenizer initialized flag'}
                finally:
                    self.active -= 1
        except BaseException:
            if not acquired:
                self.waiting -= 1
            raise


CACHE = PreparedInputCache()
