"""Forge's InputAccordion for tests/test_forge_harness.py.

Kept in its own module at top level: Gradio reads the source code of a
component subclass when the class is defined.
"""
import gradio as gr


class InputAccordion(gr.Checkbox):
    """A hidden checkbox plus an accordion, like modules.ui_components."""

    def __init__(self, value=None, **kwargs):
        label = kwargs.pop('label', '')
        kwargs.pop('elem_id', None)
        super().__init__(value=value, label=label, visible=False)
        self.accordion = gr.Accordion(label, open=bool(value))

    def __enter__(self):
        self.accordion.__enter__()
        return self

    def __exit__(self, *exc):
        return self.accordion.__exit__(*exc)
