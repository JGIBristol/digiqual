"""The API reference is generated from each module's ``__all__`` and its docstrings
(quartodoc, see `_quarto.yml`). These tests keep both complete, so a new public
function shows up in the docs, fully described, without editing the site config:

- every public function or class defined in an API module is in its ``__all__``
  (prefix helpers with ``_`` to keep them out of the docs);
- every exported function, class and public method has a Google-style docstring
  that documents each parameter under ``Args:`` and the result under ``Returns:``.
"""

import importlib
import inspect
import re

import pytest

API_MODULES = [
    "core", "sampling", "diagnostics", "adaptive", "pod", "integration", "ahat",
    "plotting", "executors", "io", "cpp_fallback", "defaults",
]

_SECTION = re.compile(r"^(\w[\w ]*):\s*$")


def _section(doc: str, name: str) -> list[str] | None:
    """The lines of a Google-style docstring section (None if it is absent)."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"{name}:" and not line.startswith(" "):
            body = []
            for nxt in lines[i + 1:]:
                if _SECTION.match(nxt):
                    break
                body.append(nxt)
            return body
    return None


def _documented_args(doc: str) -> set[str]:
    body = _section(doc, "Args") or []
    names = set()
    for line in body:
        m = re.match(r"^\s{4}(\*{0,2}\w+)(?:\s*\(.*?\))?\s*:", line)
        if m:
            names.add(m.group(1).lstrip("*"))
        # a shared description, e.g. "    a, b: As in `pod()`."
        m = re.match(r"^\s{4}((?:\w+,\s*)+\w+)\s*:", line)
        if m:
            names.update(n.strip() for n in m.group(1).split(","))
    return names


def _params(func) -> list[str]:
    return [p.name for p in inspect.signature(func).parameters.values() if p.name not in ("self", "cls")]


def _api_objects():
    for mod_name in API_MODULES:
        mod = importlib.import_module(f"digiqual.{mod_name}")
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if inspect.isfunction(obj):
                yield f"{mod_name}.{name}", obj, "function"
            elif inspect.isclass(obj):
                yield f"{mod_name}.{name}", obj, "class"
                for meth_name, meth in vars(obj).items():
                    if inspect.isfunction(meth) and not meth_name.startswith("_"):
                        yield f"{mod_name}.{name}.{meth_name}", meth, "method"


API_OBJECTS = list(_api_objects())


@pytest.mark.parametrize("mod_name", API_MODULES)
def test_module_has_docstring_and_all(mod_name):
    mod = importlib.import_module(f"digiqual.{mod_name}")
    assert mod.__doc__, f"digiqual.{mod_name} needs a module docstring (the API page's description)"
    assert hasattr(mod, "__all__"), f"digiqual.{mod_name} needs an __all__"
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"__all__ of digiqual.{mod_name} names undefined objects: {missing}"


@pytest.mark.parametrize("mod_name", API_MODULES)
def test_public_definitions_are_exported(mod_name):
    mod = importlib.import_module(f"digiqual.{mod_name}")
    public = [
        name for name, obj in vars(mod).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == mod.__name__
    ]
    unexported = sorted(set(public) - set(mod.__all__))
    assert not unexported, (
        f"digiqual.{mod_name} defines public {unexported} but doesn't export them: add them to "
        "__all__ (so they appear in the API reference) or prefix them with '_'."
    )


@pytest.mark.parametrize("qualname,obj,kind", API_OBJECTS, ids=[o[0] for o in API_OBJECTS])
def test_docstring_is_complete(qualname, obj, kind):
    doc = inspect.getdoc(obj) if kind != "class" else (obj.__doc__ and inspect.cleandoc(obj.__doc__))
    assert doc, f"{qualname} has no docstring"
    assert doc.splitlines()[0].strip(), f"{qualname}: the docstring should start with a one-line summary"

    # Parameters: a function's own, or a class's __init__ (documented on the class)
    func = obj.__init__ if kind == "class" else obj
    if kind == "class" and func is object.__init__:
        params = []
    elif kind == "class" and not any("__init__" in vars(c) for c in obj.__mro__[:-1] if c.__module__.startswith("digiqual")):
        params = []  # inherits a third-party constructor (e.g. an Exception)
    else:
        params = [p for p in _params(func)
                  if inspect.signature(func).parameters[p].kind not in (inspect.Parameter.VAR_POSITIONAL,
                                                                         inspect.Parameter.VAR_KEYWORD)]
    missing = [p for p in params if p not in _documented_args(doc)]
    assert not missing, f"{qualname}: document these parameters under 'Args:': {missing}"

    if kind != "class":
        returns_none = inspect.signature(obj).return_annotation in (None, "None")
        if not returns_none:
            assert _section(doc, "Returns") is not None or _section(doc, "Yields") is not None, (
                f"{qualname}: add a 'Returns:' section (or annotate '-> None')"
            )
