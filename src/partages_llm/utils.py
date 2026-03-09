import os
import sys
import inspect
import logging
import importlib.util
from typing import Any, Callable, List, Optional, Union
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from inspect import signature

_DATADIR_BASE = Path(os.getenv("HOME")) / "partages-llm-data"


class Bunch:

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __getattr__(self, name: str):
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __iter__(self):
        return iter(self.__dict__)

    def as_dict(self):
        return self.__dict__


@contextmanager
def ignored(*exceptions: Exception):
    try:
        yield
    except exceptions:
        pass


def basic_logger_init(lvl: str="info"):
    logger = logging.getLogger()
    logfmt = "%(asctime)s - %(levelname)-8s - %(message)s"
    level = getattr(logging, lvl.upper())
    logging.basicConfig(
        format=logfmt,
        datefmt="%d/%m/%Y %H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logger


def get_function_origin_info(func: Callable):
    info = {
        "name": func.__name__,
        "module": func.__module__,
        "qualname": getattr(func, "__qualname__", "N/A"),
        "file": inspect.getfile(func) if inspect.isfunction(func) else "N/A",
        "id": id(func)
    }
    return Bunch(**info)


def make_version_subdir_path(
    p: Path,
    return_strings: bool=False,
    make: bool=False,
    stem: str="v",
    suffix: Optional[str] = None
):
    existing_versions = list(p.glob(stem + "*/"))
    this_version = max(int(p.name[len(stem):]) for p in existing_versions) + 1 \
        if existing_versions else 0
    name = stem + str(this_version)
    if suffix is not None:
        name += suffix
    if return_strings:
        return [str(p), name]
    p_ret = p / name
    if make:
        p_ret.mkdir(parents=True)
    return p_ret


def import_from_path(module_name: str, file_path: Union[Path, str]):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_answer_mapping(letters: str):
    return dict(zip(letters, range(len(letters))))


def clean_quotes(s: str):
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def get_named_entities(text: List[str], tags: List[int]):
    result = []
    current_phrase = []
    for i, (token, label) in enumerate(zip(text, tags)):
        if label != 0:
            if i > 0 and tags[i - 1] != 0 and label < tags[i - 1]:
                current_phrase = [token]
                result.append(" ".join(current_phrase))
            else:
                current_phrase.append(token)
        else:
            if current_phrase:
                result.append(" ".join(current_phrase))
                current_phrase = []
    if current_phrase:
        result.append(" ".join(current_phrase))
    return result


def sanitize_path(path: Any, default_name: str, check_is_dir: bool = False):
    if not isinstance(path, Path):
        try:
            spath = Path(path)
        except TypeError:
            spath = _DATADIR_BASE / default_name
        else:
            existence = spath.is_dir() if check_is_dir else spath.is_file()
            if not existence:
                spath = _DATADIR_BASE / default_name
    return spath


def handle_input_paths(
    input_default: Optional[Union[Path, str]] = None,
    output_default: Optional[Union[Path, str]] = None
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)  # puts the values provided to `wrapper` into the signature of `func`
            bound_args.apply_defaults()  # fills in the default values for `func` arguments that were not provided to `wrapper`
            bound_args.arguments["input_path"] = sanitize_path(
                bound_args.arguments["input_path"], input_default
            )
            bound_args.arguments["output_path"] = sanitize_path(
                bound_args.arguments["output_path"], output_default, check_is_dir=True
            )
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def format_model_name(model_name: str, base_only: bool = False):
    s = -2 if base_only else -3
    split_slice = slice(s, -1) if model_name.endswith("__") else slice(s + 1, None)
    return "_".join(model_name.split("__")[split_slice])


