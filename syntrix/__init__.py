from importlib.metadata import version, PackageNotFoundError

__all__ = ["utils", "__version__"]

try:
    __version__ = version("syntrix")
except PackageNotFoundError:
    # Editable/development install fallback
    __version__ = "0.0.0+dev"
