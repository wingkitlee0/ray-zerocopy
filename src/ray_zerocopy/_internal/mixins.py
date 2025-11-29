import functools
from typing import Generic, TypeVar

# Type variable for preserving wrapped type
T = TypeVar("T")


class WrapperMixin(Generic[T]):
    """
    Mixin providing common wrapper functionality.
    Handles metadata preservation (__doc__, __annotations__) and reference to original object.
    """

    def _configure_wrapper(self, pipeline: T):
        """
        Common initialization for wrappers.
        - Stores reference to original pipeline
        - Copies class docstring
        - Preserves annotations
        """
        # Copy class-level metadata from the wrapped pipeline
        if pipeline.__class__.__doc__:
            # Append original docstring to wrapper's docstring
            self.__doc__ = (
                f"{self.__class__.__doc__}\n\n"
                f"Wrapped class documentation:\n{pipeline.__class__.__doc__}"
            )

        # Preserve annotations if available
        if hasattr(pipeline, "__annotations__"):
            self.__annotations__ = pipeline.__annotations__

    def _preserve_call_signature(self, pipeline: T):
        """
        Attempts to preserve __call__ signature and docstring from the wrapped pipeline.
        """
        # Preserve __call__ signature and docstring if it exists
        if hasattr(pipeline, "__call__") and callable(
            getattr(pipeline, "__call__", None)
        ):
            original_call = getattr(pipeline.__class__, "__call__", None)
            if original_call is not None:
                # Update __call__ wrapper to preserve signature
                try:
                    functools.update_wrapper(self.__call__, original_call)  # type: ignore
                except (AttributeError, TypeError):
                    # If update_wrapper fails, just continue
                    pass
