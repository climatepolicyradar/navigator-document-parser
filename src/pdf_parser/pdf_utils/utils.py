from pydantic import BaseModel as PydanticBaseModel


# We need this class in order to add a new attribute to custom types e.g. LayoutWithFractions in disambiguate_layout.py
class BaseModel(PydanticBaseModel):
    """Base class for all models."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
