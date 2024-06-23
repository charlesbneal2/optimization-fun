from pydantic import BaseModel, NonNegativeInt, model_validator
from typing import List


class TSPInstance(BaseModel):
    distances: List[List[NonNegativeInt]]
    names: List[str]

    @model_validator(mode="after")
    def validate_dimensions(cls, v):
        if len(v.distances) != len(v.distances[0]):
            raise ValueError("Distance matrix is not square")
        if len(v.names) != len(v.distances):
            raise ValueError("Number of names does not match rows in distance array")
        return v


