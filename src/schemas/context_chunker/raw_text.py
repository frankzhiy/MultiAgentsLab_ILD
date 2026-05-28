from pydantic import BaseModel, ConfigDict, Field, field_validator


class RawTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(..., min_length=1)

    @field_validator("raw_text")
    @classmethod
    def raw_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_text must not be blank")
        return value



