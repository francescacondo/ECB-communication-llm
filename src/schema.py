from typing import Literal

from pydantic import BaseModel, model_validator


class SpeechSignals(BaseModel):
    inflation_attention: Literal[0, 1]
    inflation_outlook: Literal[-1, 0, 1]

    growth_attention: Literal[0, 1]
    growth_outlook: Literal[-1, 0, 1]

    financial_stability_attention: Literal[0, 1]
    uncertainty_attention: Literal[0, 1]

    @model_validator(mode="after")
    def check_attention_outlook_consistency(self):
        if self.inflation_attention == 0 and self.inflation_outlook != 0:
            raise ValueError(
                "inflation_outlook must be 0 when "
                "inflation_attention is 0"
            )

        if self.growth_attention == 0 and self.growth_outlook != 0:
            raise ValueError(
                "growth_outlook must be 0 when "
                "growth_attention is 0"
            )

        return self