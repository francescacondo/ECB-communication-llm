from schema import SpeechSignals


valid_example = SpeechSignals(
    inflation_attention=1,
    inflation_outlook=1,
    growth_attention=1,
    growth_outlook=-1,
    financial_stability_attention=0,
    uncertainty_attention=1,
)

print(valid_example)
