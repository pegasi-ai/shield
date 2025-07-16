from pegasi_shield.input_detectors import (
    Anonymize,
    StopInputSubstrings,
    LanguageInput,
    Secrets,
    PromptInjections,
    DoSTokens,
    MalwareInputURL,
    HarmfulInput,
    TextQualityInput,
    ToxicityInput,
    CodingLanguageInput,
    RegexInput,
)

from pegasi_shield.output_detectors import (
    HallucinationDetector,
    FactualConsistency,
    Contradictions,
    PromptOutputRelevance,
    OutputContextRelevance,
    PromptContextRelevance,
    SensitivePII,
    Deanonymize,
    StopOutputSubstrings,
    MalwareOutputURL,
    TextQualityOutput,
    HarmfulOutput,
    ToxicityOutput,
    CodingLanguageOutput,
    RegexOutput,
    Bias,
    LanguageOutput,
    Equity,
    TemporalMismatchDetector,
)

from pegasi_shield.vault import Vault


if __name__ == "__main__":

    # CodingLanguageInput(?),
    # https://docs.withsafeguards.com/output_detectors/coding_language/
    allowed_languages = ["Python", "Go"]
    denied_languages = ["JavaScript", "PHP", "Ruby", "Java"]

    # RegexInput(?)
    # https://docs.withsafeguards.com/output_detectors/regex/#configuration
    good_patterns = ["\b(union(\s+all)?|select|insert|update|delete|from|where)\b"]
    bad_patterns = ["\b(union(\s+all)?|select|insert|update|delete|from|where)\b"]

    # LanguageOutput(?)
    # https://docs.withsafeguards.com/input_detectors/language/
    valid_languages = ["en"]

    input_detectors = [
        HarmfulInput(),
        Anonymize(Vault()),
        StopInputSubstrings(),
        LanguageInput(),
        Secrets(),
        PromptInjections(),
        DoSTokens(),
        MalwareInputURL(),
        TextQualityInput(),
        ToxicityInput(),
        CodingLanguageInput(allowed=allowed_languages),
        RegexInput(good_patterns=good_patterns),
    ]

    output_detectors = [
        HallucinationDetector(threshold=0.6),  # TODO: TypeError: T5ForTokenClassification.forward() got an unexpected keyword argument 'token_type_ids'
        HarmfulOutput(),
        Equity(),  # Need `brew install libomp` to solve: OSError: dlopen(/Users/kuanwei/Github/pegasi-shield-safeguards/env_safeguards/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib, 0x0006): Library not loaded: @rpath/libomp.dylib
        FactualConsistency(threshold=0.6),
        Contradictions(),
        PromptOutputRelevance(threshold=0.8),
        PromptContextRelevance(threshold=0.8),
        OutputContextRelevance(threshold=0.8),
        SensitivePII(),
        TemporalMismatchDetector(),
        Deanonymize(Vault()),
        StopOutputSubstrings(),
        MalwareOutputURL(),
        TextQualityOutput(),
        ToxicityOutput(),
        Bias(),  # TODO d4data/bias-detection-model is only tensorflow in TFAutoModelForSequenceClassification. no pytorch.?
        CodingLanguageOutput(denied=denied_languages),
        RegexOutput(bad_patterns=bad_patterns),
        LanguageOutput(valid_languages=valid_languages),
    ]

    response_text = "No, the company is managing its CAPEX and Fixed Assets pretty efficiently, which is evident from below key metrics: CAPEX/Revenue Ratio: 5.1% Fixed assets/Total Assets: 20% Return on Assets= 12.4%"
    sanitized_prompt = "Is 3M a capital-intensive business based on FY2022 data?"
    context = "3M Company and Subsidiaries Consolidated Statement of Income Years ended December 31 (Millions, except per share amounts) 2022 2021 2020 Net sales $ 34,229 $ 35,355 $ 32,184 3M Company and Subsidiaries Consolidated Balance Sheet At December 31 (Dollars in millions, except per share amount) 2022 2021 Assets Current assets Cash and cash equivalents $ 3,655 $ 4,564  Marketable securities current 238  201  Accounts receivable net of allowances of $174 and $189 4,532  4,660  Inventories Finished goods 2,497  2,196  Work in process 1,606  1,577  Raw materials and supplies 1,269  1,212  Total inventories 5,372  4,985  Prepaids 435  654  Other current assets 456  339  Total current assets 14,688  15,403  Property, plant and equipment 25,998  27,213  Less: Accumulated depreciation (16,820) (17,784) Property, plant and equipment net 9,178  9,429  Operating lease right of use assets 829  858  Goodwill 12,790  13,486  Intangible assets net 4,699  5,288  Other assets 4,271  2,608  Total assets $ 46,455 $ 47,072 3M Company and Subsidiaries Consolidated Statement of Cash Flows Years ended December 31 (Millions) 2022 2021 2020 Cash Flows from Operating Activities Net income including noncontrolling interest $ 5,791 $ 5,929 $ 5,453  Adjustments to reconcile net income including noncontrolling interest to net cash provided by operating activities Depreciation and amortization 1,831  1,915  1,911  Long-lived and indefinite-lived asset impairment expense 618    6  Goodwill impairment expense 271      Company pension and postretirement contributions (158) (180) (156) Company pension and postretirement expense 178  206  322  Stock-based compensation expense 263  274  262  Gain on business divestitures (2,724)   (389) Deferred income taxes (663) (166) (165) Changes in assets and liabilities Accounts receivable (105) (122) 165  Inventories (629) (903) (91) Accounts payable 111  518  252  Accrued income taxes (current and long-term) (47) (244) 132  Other net 854  227  411  Net cash provided by (used in) operating activities 5,591  7,454  8,113  Cash Flows from Investing Activities Purchases of property, plant and equipment (PP&E) (1,749) (1,603) (1,501)"

    for detector in input_detectors:
        sanitized_response, valid_results, risk_score = detector.scan(sanitized_prompt)

        print(f"{detector = }\n")
        print(f"{sanitized_response = }\n")
        print(f"{valid_results = }\n")
        print(f"{risk_score = }\n")

    for detector in output_detectors:
        sanitized_response, valid_results, risk_score = detector.scan(
            sanitized_prompt, response_text, context
        )

        print(f"{detector = }\n")
        print(f"{sanitized_response = }\n")
        print(f"{valid_results = }\n")
        print(f"{risk_score = }\n")
