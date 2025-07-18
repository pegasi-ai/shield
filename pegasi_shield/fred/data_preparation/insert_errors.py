from numpy import int32
import pandas as pd
import ast
import random
import json
import json_repair
from json_repair import repair_json
import logging
from groq import Groq
from openai import OpenAI
import sys
import os
import tiktoken

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from utils import *

# Configure logging to write to a file
logging.basicConfig(filename='insert_errors.log', 
                    level=logging.ERROR,   # Only log ERROR and higher level messages
                    format='%(asctime)s - %(levelname)s - %(message)s')


def call_llm(client, model_name, messages, temperature, max_tokens):
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,     
        )

        # Try to parse the response to ensure it's valid JSON
        result = chat_completion.choices[0].message.content
    except Exception as e:
        error_message = str(e)
        logging.error(f"ERROR CAUGHT: {error_message}")

        if "'failed_generation':" in error_message:
            try:
                logging.error("Error message contains 'failed_generation'")
                # Extract the JSON part using the function
                result = extract_failed_generation_json(error_message)
                logging.error(f"Extracted JSON part: {result}")
            except Exception:  # parse_error
                logging.error(
                    "An unexpected error occurred while parsing 'failed_generation':"
                )
                raise
        else:
            logging.error("An unexpected error occurred:")
            raise

    # Clean result and attempt to parse the cleaned json
    result = extract_wrapped_json(result)
    result = json_repair.repair_json(result)
    result = json.loads(result)
    return result["Edited"]


### create numerical errors
def create_numerical_error(passage, model_name, max_tokens, temperature):

    messages=[
                {"role": "user", "content": "Given a passage with possibly already inserted error tokens wrapped in <temporal>, <entity>, <relation>, "
                + "<contradictory>, or <unverifiable>, insert numerical errors "
                + "in the passage below, wrapped in tokens to make the passage factually incorrect. Ensure these insertions are outside these existing <> tags, and don't modify the <> tags at all. "
                + "The error is defined as such:\n"
                + "1. numerical errors (<numerical>): an incorrect calculation, estimation, or interpretation of numerical data such as percentages, growth rates, totals, differences, or ratios. " 
                + "These errors can arise from misapplying formulas, misreading data, rounding incorrectly, or failing to consider time periods or units.\n" 

                + "Example 1: Net income was calculated as revenue ($9.5 billion) minus expenses ($7.3 billion), totaling <numerical><delete>$2.2 billion</delete><mark>$1.2 million</mark></numerical>.\n" 
                + "Example 2: Operating expenses increased by <numerical><delete>3.5%</delete><mark>13.5%</mark></numerical> from the previous quarter.\n"
                + "Example 3: The average proportional recoverable environmental capital expenditures for the years December 31, 2015, 2014, " 
                + "and 2013 were $159.33 million (($205 + $163 + $110) / 3 = <numerical><delete>$159.33</delete><mark>$159.34</mark></numerical> million).\n"

                + "Now, insert numerical error tokens in the given passage but make sure that you don't modify "
                + "anything inside any already existing <> error tokens, only add numerical errors with <numerical>"
                + "</numerical> tokens outside the already existing <> tags. Also, avoid inserting errors in the "
                + "first sentence. Also make sure to tag every single edit with <mark></mark>, <delete></delete> "
                + "and <numerical><numerical> tags like done in the examples below.\n##\n"

                + "Paragraph: The estimated total cost to replace the annuities the company was liable for in 2017 was $179,062, "
                + "with $144,618 being from The Prudential Insurance Company of America and $34,444 from the unaffiliated life insurance company.\n"

                + "Edited: The estimated total cost to replace the annuities the company was liable for in 2017 was <numerical><delete>$179,062</delete><mark>$180,367</mark></numerical>, "
                + "with <numerical><delete>$144,618</delete><mark>$146,507</mark></numerical> being from The Prudential Insurance Company of America and " 
                + "<numerical><delete>$34,444</delete><mark>$33,860</mark></numerical> from the unaffiliated life insurance company.\n"

               + "Paragraph: The one-percentage-point increase of effect on total of service and interest cost components is $7,367. To calculate the percentage of this "
                + "amount in relation to the effect on other postretirement benefit obligation, we <relation><delete>divide</delete><mark>substract</mark></relation> $7,367 by $72,238 and "
                + "<relation><delete>multiply</delete><mark>add</mark></relation> the result by 100.\n\n($7,367 / $72,238) * 100 = 10.19%\n\n"
                + "Therefore, the one-percentage-point increase in effect on total of service and interest cost components is 10.19% of the effect on other postretirement benefit obligation.\n"

                + "Edited: The one-percentage-point increase of effect on total of service and interest cost components is $7,367. To calculate the percentage of this "
                + "amount in relation to the effect on other postretirement benefit obligation, we divide $7,367 by $72,238 and "
                + "<relation><delete>multiply</delete><mark>add</mark></relation> the result by 100.\n\n($7,367 / $72,238) * 100 = <numerical><delete>10.19%</delete><mark>23.57%</mark></numerical>\n\n"
                + "Therefore, the one-percentage-point increase in effect on total of service and interest cost components is <numerical><delete>10.19%</delete><mark>23.57%</mark></numerical> of the effect on other postretirement benefit obligation.\n"
                + "Below is the paragraph to edit:\n"
                + "Paragraph: " + passage + "\n"
                + "Return valid JSON in the following format:"
                + "{Edited: paragraph with inserted errors}"}]

    content = call_llm(client, model_name, messages, temperature, max_tokens)

    return content


### create temporal errors
def create_temporal_error(passage, model_name, max_tokens, temperature):

    messages=[
                {"role": "user", "content": "Given a passage with possibly already inserted error tokens wrapped in <numerical>, <entity>, <relation>, "
                + "<contradictory>, or <unverifiable>, insert temporal errors "
                + "in the passage below, wrapped in tokens to make the passage factually incorrect. Ensure these insertions are outside these existing <> tags, and don't modify the <> tags at all. "
                + "The error is defined as such:\n"
                + "1. temporal errors (<temporal>): incorrect reference or use of figures from the wrong time period. These errors typically "  
                + "arise from misinterpreting the reference's temporal context, such as year-over-year comparisons or quarter-specific data.\n" 

                + "Example 1: Apple reported record quarterly revenue of $123.9 billion in <temporal><delete>Q1</delete><mark>Q3</mark></temporal> of fiscal year 2022.\n" 
                + "Example 2: As of <temporal><delete>November 28, 2008</delete><mark>January 12, 2009</mark></temporal>, the gross liability for unrecognized tax benefits was $139.5 million.\n" 
                + "Example 3: In <temporal><delete>2017</delete><mark>2019</mark></temporal>, Entergy Mississippi's net income increased by $0.8 million compared to <temporal><delete>2016</delete><mark>2017</mark></temporal>, " 
                + "primarily due to higher other income and lower operation and maintenance expenses.\n"

                + "Now, insert temporal error tokens in the given passage but make sure that you don't modify "
                + "anything inside any already existing <> error tokens, only add temporal errors with <temporal>"
                + "</temporal> tokens outside the already existing <> tags. Also, avoid inserting errors in the "
                + "first sentence. Also make sure to tag every single edit with <mark></mark>, <delete></delete> "
                + "and <temporal><temporal> tags like done in the examples below.\n##\n"

                + "Paragraph: The variation between the capital expenditures on a GAAP basis and the one on a non-GAAP basis " 
                + "in the year 2014 was $202.9 ($1885.1 - $1682.2).\n"

                + "Edited: The variation between the capital expenditures on a GAAP basis and the one on a non-GAAP basis " 
                + "in the year <temporal><delete>2014</delete><mark>2013</mark></temporal> was $202.9 ($1885.1 - $1682.2).\n"

                + "Paragraph: The annual interest expense for entergy louisiana incurred from the series first mortgage bonds " 
                + "due September 2018 is $19.5 million. <contradictory>the net proceeds of the issuance will not be used for capital expenditures, " 
                + "working capital needs, or general corporate purposes.</contradictory> This is calculated by taking $300 million (the amount of bonds issued) "  
                + "multiplied by 6.50% (the interest rate) which equals $19.5 million.\n"

                + "Edited: The annual interest expense for entergy louisiana incurred from the series first mortgage bonds " 
                + "due <temporal><delete>September 2018</delete><mark>August 2008</mark></temporal> is $19.5 million. <contradictory>the net proceeds of the issuance will not be used for capital expenditures, " 
                + "working capital needs, or general corporate purposes.</contradictory> This is calculated by taking $300 million (the amount of bonds issued) "  
                + "multiplied by 6.50% (the interest rate) which equals $19.5 million.\n"

                + "Below is the paragraph to edit:\n"
                + "Paragraph: " + passage + "\n"
                + "Return valid JSON in the following format:"
                + "{Edited: paragraph with inserted errors}"}]

    content = call_llm(client, model_name, messages, temperature, max_tokens)

    return content


### create entity errors
def create_entity_error(passage, model_name, max_tokens, temperature):

    messages=[
                {"role": "user", "content": "Given a passage with possibly already inserted error tokens wrapped in <numerical>, <temporal>, <relation>, "
                + "<contradictory>, or <unverifiable>, insert entity errors "
                + "in the passage below, wrapped in tokens to make the passage factually incorrect. Ensure these insertions are outside these existing <> tags, and don't modify the <> tags at all. "
                + "The error is defined as such:\n"
                + "1. entity errors (<entity>): a small part of a sentence, often an entity (e.g., location name), is "
                + "incorrect (usually 1-3 words). Entity errors often involve noun phrases or nouns.\n" 

                + "Example 1: Amazon opened its new fulfillment center in <entity><delete>Nashville, Tennessee</delete><mark>San Francisco, California</mark></entity>, creating over 5,000 jobs.\n" 
                + "Example 2: The working capital accounts were a use of cash of $265.4, with <entity><delete>payables and accrued liabilities</delete><mark>depreciation and amortization</mark></entity>, "
                + "<entity><delete>inventories</delete><mark>deferred income taxes</mark></entity>, and trade receivables being the main drivers.\n"
                + "Example 3: Verizon <entity><delete>Wireless</delete><mark>Media</mark></entity> generated $91.7 billion in operating revenue in 2019.\n"

                + "Now, insert entity error tokens in the given passage but make sure that you don't modify "
                + "anything inside any already existing <> error tokens, only add entity errors with <entity>"
                + "</entity> tokens outside the already existing <> tags. Also, avoid inserting errors in the "
                + "first sentence. Also make sure to tag every single edit with <mark></mark>, <delete></delete> "
                + "and <entity></entity> tags like done in the examples below.\n##\n"

               + "Paragraph: The total possible purchase price for Impella, including potential contingent payments, " 
                + "is $45.1 million for the initial purchase price, plus up to an additional $11.2 million in contingent payments " 
                + "based on milestone achievements related to FDA approvals, totaling $56.3 million.\n"

                + "Edited: The total possible purchase price for Impella, including potential contingent payments, " 
                + "is $45.1 million for the <entity><delete>initial purchase price</delete><mark>aggregate purchase price</mark></entity>, plus up to an additional $11.2 million in contingent payments " 
                + "based on milestone achievements related to FDA approvals, totaling $56.3 million.\n"

               + "Paragraph: The one-percentage-point increase of effect on total of service and interest cost components is $7,367. To calculate the percentage of this "
                + "amount in relation to the effect on other postretirement benefit obligation, we <relation><delete>divide</delete><mark>substract</mark></relation> $7,367 by $72,238 and "
                + "<relation><delete>multiply</delete><mark>add</mark></relation> the result by 100.\n\n($7,367 / $72,238) * 100 = 10.19%\n\n"
                + "Therefore, the one-percentage-point increase in effect on total of service and interest cost components is 10.19% of the effect on other postretirement benefit obligation. "
                + "<contradictory>The one-percentage-point increase had no effect on benefit costs.</contradictory>\n"

                + "Edited: The one-percentage-point increase of effect on total of <entity><delete>service and interest cost</delete><mark>plan asset</mark></entity> components is $7,367. To calculate the percentage of this "
                + "amount in relation to the effect on other postretirement benefit obligation, we divide $7,367 by $72,238 and "
                + "<relation><delete>multiply</delete><mark>add</mark></relation> the result by 100.\n\n($7,367 / $72,238) * 100 = 10.19%\n\n"
                + "Therefore, the one-percentage-point increase in effect on total of service and interest cost components is 10.19% of the effect on other postretirement benefit obligation."
                + "<contradictory>The one-percentage-point increase had no effect on benefit costs.</contradictory>\n"
                + "Below is the paragraph to edit:\n"
                + "Paragraph: " + passage + "\n"
                + "Return valid JSON in the following format:"
                + "{Edited: paragraph with inserted errors}"}]

    content = call_llm(client, model_name, messages, temperature, max_tokens)

    return content


### create relation errors
def create_relation_error(passage, model_name, max_tokens, temperature):

    messages=[
                {"role": "user", "content": "Given a passage with possibly already inserted error tokens wrapped in <numerical>, <temporal>, <entity>, "
                + "<contradictory>, or <unverifiable>, insert relation errors, "
                + "outside the already inserted tokens without modifying the content within already existing tokens. Wrap "
                + "the relational errors in tokens to make the passage factually incorrect. The error is defined as such:\n"
                + "1. relational error (<relation>): a sentence is partially incorrect as a small part (usually 1 - 3 words). "
                + "Relational errors often involve verbs and are often the opposite of what it should be.\n"
                + "Example 1: The company’s debt-to-equity ratio in 2022 was <relation><delete>lower</delete><mark>higher</mark></relation> than in 2021, indicating improved leverage control.\n "
                + "Example 2: The decrease in net sales was <relation><delete>correlated with a drop</delete><mark>caused by an increase</mark></relation> in consumer demand.\n" 
                + "Example 3: EPS rose to $2.15 in Q4 2022, <relation><delete>after recovering from</delete><mark>before dropping due to</mark></relation> a one-time charge in Q3.\n"

                + "Now, insert relation error tokens in the given passage but make sure that you don't modify "
                + "anything inside any already existing <> error tokens, only add relational errors with "
                + "<relation></relation> tokens outside the already existing <> tags. Also avoid inserting "
                + "errors in the first sentence. Also make sure to tag every single edit with <mark></mark>, "
                + "<delete></delete> and <relation></relation> tags like done in the examples below:\n##\n"

                + "Paragraph: EMEA accounted for 0.76% of the total operating income in 2015. This can be calculated by dividing "
                + "the operating income of EMEA ($3.1 million) by the total operating income ($408.5 million) "
                + "and then multiplying by 100 to get the percentage.\n\n($3.1 million / $408.5 million) x 100 = 0.76%\n"

                + "Edited: EMEA accounted for 0.76% of the total operating income in 2015. This can be calculated by <relation><delete>dividing</delete><mark>multiplying</mark></relation> "
                + "the operating income of EMEA ($3.1 million) by the total operating income ($408.5 million) "
                + "and then multiplying by 100 to get the percentage.\n\n($3.1 million / $408.5 million) x 100 = 0.76%\n"

                + "Paragraph: The one-percentage-point increase of effect on total of <entity><delete>service and interest cost</delete><mark>plan asset</mark></entity> components is <entity><delete>$7,367</delete><mark>$72238</mark></entity>. To calculate the percentage of this "
                + "amount in relation to the effect on other postretirement benefit obligation, we divide $7,367 by $72,238 and "
                + "multiply the result by 100.\n\n($7,367 / $72,238) * 100 = 10.19%\n\n"
                + "Therefore, the one-percentage-point increase in effect on total of service and interest cost components is 10.19% of the effect on other postretirement benefit obligation.\n"

                + "Edited: The one-percentage-point increase of effect on total of <entity><delete>service and interest cost</delete><mark>plan asset</mark></entity> components is <entity><delete>$7,367</delete><mark>$72238</mark></entity>. To calculate the percentage of this "
                + "amount <relation><delete>in relation to</delete><mark>irrelevant to</mark></relation> the effect on other postretirement benefit obligation, we divide $7,367 by $72,238 and "
                + "multiply the result by 100.\n\n($7,367 / $72,238) * 100 = 10.19%\n\n"
                + "Therefore, the one-percentage-point increase in effect on total of service and interest cost components is 10.19% of the effect on other postretirement benefit obligation.\n"
                + "Below is the paragraph to edit:\n"
                + "Paragraph: " + passage + "\n"
                + "Return valid JSON in the following format:"
                + "{Edited:: paragraph with inserted errors}"}]


    content = call_llm(client, model_name, messages, temperature, max_tokens)

    return content


### create contradictory errors
def create_contradictory_error(reference, passage, model_name, max_tokens, temperature):

    messages=[
                {"role": "user", "content": "Given a reference and a passage with possibly already inserted error "
                + "tokens wrapped in <numerical>, <temporal>, <entity>, <relation>, or <unverifiable>, insert "
                + "contradictory sentence errors in the passage outside the already inserted tokens without modifying the content within already existing tokens. Wrap the inserted errors in tokens to make the passage "
                + "factually incorrect.  The contradictory error is defined as such:\n"
                + "1. contradictory sentence error (<contradictory>): a sentence where the entire sentence is contradicted "
                + "by the given reference, meaning the sentence can be proven false due to a contradiction with information in the reference provided.\n##\n"

                + "Example 1:\nReference: Stock-based compensation in 2019 includes stock options, restricted stock units (RSUs), " 
                + "and grants of contingent shares tied to total shareholder return.\n"
                + "Contradictory Sentence: <contradictory>The company’s 2019 stock-based compensation did not include any restricted stock units or " 
                + "contingent shares.</contradictory>\n"
                + "Explanation: This sentence contradicts the reference, which explicitly lists RSUs and contingent shares as part of the compensation.\n"

                + "Example 2:\nReference: In Q4, the firm repaid $500 million of long-term debt ahead of schedule to reduce interest expenses.\n"
                + "Contradictory Sentence: <contradictory>The firm took on an additional $500 million in long-term debt in Q4 to expand operations.</contradictory>\n"
                + "Explanation: This reverses the direction of the financial action: debt repayment vs. debt acquisition.\n"

                + "Example 3:\nReference: Higher margins and cost reductions were the primary drivers of the company's profit growth.\n"
                + "Contradictory Sentence: <contradictory>The company's profit growth was driven by increasing operational costs and margin compression.</contradictory>\n"  
                + "Explanation: The contradiction stems from attributing profit growth to negative performance factors.\n"

                + "Now, insert contradictory sentences with tokens in the given passage but make sure that you "
                + "don't modify anything inside any already existing <> error tokens at all, keep those untouched, only insert new contradictory "
                + "sentences (entire sentences) with <contradictory></contradictory> tokens outside the already "
                + "existing <> tags in the passage. Also avoid inserting errors before the first sentence. Also make sure you tag "
                + "each edit with <contradictory></contradictory> tags like done in the examples below:\n##\n"

                + "Reference: A.C. Cesena, commonly referred to as Cesena, was an Italian football club based in "
                + "Cesena, Emilia-Romagna. The club spent most of its history in professional leagues such as "
                + "Serie A and Serie B, but went bankrupt and folded in 2018. Another club from Cesena, A.S.D. "
                + "Romagna Centro Cesena, claims to be the bankrupted club's successor and in 2019 changed its "
                + "name to \"Cesena F.C.\"."

                + "\nPassage: A.C. Cesena was an Italian professional football club based "
                + "in Cesena, Emilia-Romagna. The club was founded in 1940 and was known by the nickname \"Bianconeri\" "
                + "(White and Blacks). Cesena has had a mixed history, having spent a significant amount of time in the "
                + "lower tiers of Italian football, but has also experienced some success in the Serie A. In its history, "
                + "Cesena has had a few notable achievements, including finishing seventh in the Serie A in the 1977-78 "
                + "season.\n"

                + "Edited: A.C. Cesena was an Italian professional football club based in Cesena, Emilia-Romagna. "
                + "The club was founded in 1940 and was known by the nickname ""Bianconeri"" (White and Blacks). "
                + "Cesena has had a mixed history, having spent a significant amount of time in the lower tiers of "
                + "Italian football, but has also experienced some success in the Serie A. <contradictory>Currently, "
                + "A.C. Cesena plays in Serie C, the third tier of Italian football.</contradictory> In its history, "
                + "Cesena has had a few notable achievements, including finishing seventh in the Serie A in the 1977-78 "
                + "season. <contradictory>It continues to compete in the hope of returning to the higher ranks of Italian "
                + "football</contradictory>.\n##\n"

                + "Reference: McPherson () is a city in and the county seat of McPherson County, Kansas, United States. "
                + "The city is named after Union General James Birdseye McPherson, a Civil War general. It is home to "
                + "McPherson College and Central Christian College.\n"

                + "Passage: McPherson is a city located in <entity><delete>McPherson County</delete><mark>Thomasville"
                + "</mark></entity>, Kansas, United States. <unverifiable>The town has Jabberjay birds.</unverifiable> "
                + "The town is named in honor of Union General James Birdseye McPherson who was a general during the "
                + "American Civil War.\n"

                + "Edited: McPherson is a city located in <entity><delete>McPherson County</delete><mark>"
                + "Thomasville</mark></entity>, Kansas, United States. <unverifiable>The town has Jabberjay "
                + "birds.</unverifiable> <contradictory>This city is home to just Stanford University.</contradictory> "
                + "The town is named in honor of Union General James Birdseye McPherson who "
                + "was a general during the American Civil War.\n##\n"

                + "Below is the reference and passge:\n"
                + "Reference: " + reference + "\n"
                + "Passage: " + passage + "\n"
                + "Return valid JSON in the following format:"
                + "{Edited:: edited passage with contradictory information to reference}"}]


    content = call_llm(client, model_name, messages, temperature, max_tokens)

    return content


### create unverifiable errors
def create_unverifiable_error(reference, passage, model_name, max_tokens, temperature):

    messages=[
                {"role": "user", "content": "Given a reference and a passage with possibly already inserted error tokens wrapped in <numerical>, <temporal>, <entity>, <relation>, or <contradictory>, " 
                + "insert unverifiable errors outside the already inserted tokens without modifying the content within already existing tokens. Wrap the insertions in tokens to make the passage factually incorrect. "
                + "The error is defined as such:\n"
                + "1. unverifiable sentence (<unverifiable>): statements or claims made that cannot be directly confirmed or refuted using the reference. These errors may not be obviously false, but they introduce " 
                + "information that lacks explicit support or contradiction within the reference, making them unverifiable or speculative from a grounding perspective.\n"

                + "Example 1:\nReference: A company’s quarterly filing states that its revenue grew 8% year-over-year due to increased demand in North America.\n"
                + "Unverifiable error: <unverifiable>The company's growth is likely to continue, driven by its strong leadership and innovative culture.</unverifiable>\n"
                + "Explanation: The statement about 'strong leadership and innovative culture' introduces speculative or subjective attributes not found in the reference.\n"

                + "Example 2:\nReference: [ 'Other Expense, Net [[\'\', \'Year Ended December 31,\', \'\', \'Change\', \'\'], [\'\', \'2018\', \'2017\', \'$\', \'%\'], [\'\', \'\', \'(dollars in thousands)\', \'\', \'\'], [\'Other expense, net\', \'$ 4,628\', \'$ 302\', \'$ 4,326\', \'1432.5%\'], " 
                " [\'% of revenue\', \'3%\', \'0%\', \'\', \'\']]', 'Other expense, net decreased by $4.3 million in 2018 compared to 2017 as a result of an increase in interest expense of $5.7 million related to interest expense due under our convertible senior notes. This increase was offset by an increase " 
                " of $1.4 million of interest income earned on our short-term investments.' ]\n"
                + "Unverifiable statement: <unverifiable>The increase in interest expense was primarily due to a significant rise in the company's debt obligations, which is expected to continue in future years.</unverifiable>\n"
                + "Explanation: The reference only mentions the cause of the increase as related to interest expense on convertible senior notes but does not provide further details about the company's future debt obligations or projections.\n"

                + "Example 3:\nReference: [ 'NOTE 17. FINANCIAL INSTRUMENTS AND FINANCIAL RISK MANAGEMENT [[\'\', \'December 31,\', \'\'], [\'\', \'2018\', \'2019\'], [\'Financial assets:\', \'\', \'\'], [\'Cash and cash equivalents\', \'285,907\', \'497,874\'], [\'Accounts receivable\', \'173,450\', \'199,535\'], "   
                + "[\'Financial liabilities:\', \'\', \'\'], [\'Accounts payable\', \'80,640\', \'119,712\']]', 'FINANCIAL INSTRUMENTS', 'Financial instruments include:', 'The carrying amounts of cash and cash equivalents, accounts receivable and accounts payable equal their fair values because of the short-term nature of these instruments.' ]\n"
                + "Unverifiable error: The company’s net financial position in 2019 is $577,697, calculated as total financial assets of $697,409 minus total financial liabilities of $119,712, <unverifiable>indicating a 15% year-over-year improvement in financial health.</unverifiable>\n"
                + "Explanation: The explanation includes 'international debt obligations and rising inflation, which are not mentioned in the reference.\n"

                + "Now, insert unverifiable error tokens in the given passage but make sure that you don't modify anything inside any already existing <> error tokens at all, keep those untouched, only insert unverifiable sentences or phrases with <unverifiable></unverifiable> tokens outside the already existing <> tags in the given passage. Remember, unverifiable sentences seem like they are true but cannot be confirmed or denied. Also avoid inserting errors before the first sentence. Also make sure you tag each edit with <unverifiable></unverifiable> tags like done in the examples below:\n"

                + "##\nReference: [ 'Revenue, Net [[\'\', \'Year Ended December 31,\', \'\', \'Change\', \'\'], [\'\', \'2020\', \'2019\', \'$\', \'%\'], [\'\', \'\', \'(dollars in thousands)\', \'\', \'\'], [\'Revenue, net\', \'$ 10,500\', \'$ 9,800\', \'$ 700\', \'7.14%\'], " 
                + "[\'% of total sales\', \'15%\ ', \'13%\ ', \'\', \'\']]', 'Revenue increased by $700,000 in 2020 compared to 2019, reflecting higher sales in our core product lines, as well as an expansion into international markets.' ]\n"
                + "Passage: Revenue increased by $700,000 in 2020 compared to 2019. This increase was largely driven by a surge in demand for our products in emerging markets, which is expected to continue in the coming years.\n"
                + "Edited: Revenue increased by $700,000 in 2020 compared to 2019. <unverifiable>This increase was largely driven by a surge in demand for our products in emerging markets, which is expected to continue in the coming years.</unverifiable>\n"

                + "##\nReference: [ 'state street corporation | 52 shareholder return performance presentation the graph presented below compares the cumulative total shareholder return on state street's common stock to the cumulative total return of the s&p 500 index , the s&p financial index and the kbw bank index over a five-year period . " 
                + "the cumulative total shareholder return assumes the investment of $ 100 in state street common stock and in each index on december 31 , 2012 . it also assumes reinvestment of common stock dividends . the s&p financial index is a publicly available , capitalization-weighted index , comprised of 67 of the standard & poor 2019s 500 companies , " 
                + "representing 27 diversified financial services companies , 23 insurance companies , and 17 banking companies . the kbw bank index is a modified cap-weighted index consisting of 24 exchange-listed stocks , representing national money center banks and leading regional institutions. .', '[[\'\', \'2012\', \'2013\', \'2014\', \'2015\', \'2016\', \'2017\'], " 
                + "[\"state street corporation\', \'$ 100\', \'$ 159\', \'$ 172\', \'$ 148\', \'$ 178\', \'$ 227\'], [\'s&p 500 index\', \'100\', \'132\', \'151\', \'153\', \'171\', \'208\'], [\'s&p financial index\', \'100\', \'136\', \'156\', \'154\', \'189\', \'230\'], [\'kbw bank index\', \'100\', \'138\', \'151\', \'151\', \'195\', \'231\']]' ]\n"
                + "Passage: The S&P 500 Index increased from 100 in 2011 to 208 in 2016, under the same investment assumptions. In 2015, State Street’s cumulative return declined to 153, reflecting a temporary underperformance relative to the broader market. State Street’s 2015 downturn may have been influenced by sector-specific challenges, such as regulatory pressures or declining client demand.\n"
                + "Edited: The S&P 500 Index increased from 100 in <temporal><delete>2011</delete><mark>2012</mark></temporal> to 208 in <temporal><delete>2016</delete><mark>2017</mark></temporal>, under the same investment assumptions. In 2015, State Street’s cumulative return declined to 153, reflecting a temporary underperformance relative to the broader market. <unverifiable>State Street’s 2015 downturn may have been influenced by sector-specific challenges, such as regulatory pressures or declining client demand.</unverifiable>\n"

                + "Below is the reference and passge:\n"
                + "Reference: " + reference + "\n"
                + "Passage: " + passage + "\n"
                + "Return valid JSON in the following format:"
                + "{Edited:: edited passage with contradictory information to reference}"}]


    content = call_llm(client, model_name, messages, temperature, max_tokens)

    return content


def get_num_tokens(text):
    # Choose a tokenizer, e.g., for gpt-3.5 or gpt-4
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    num_tokens = len(encoding.encode(text))
    return num_tokens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input .csv file to generate training data")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output .csv file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma2-9b-it",
        help="model name for generations")
    parser.add_argument(
        "--max_tokens",
        type=int32,
        default=512,
        help="max_token for model")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="temperature for model")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for generations")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ## Load Data
    df = pd.read_csv(args.input_file)

    ## Load model
    if args.model_name == "gemma2-9b-it":
        client = Groq(max_retries=5)
    else:
        client = OpenAI()

    client.api_key = args.api_key


    ### Create all errors
    error_tags = ["entity", "numerical", "temporal", "relation", "relation", "contradictory", "unverifiable"]

    response_w_tags = []

    for i, row in df.iterrows():
        
        reference = ast.literal_eval(row['documents'])[0]
        passage = row['response']

        random.shuffle(error_tags)
        if get_num_tokens(passage)<50:
            N = 1
        elif get_num_tokens(passage)>=50 and get_num_tokens(passage)<=200:
            N = 2
        else:
            N = 3
        selected_tags = error_tags[0:N]
    
        for err in selected_tags: 
            if err == "numerical":
                passage = create_numerical_error(passage, args.model_name, args.max_tokens, args.temperature)
            elif err == "temporal":
                passage = create_temporal_error(passage, args.model_name, args.max_tokens, args.temperature)
            elif err == "entity":
                passage = create_entity_error(passage, args.model_name, args.max_tokens, args.temperature)
            elif err == "relation":
                passage = create_relation_error(passage, args.model_name, args.max_tokens, args.temperature)
            elif err == "contradictory":
                passage = create_contradictory_error(reference, passage, args.model_name, args.max_tokens, args.temperature)
            elif err == "unverifiable":
                passage = create_unverifiable_error(reference, passage, args.model_name, args.max_tokens, args.temperature)

        response_w_tags.append(passage)


    df['response_w_tags'] = response_w_tags
    df.to_csv(args.output_file)
