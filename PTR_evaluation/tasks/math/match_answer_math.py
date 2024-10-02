import re

def str_to_float(text: str):
    # convert string like '1,234.00' to float
    return float(text.replace(",", ""))


def match_answer_math(infer_result, round_idx, args):
    number_pattern = r'(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    math_data_pattern = "#### " + number_pattern
    exact_pattern = r'The answer is[:\s#\$]*\s*' + number_pattern
    flexible_pattern = number_pattern
    exact_match_cnt = 0
    flexible_match_cnt = 0 
    result = {}
    for item in infer_result["math"]:
        answer = str_to_float(re.findall(math_data_pattern, item["answer"])[0])
        # match answer after 'The answer is #### '
        exact_answer = re.findall(exact_pattern, item[f"infer_round{round_idx}"])
        item[f"judge{round_idx}"] = False
        if len(exact_answer) > 0:
            model_answer = str_to_float(exact_answer[0])
            item[f"exact_match{round_idx}"] = exact_answer
            if abs(model_answer - answer) < 1e-6:
                exact_match_cnt += 1
                item[f"judge{round_idx}"] = True
        else:
            item[f"exact_match{round_idx}"] = None
        
        # match every number in the response
        flexible_answers = re.findall(flexible_pattern, item[f"infer_round{round_idx}"])
        if len(flexible_answers) > 0:
            for num_str in flexible_answers:
                if abs(str_to_float(num_str) - answer) < 1e-6:
                    flexible_match_cnt += 1
                    break
        item[f"flexible_match{round_idx}"] = flexible_answers

    result["math"] = {
        "acc": exact_match_cnt / len(infer_result["math"]),  # exact match
        "flexible_match": flexible_match_cnt / len(infer_result["math"])
    }
    return result