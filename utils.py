# utils.py
def normalize_law_number(law_number: str) -> str:
    law_number = law_number.strip()
    if '.' in law_number:
        return law_number
    parts = law_number.split('/')
    if len(parts) != 2:
        return law_number
    num_part, year_part = parts
    if len(num_part) > 1:
        normalized = num_part[0] + '.' + num_part[1:] + '/' + year_part
        return normalized
    return law_number
