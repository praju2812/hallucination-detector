import re
def norm(s):
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())
