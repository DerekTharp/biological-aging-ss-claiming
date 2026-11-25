"""Statutory Social Security schedule by birth year: full retirement age,
delayed-retirement credit, and the benefit as a fraction of the primary
insurance amount (PIA) by claiming age.
Source: https://www.ssa.gov/oact/ProgData/ar_drc.html
"""

# Real discount rates for the present-value benchmark (benefits are indexed
# to prices): 1 to 3 percent, with 4 to 6 percent as sensitivities.
DISCOUNT_RATES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def fra_years(birth_year):
    """Full retirement age in years."""
    if birth_year <= 1937:
        return 65.0
    if birth_year <= 1942:                 # 1938-1942: two more months per year
        return 65.0 + (birth_year - 1937) * 2.0 / 12
    if birth_year <= 1954:
        return 66.0
    if birth_year <= 1959:                 # 1955-1959: two more months per year
        return 66.0 + (birth_year - 1954) * 2.0 / 12
    return 67.0


def drc_annual(birth_year):
    """Delayed-retirement credit per year of delay past full retirement age:
    3.0% for 1924 and earlier, rising half a point every two birth years to
    8.0% for 1943 and later."""
    by = int(birth_year)
    if by <= 1924:
        return 0.030
    if by >= 1943:
        return 0.080
    return {1925: 0.035, 1926: 0.035, 1927: 0.040, 1928: 0.040,
            1929: 0.045, 1930: 0.045, 1931: 0.050, 1932: 0.050,
            1933: 0.055, 1934: 0.055, 1935: 0.060, 1936: 0.060,
            1937: 0.065, 1938: 0.065, 1939: 0.070, 1940: 0.070,
            1941: 0.075, 1942: 0.075}[by]


def ss_multiplier(claim_age, birth_year):
    """Benefit as a fraction of PIA for a claim at ``claim_age`` (monthly adjustments)."""
    fra = fra_years(birth_year)
    if claim_age > fra:
        return 1.0 + (claim_age - fra) * drc_annual(birth_year)
    m = round((fra - claim_age) * 12)      # months claimed before full retirement age
    reduction = min(m, 36) * (5.0 / 9) / 100 + max(0, m - 36) * (5.0 / 12) / 100
    return 1.0 - reduction
