from factcheck import FactChecker

fc = FactChecker(index_path="index_full.pkl")

report = fc.check(
    question="I feel a lump on my right breast, what is it and should i get it checked out",
    answer="Breast cancer is not a previlant cancer for american women",
)
print(report.summary())
print("Total is: "+str(report.support_rate))       # e.g. 0.6 -> 60% of claims supported
print(report.flagged_claims)     # claims MiniCheck couldn't ground