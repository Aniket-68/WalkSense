# reasoning/fusion_reasoner.py

class FusionReasoner:
    """
    Answers user queries
    """

    def answer(self, query: str) -> str:
        query = query.lower()

        if "what is in front" in query:
            return "There is a clear path ahead."

        if "where am i" in query:
            return "You appear to be indoors."

        return "I did not understand your question."
