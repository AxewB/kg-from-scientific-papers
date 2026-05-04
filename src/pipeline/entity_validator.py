class EntityValidator:
    def __init__(self, ner_extractor):
        self.ner = ner_extractor

    def is_valid(self, text: str) -> bool:
        doc = self.ner.nlp(text)
        return len(doc.ents) > 0
