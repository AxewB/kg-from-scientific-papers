import spacy


class SentenceSplitter:
  def __init__(self, model: str = "en_core_sci_sm"):
    self.nlp = spacy.load(model)
    self.nlp.disable_pipes("ner", "tagger", "lemmatizer", "attribute_ruler")

  def split(self, text: str) -> list[str]:
    doc = self.nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

  def split_sections(self, sections: list[dict]) -> list[str]:
    sentences = []

    for sec in sections:
      sentences.extend(self.split(sec["text"]))

    return sentences

  def split_batch(self, texts: list[str]) -> list[list[str]]:
    """process multiple texts"""
    results = []

    for doc in self.nlp.pipe(texts, batch_size=32):
      results.append([s.text.strip() for s in doc.sents])

    return results
