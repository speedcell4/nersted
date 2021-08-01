import aku

from ner.nn.tagger import train_tagger

app = aku.Aku()

app.option(train_tagger)

app.run()
