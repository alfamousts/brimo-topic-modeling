from nltk.tag import CRFTagger
ct = CRFTagger()
ct.set_model_file('model/all_indo_man_tag_corpus_model.crf.tagger')
hasil = ct.tag_sents([['Saya','bekerja','di','Bandung']])