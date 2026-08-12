from deepmultilingualpunctuation import PunctuationModel
model = PunctuationModel()
text = "He seemed content with this small group in the villa some kind of loose star on the edge of their system. This is like a... for him after the war of mud, rivers, and bridges. villages. He enters the Hugg house only when invited in, just a... the way he had done that first night when he had followed the faltering sound of Hana's piano and come up the cypress-lined path and sti. stepped into the library."
result = model.restore_punctuation(text)
print(result)