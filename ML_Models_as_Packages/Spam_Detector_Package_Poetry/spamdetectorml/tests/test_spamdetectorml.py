from spamdetectorml import __version__
from spamdetectorml import TextClassifier


def test_version():
    assert __version__ == '0.0.1'

def test_is_spam():
    sd = TextClassifier()
    sd.text = 'Click now buy one get ten free'
    prediction = sd.predict()
    assert prediction == 'Is Spam'

def test_is_not_spam():
    sd = TextClassifier()
    sd.text = 'Great'
    prediction = sd.predict()
    assert prediction == 'Not Spam'