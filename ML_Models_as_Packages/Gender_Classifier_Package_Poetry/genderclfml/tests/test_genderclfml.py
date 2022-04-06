from genderclfml import __version__
from genderclfml import GenderClassifier

def test_version():
    assert __version__ == '0.1.0'

def test_genderclfml_for_male():
    g = GenderClassifier()
    g.name = 'Anthony'
    prediction = g.predict()
    assert prediction == 'Male'

def test_genderclfml_for_female():
    g = GenderClassifier()
    g.name = 'Emily'
    prediction = g.predict()
    assert prediction == 'Female'