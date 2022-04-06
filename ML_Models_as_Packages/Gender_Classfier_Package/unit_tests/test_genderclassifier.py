from genderclassifier import __version__
from genderclassifier import GenderClassifier

def test_version():
    assert __version__ == '0.0.1'

def test_genderclassifier_for_male():
    g = GenderClassifier()
    g.name = 'Anthony'
    prediction = g.predict()
    assert prediction == 'Male'

def test_genderclassifier_for_female():
    g = GenderClassifier()
    g.name = 'Emily'
    prediction = g.predict()
    assert prediction == 'Female'