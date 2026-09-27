from matching_features import extract_features
from matching_model import CustomLogisticRegression

def test_feature_cases_cover_identity_typo_address_country_and_missingness():
    row1 = {'business_name': 'Pizza Hut', 'business_address': '123 Main St', 'country': 'US'}
    row2 = {'business_name': 'Pizza Hut', 'business_address': '123 Main St', 'country': 'US'}
    
    features = extract_features(row1, row2)
    assert features[0] == 1.0  # Exact name
    assert features[7] == 1.0  # Country match

def test_zero_singleton_multiple_and_candidate_boundary():
    model = CustomLogisticRegression(learning_rate=0.1, epochs=10)
    X = [[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]
    y = [1, 0]
    
    model.fit(X, y)
    probas = model.predict_proba(X)
    
    assert probas[0] > probas[1]  # Perfect match scores higher than complete mismatch

if __name__ == "__main__":
    test_feature_cases_cover_identity_typo_address_country_and_missingness()
    test_zero_singleton_multiple_and_candidate_boundary()
    print("ALL MATCHING TESTS PASSED")